from dotenv import load_dotenv
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import bcrypt
import json
import jwt
import logging
import os
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional

from fastapi import FastAPI, APIRouter, HTTPException, Depends, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, EmailStr

from emergentintegrations.llm.chat import LlmChat, UserMessage
from emergentintegrations.payments.stripe.checkout import (
    StripeCheckout, CheckoutSessionResponse, CheckoutSessionRequest
)

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

env = os.environ
mongo_url = env.get("MONGO_URL")
if not mongo_url:
    raise RuntimeError("MONGO_URL is required")

db_name = env.get("DB_NAME")
if not db_name:
    raise RuntimeError("DB_NAME is required")

JWT_SECRET = env.get("JWT_SECRET")
if not JWT_SECRET:
    raise RuntimeError("JWT_SECRET is required")

EMERGENT_LLM_KEY = env.get("EMERGENT_LLM_KEY", "")
STRIPE_API_KEY = env.get("STRIPE_API_KEY", "")
PREMIUM_PRICE_USD = float(env.get("PREMIUM_PRICE_USD", "9.99"))
COOKIE_SECURE = env.get("COOKIE_SECURE", "true").lower() in ("1", "true", "yes")
CORS_ORIGINS = [o.strip() for o in env.get("CORS_ORIGINS", "").split(",") if o.strip()]

client = AsyncIOMotorClient(mongo_url)
db = client[db_name]

app = FastAPI()
api = APIRouter(prefix="/api")
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode(), password_hash.encode())
    except Exception:
        return False


def create_token(user_id: str, email: str, ttl_minutes: int = 60 * 24 * 7) -> str:
    payload = {
        "sub": user_id,
        "email": email,
        "exp": datetime.now(timezone.utc) + timedelta(minutes=ttl_minutes),
        "type": "access",
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


async def get_current_user(request: Request) -> dict:
    token = request.cookies.get("access_token")
    if not token:
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            token = auth[7:]

    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

    user = await db.users.find_one(
        {"id": payload["sub"]}, {"_id": 0, "password_hash": 0}
    )
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    if user.get("email") != payload.get("email"):
        raise HTTPException(status_code=401, detail="Invalid token payload")

    return user


def today_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


class RegisterIn(BaseModel):
    email: EmailStr
    password: str
    name: str


class LoginIn(BaseModel):
    email: EmailStr
    password: str


class GoalIn(BaseModel):
    title: str
    description: Optional[str] = ""
    deadline: Optional[str] = None


class CheckInIn(BaseModel):
    mood: Optional[str] = None
    note: Optional[str] = ""


class CheckoutIn(BaseModel):
    origin_url: str
    plan: str = "premium_monthly"


async def ai_breakdown_goal(goal_title: str, goal_description: str, user_history: dict):
    system = (
        "You are MOMENTUM, a no-nonsense behavioral coach for procrastinators. "
        "You break goals into TINY, immediately actionable micro-tasks (5-25 min). "
        "Concrete verbs, no fluff. Return STRICT JSON array of objects with: "
        "title, duration_minutes (5-30), difficulty (1-3), day_offset (0-7). "
        "5-10 tasks total."
    )
    chat = LlmChat(
        api_key=EMERGENT_LLM_KEY,
        session_id=f"goal-{uuid.uuid4()}",
        system_message=system,
    ).with_model("openai", "gpt-5.2")

    hint = " Keep tasks 5-10 min — user has been struggling." if user_history.get("recent_fail_rate", 0) > 0.5 else ""
    msg = UserMessage(
        text=(
            f"Goal: {goal_title}\n"
            f"Details: {goal_description or 'none'}.{hint}\n"
            "Return JSON array only."
        )
    )

    try:
        raw = (await chat.send_message(msg)).strip()
        if raw.startswith("```"):
            raw = raw.split("```", 2)[1]
            if raw.startswith("json"):
                raw = raw[4:]
        tasks = json.loads(raw.strip())
        result = []
        for i, t in enumerate(tasks[:12]):
            day_offset = t.get("day_offset")
            result.append(
                {
                    "title": str(t.get("title", "Take small action"))[:120],
                    "duration_minutes": int(t.get("duration_minutes", 15)),
                    "difficulty": int(t.get("difficulty", 2)),
                    "day_offset": int(day_offset) if day_offset is not None else int(i // 2),
                }
            )
        return result
    except Exception as e:
        logger.warning(f"AI breakdown failed: {e}")
        return [
            {
                "title": f"Spend 10 min on: {goal_title}",
                "duration_minutes": 10,
                "difficulty": 1,
                "day_offset": 0,
            },
            {
                "title": f"Identify smallest next step for: {goal_title}",
                "duration_minutes": 5,
                "difficulty": 1,
                "day_offset": 0,
            },
            {
                "title": f"20 focused min on: {goal_title}",
                "duration_minutes": 20,
                "difficulty": 2,
                "day_offset": 1,
            },
        ]


NUDGE_MESSAGES = [
    "Two minutes. Just start. Future you will thank you.",
    "Procrastination is a lie your brain tells you. Open the task.",
    "You don't need motivation. You need to begin.",
    "The hardest part is the first 60 seconds. Press Start.",
    "Small steps, every day. That's how momentum works.",
    "Done is better than perfect. Take the next tiny action.",
]


def append_query_params(url: str, params: dict) -> str:
    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query))
    query.update(params)
    return urlunparse(parsed._replace(query=urlencode(query)))


@api.post("/auth/register")
async def register(payload: RegisterIn, response: Response):
    email = payload.email.lower().strip()
    if await db.users.find_one({"email": email}):
        raise HTTPException(status_code=400, detail="Email already registered")

    user_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    await db.users.insert_one(
        {
            "id": user_id,
            "email": email,
            "name": payload.name.strip(),
            "password_hash": hash_password(payload.password),
            "is_premium": False,
            "streak": 0,
            "longest_streak": 0,
            "last_active_date": None,
            "created_at": now,
        }
    )

    token = create_token(user_id, email)
    response.set_cookie(
        "access_token",
        token,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite="lax",
        max_age=60 * 60 * 24 * 7,
        path="/",
    )
    return {
        "id": user_id,
        "email": email,
        "name": payload.name,
        "is_premium": False,
        "streak": 0,
        "token": token,
    }


@api.post("/auth/login")
async def login(payload: LoginIn, response: Response):
    email = payload.email.lower().strip()
    u = await db.users.find_one({"email": email})
    if not u or not verify_password(payload.password, u["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_token(u["id"], email)
    response.set_cookie(
        "access_token",
        token,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite="lax",
        max_age=60 * 60 * 24 * 7,
        path="/",
    )
    return {
        "id": u["id"],
        "email": u["email"],
        "name": u["name"],
        "is_premium": u.get("is_premium", False),
        "streak": u.get("streak", 0),
        "token": token,
    }


@api.post("/auth/logout")
async def logout(response: Response):
    response.delete_cookie("access_token", path="/")
    return {"ok": True}


@api.get("/auth/me")
async def me(user: dict = Depends(get_current_user)):
    return user


@api.post("/goals")
async def create_goal(payload: GoalIn, user: dict = Depends(get_current_user)):
    if not user.get("is_premium"):
        active = await db.goals.count_documents(
            {"user_id": user["id"], "archived": False}
        )
        if active >= 2:
            raise HTTPException(
                status_code=402,
                detail="Free tier limit: 2 active goals. Upgrade to Premium.",
            )

    total = await db.tasks.count_documents({"user_id": user["id"]})
    failed = await db.tasks.count_documents(
        {"user_id": user["id"], "status": "missed"}
    )
    rate = failed / total if total else 0

    gid = str(uuid.uuid4())
    micro_tasks = await ai_breakdown_goal(
        payload.title, payload.description or "", {"recent_fail_rate": rate}
    )
    now = datetime.now(timezone.utc)
    goal = {
        "id": gid,
        "user_id": user["id"],
        "title": payload.title,
        "description": payload.description or "",
        "deadline": payload.deadline,
        "archived": False,
        "created_at": now,
    }
    await db.goals.insert_one(goal)

    base = datetime.now(timezone.utc).date()
    docs = []
    for t in micro_tasks:
        docs.append(
            {
                "id": str(uuid.uuid4()),
                "user_id": user["id"],
                "goal_id": gid,
                "title": t["title"],
                "duration_minutes": t["duration_minutes"],
                "difficulty": t["difficulty"],
                "scheduled_date": (base + timedelta(days=t["day_offset"])).strftime("%Y-%m-%d"),
                "status": "pending",
                "started_at": None,
                "completed_at": None,
                "created_at": now,
            }
        )
    if docs:
        await db.tasks.insert_many(docs)

    return {"goal": goal, "tasks_created": len(docs)}


@api.get("/goals")
async def list_goals(user: dict = Depends(get_current_user)):
    goals = await db.goals.find({"user_id": user["id"]}, {"_id": 0}).sort("created_at", -1).to_list(100)
    for goal in goals:
        total = await db.tasks.count_documents({"goal_id": goal["id"]})
        done = await db.tasks.count_documents(
            {"goal_id": goal["id"], "status": "done"}
        )
        goal["progress"] = {
            "total": total,
            "done": done,
            "percent": int(done / total * 100) if total else 0,
        }
    return goals


@api.get("/goals/{goal_id}")
async def get_goal(goal_id: str, user: dict = Depends(get_current_user)):
    goal = await db.goals.find_one(
        {"id": goal_id, "user_id": user["id"]}, {"_id": 0}
    )
    if not goal:
        raise HTTPException(status_code=404, detail="Goal not found")

    tasks = await db.tasks.find(
        {"goal_id": goal_id}, {"_id": 0}
    ).sort("scheduled_date", 1).to_list(500)
    return {"goal": goal, "tasks": tasks}


@api.delete("/goals/{goal_id}")
async def archive_goal(goal_id: str, user: dict = Depends(get_current_user)):
    res = await db.goals.update_one(
        {"id": goal_id, "user_id": user["id"]},
        {"$set": {"archived": True}},
    )
    if not res.matched_count:
        raise HTTPException(status_code=404, detail="Goal not found")
    return {"ok": True}


@api.get("/today")
async def get_today(user: dict = Depends(get_current_user)):
    today = today_str()
    await db.tasks.update_many(
        {
            "user_id": user["id"],
            "status": {"$in": ["pending", "in_progress"]},
            "scheduled_date": {"$lt": today},
        },
        {"$set": {"status": "missed"}},
    )

    rows = await db.tasks.find(
        {
            "user_id": user["id"],
            "scheduled_date": today,
            "status": {"$in": ["pending", "in_progress", "done"]},
        },
        {"_id": 0},
    ).sort("difficulty", 1).to_list(50)

    pending = [t for t in rows if t["status"] != "done"][:3]
    done = [t for t in rows if t["status"] == "done"]
    nudge = (
        NUDGE_MESSAGES[(datetime.now(timezone.utc).hour + len(pending)) % len(NUDGE_MESSAGES)]
        if pending
        else None
    )
    return {
        "date": today,
        "active_tasks": pending,
        "completed_today": done,
        "nudge": nudge,
        "streak": user.get("streak", 0),
    }


@api.post("/tasks/{task_id}/start")
async def start_task(task_id: str, user: dict = Depends(get_current_user)):
    t = await db.tasks.find_one({"id": task_id, "user_id": user["id"]})
    if not t:
        raise HTTPException(status_code=404, detail="Task not found")

    await db.tasks.update_one(
        {"id": task_id, "user_id": user["id"]},
        {
            "$set": {
                "status": "in_progress",
                "started_at": datetime.now(timezone.utc),
            }
        },
    )
    return {"ok": True}


@api.post("/tasks/{task_id}/complete")
async def complete_task(task_id: str, user: dict = Depends(get_current_user)):
    t = await db.tasks.find_one({"id": task_id, "user_id": user["id"]})
    if not t:
        raise HTTPException(status_code=404, detail="Task not found")

    await db.tasks.update_one(
        {"id": task_id, "user_id": user["id"]},
        {
            "$set": {
                "status": "done",
                "completed_at": datetime.now(timezone.utc),
            }
        },
    )

    today = today_str()
    last = user.get("last_active_date")
    streak = user.get("streak", 0)
    longest = user.get("longest_streak", 0)
    if last != today:
        yesterday = (datetime.now(timezone.utc).date() - timedelta(days=1)).strftime("%Y-%m-%d")
        streak = streak + 1 if last == yesterday else 1
        longest = max(longest, streak)
        await db.users.update_one(
            {"id": user["id"]},
            {
                "$set": {
                    "streak": streak,
                    "longest_streak": longest,
                    "last_active_date": today,
                }
            },
        )

    return {"ok": True, "streak": streak}


@api.post("/tasks/{task_id}/skip")
async def skip_task(task_id: str, user: dict = Depends(get_current_user)):
    res = await db.tasks.update_one(
        {"id": task_id, "user_id": user["id"]},
        {"$set": {"status": "missed"}},
    )
    if not res.matched_count:
        raise HTTPException(status_code=404, detail="Task not found")

    cutoff = (datetime.now(timezone.utc).date() - timedelta(days=3)).strftime("%Y-%m-%d")
    recent = await db.tasks.count_documents(
        {
            "user_id": user["id"],
            "status": "missed",
            "scheduled_date": {"$gte": cutoff},
        }
    )
    if recent >= 3:
        await db.tasks.update_many(
            {
                "user_id": user["id"],
                "status": "pending",
                "duration_minutes": {"$gt": 10},
            },
            {"$set": {"duration_minutes": 10, "difficulty": 1}},
        )
    return {"ok": True, "auto_adjusted": recent >= 3}


@api.post("/checkin")
async def check_in(payload: CheckInIn, user: dict = Depends(get_current_user)):
    today = today_str()
    await db.checkins.update_one(
        {"user_id": user["id"], "date": today},
        {
            "$set": {
                "id": str(uuid.uuid4()),
                "user_id": user["id"],
                "date": today,
                "mood": payload.mood,
                "note": payload.note,
                "created_at": datetime.now(timezone.utc),
            }
        },
        upsert=True,
    )
    return {"ok": True}


@api.get("/insights")
async def insights(user: dict = Depends(get_current_user)):
    days = []
    for i in range(13, -1, -1):
        d = (datetime.now(timezone.utc).date() - timedelta(days=i)).strftime("%Y-%m-%d")
        total = await db.tasks.count_documents({"user_id": user["id"], "scheduled_date": d})
        done = await db.tasks.count_documents(
            {"user_id": user["id"], "scheduled_date": d, "status": "done"}
        )
        days.append(
            {
                "date": d,
                "total": total,
                "done": done,
                "rate": int(done / total * 100) if total else 0,
            }
        )

    total = await db.tasks.count_documents({"user_id": user["id"]})
    done = await db.tasks.count_documents(
        {"user_id": user["id"], "status": "done"}
    )
    missed = await db.tasks.count_documents(
        {"user_id": user["id"], "status": "missed"}
    )
    goals_count = await db.goals.count_documents(
        {"user_id": user["id"], "archived": False}
    )

    best_hour = None
    try:
        async for doc in db.tasks.aggregate(
            [
                {
                    "$match": {
                        "user_id": user["id"],
                        "status": "done",
                        "completed_at": {"$ne": None},
                    }
                },
                {
                    "$project": {
                        "hour": {"$hour": "$completed_at"}
                    }
                },
                {
                    "$group": {
                        "_id": "$hour",
                        "count": {"$sum": 1},
                    }
                },
                {"$sort": {"count": -1}},
                {"$limit": 1},
            ]
        ):
            best_hour = doc["_id"]
    except Exception:
        pass

    return {
        "totals": {"total": total, "done": done, "missed": missed, "goals": goals_count},
        "completion_rate": int(done / total * 100) if total else 0,
        "streak": user.get("streak", 0),
        "longest_streak": user.get("longest_streak", 0),
        "best_hour": best_hour,
        "last_14_days": days,
    }


@api.post("/payments/checkout")
async def create_checkout(
    payload: CheckoutIn, request: Request, user: dict = Depends(get_current_user)
):
    if user.get("is_premium"):
        raise HTTPException(status_code=400, detail="Already a premium member")

    origin = payload.origin_url.rstrip("/")
    host = str(request.base_url)
    stripe = StripeCheckout(
        api_key=STRIPE_API_KEY,
        webhook_url=f"{host}api/webhook/stripe",
    )
    success = append_query_params(f"{origin}/payment-success", {"session_id": "{CHECKOUT_SESSION_ID}"})
    cancel = f"{origin}/premium"
    metadata = {
        "user_id": user["id"],
        "plan": payload.plan,
        "user_email": user["email"],
    }

    req = CheckoutSessionRequest(
        amount=PREMIUM_PRICE_USD,
        currency="usd",
        success_url=success,
        cancel_url=cancel,
        metadata=metadata,
    )
    session: CheckoutSessionResponse = await stripe.create_checkout_session(req)
    await db.payment_transactions.insert_one(
        {
            "id": str(uuid.uuid4()),
            "session_id": session.session_id,
            "user_id": user["id"],
            "amount": PREMIUM_PRICE_USD,
            "currency": "usd",
            "metadata": metadata,
            "payment_status": "initiated",
            "status": "pending",
            "created_at": datetime.now(timezone.utc),
        }
    )
    return {"url": session.url, "session_id": session.session_id}


@api.get("/payments/status/{session_id}")
async def payment_status(
    session_id: str, request: Request, user: dict = Depends(get_current_user)
):
    txn = await db.payment_transactions.find_one(
        {"session_id": session_id, "user_id": user["id"]}, {"_id": 0}
    )
    if not txn:
        raise HTTPException(status_code=404, detail="Payment not found")

    if txn.get("payment_status") == "paid":
        return txn

    host = str(request.base_url)
    stripe = StripeCheckout(
        api_key=STRIPE_API_KEY,
        webhook_url=f"{host}api/webhook/stripe",
    )
    try:
        status = await stripe.get_checkout_status(session_id)
    except Exception as e:
        logger.warning(f"stripe status check failed: {e}")
        return txn

    update = {
        "status": status.status,
        "payment_status": status.payment_status,
        "updated_at": datetime.now(timezone.utc),
    }
    await db.payment_transactions.update_one(
        {"session_id": session_id}, {"$set": update}
    )
    if status.payment_status == "paid" and txn.get("payment_status") != "paid":
        await db.users.update_one(
            {"id": user["id"]},
            {
                "$set": {
                    "is_premium": True,
                    "premium_since": datetime.now(timezone.utc),
                }
            },
        )
    txn.update(update)
    return txn


@api.post("/webhook/stripe")
async def stripe_webhook(request: Request):
    body = await request.body()
    sig = request.headers.get("Stripe-Signature", "")
    host = str(request.base_url)
    stripe = StripeCheckout(
        api_key=STRIPE_API_KEY,
        webhook_url=f"{host}api/webhook/stripe",
    )

    try:
        event = await stripe.handle_webhook(body, sig)
    except Exception as e:
        logger.error(f"webhook error: {e}")
        raise HTTPException(status_code=400, detail="Invalid webhook payload")

    if event.payment_status == "paid" and event.session_id:
        txn = await db.payment_transactions.find_one({"session_id": event.session_id})
        if txn and txn.get("payment_status") != "paid":
            await db.payment_transactions.update_one(
                {"session_id": event.session_id},
                {"$set": {"payment_status": "paid", "status": "complete"}},
            )
            uid = (event.metadata or {}).get("user_id") or txn.get("user_id")
            if uid:
                await db.users.update_one(
                    {"id": uid},
                    {
                        "$set": {
                            "is_premium": True,
                            "premium_since": datetime.now(timezone.utc),
                        }
                    },
                )

    return {"received": True}


@api.get("/")
async def root():
    return {"app": "Momentum Coach", "ok": True}


app.include_router(api)

allow_credentials = bool(CORS_ORIGINS)
allow_origins = CORS_ORIGINS if CORS_ORIGINS else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_credentials=allow_credentials,
    allow_origins=allow_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    await db.users.create_index("email", unique=True)
    await db.tasks.create_index([("user_id", 1), ("scheduled_date", 1)])
    await db.goals.create_index([("user_id", 1), ("archived", 1)])
    admin_email = env.get("ADMIN_EMAIL", "admin@momentum.app")
    admin_pw = env.get("ADMIN_PASSWORD", "admin123")
    if not await db.users.find_one({"email": admin_email}):
        await db.users.insert_one(
            {
                "id": str(uuid.uuid4()),
                "email": admin_email,
                "name": "Admin",
                "password_hash": hash_password(admin_pw),
                "is_premium": True,
                "streak": 0,
                "longest_streak": 0,
                "last_active_date": None,
                "role": "admin",
                "created_at": datetime.now(timezone.utc),
            }
        )


@app.on_event("shutdown")
async def shutdown():
    client.close()
