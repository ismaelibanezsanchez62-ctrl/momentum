import uuid
from dataclasses import dataclass

@dataclass
class CheckoutSessionRequest:
    amount: float
    currency: str
    success_url: str
    cancel_url: str
    metadata: dict

@dataclass
class CheckoutSessionResponse:
    session_id: str
    url: str

class StripeCheckout:
    def __init__(self, api_key: str, webhook_url: str):
        self.api_key = api_key
        self.webhook_url = webhook_url

    async def create_checkout_session(self, req: CheckoutSessionRequest) -> CheckoutSessionResponse:
        session_id = str(uuid.uuid4())
        return CheckoutSessionResponse(
            session_id=session_id,
            url=f"https://example.com/checkout/{session_id}",
        )

    async def get_checkout_status(self, session_id: str):
        class Status:
            def __init__(self):
                self.status = "open"
                self.payment_status = "pending"
        return Status()

    async def handle_webhook(self, body: bytes, signature: str):
        class Event:
            def __init__(self):
                self.payment_status = "paid"
                self.session_id = None
                self.metadata = {}
        return Event()
