# momentum

Backend server implementation for Momentum Coach.

## Files

- `app/backend/server.py` — FastAPI backend with auth, goals, tasks, check-ins, insights, and Stripe payments.

## Notes

- Secure cookie usage is controlled by `COOKIE_SECURE`.
- `CORS_ORIGINS` should be configured as a comma-separated list in production.
- Timestamps are stored as native UTC datetimes.
