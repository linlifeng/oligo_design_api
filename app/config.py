import os

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "default-secret")
    FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")

