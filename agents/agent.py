import requests
import json
import time
import smtplib
from email.message import EmailMessage
from datetime import datetime

API_URL = "https://your-api-url.com/signals"
TICKERS_FILE = "tickers.json"
HISTORY_FILE = "signal_history.json"
CHECK_INTERVAL = 3600  # in seconds (1 hour)


def load_tickers():
    with open(TICKERS_FILE) as f:
        return json.load(f)['tickers']


def load_history():
    try:
        with open(HISTORY_FILE) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def save_history(history):
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)


def send_email_notification(subject, body):
    import os

    EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
    EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

    TO_EMAIL = "linlifeng@gmail.com"

    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = TO_EMAIL
    msg.set_content(body)

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        smtp.send_message(msg)


def check_signals():
    tickers = load_tickers()
    history = load_history()

    ticker_param = ','.join(tickers)
    response = requests.get(API_URL, params={'tickers': ticker_param})
    result = response.json()

    new_signals = []

    for ticker, signals in result.items():
        if "error" in signals:
            continue

        prev = history.get(ticker, {"buy_signals": [], "sell_signals": []})
        new_buys = [d for d in signals['buy_signals'] if d not in prev['buy_signals']]
        new_sells = [d for d in signals['sell_signals'] if d not in prev['sell_signals']]

        if new_buys or new_sells:
            new_signals.append((ticker, new_buys, new_sells))

        history[ticker] = signals

    if new_signals:
        body = "New trading signals detected:\n\n"
        for ticker, buys, sells in new_signals:
            if buys:
                body += f"[{ticker}] BUY on: {', '.join(buys)}\n"
            if sells:
                body += f"[{ticker}] SELL on: {', '.join(sells)}\n"

        print(body)
        send_email_notification("Stock Signal Alert", body)

    save_history(history)


def run_agent():
    while True:
        print(f"[{datetime.now()}] Checking stock signals...")
        try:
            check_signals()
        except Exception as e:
            print(f"Error: {e}")
        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    run_agent()
