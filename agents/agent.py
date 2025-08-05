import requests
import json
import time
import smtplib
from email.message import EmailMessage
from datetime import datetime, time as dt_time
import pytz
import os

# Configuration
API_BASE_URL = "https://api.oligodesign.com/api/stocks"
TICKERS_API_URL = f"{API_BASE_URL}/tickers"
SIGNALS_API_URL = f"{API_BASE_URL}/signals"
HISTORY_FILE = "signal_history.json"
CHECK_INTERVAL = 1800  # 30 minutes in seconds

# Trading hours (Eastern Time)
MARKET_OPEN = dt_time(9, 30)  # 9:30 AM ET
MARKET_CLOSE = dt_time(16, 0)  # 4:00 PM ET

def is_trading_hours():
    """Check if current time is within trading hours (Monday-Friday, 9:30 AM - 4:00 PM ET)"""
    et_tz = pytz.timezone('US/Eastern')
    now_et = datetime.now(et_tz)
    
    # Check if it's a weekday (Monday=0, Sunday=6)
    if now_et.weekday() >= 5:  # Saturday or Sunday
        return False
    
    current_time = now_et.time()
    return MARKET_OPEN <= current_time <= MARKET_CLOSE

def load_tickers():
    """Load tickers from the API"""
    try:
        response = requests.get(TICKERS_API_URL, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get('tickers', [])
    except requests.exceptions.RequestException as e:
        print(f"Error loading tickers from API: {e}")
        # Fallback to local file if API fails
        try:
            with open('tickers.json') as f:
                return json.load(f)['tickers']
        except FileNotFoundError:
            print("No local tickers file found. Using default tickers.")
            return ['AAPL', 'GOOGL', 'MSFT']  # Default tickers

def load_history():
    """Load signal history from file"""
    try:
        with open(HISTORY_FILE) as f:
            return json.load(f)
    except FileNotFoundError:
        print("No history file found. Starting fresh.")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error reading history file: {e}. Starting fresh.")
        return {}

def save_history(history):
    """Save signal history to file"""
    try:
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"Error saving history: {e}")

def send_email_notification(subject, body):
    """Send email notification"""
    try:
        EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
        EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
        TO_EMAIL = "linlifeng@gmail.com"

        if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
            print("Email credentials not found in environment variables")
            return False

        msg = EmailMessage()
        msg['Subject'] = subject
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = TO_EMAIL
        msg.set_content(body)

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)
        
        print("Email notification sent successfully")
        return True
        
    except Exception as e:
        print(f"Error sending email: {e}")
        return False

def check_signals():
    """Check for new trading signals"""
    print("Loading tickers...")
    tickers = load_tickers()
    
    if not tickers:
        print("No tickers found. Skipping signal check.")
        return
    
    print(f"Checking signals for {len(tickers)} tickers: {', '.join(tickers)}")
    
    history = load_history()
    
    # Make API request for signals
    ticker_param = ','.join(tickers)
    
    try:
        response = requests.get(SIGNALS_API_URL, params={'tickers': ticker_param}, timeout=30)
        response.raise_for_status()
        result = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching signals from API: {e}")
        return
    except json.JSONDecodeError as e:
        print(f"Error parsing API response: {e}")
        return

    new_signals = []
    today = datetime.now().strftime('%Y-%m-%d')

    for ticker, signals in result.items():
        if "error" in signals:
            print(f"Error for {ticker}: {signals['error']}")
            continue

        # Get previous signals for this ticker
        prev = history.get(ticker, {"buy_signals": [], "sell_signals": []})
        
        # Find new signals
        new_buys = [d for d in signals.get('buy_signals', []) if d not in prev.get('buy_signals', [])]
        new_sells = [d for d in signals.get('sell_signals', []) if d not in prev.get('sell_signals', [])]

        # Only include today's signals for immediate notification
        new_buys_today = [d for d in new_buys if d == today]
        new_sells_today = [d for d in new_sells if d == today]

        if new_buys_today or new_sells_today:
            new_signals.append((ticker, new_buys_today, new_sells_today))

        # Update history with all signals
        history[ticker] = {
            "buy_signals": signals.get('buy_signals', []),
            "sell_signals": signals.get('sell_signals', [])
        }

    # Send notification if there are new signals
    if new_signals:
        body = f"New trading signals detected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}:\n\n"
        
        for ticker, buys, sells in new_signals:
            if buys:
                body += f"🟢 [{ticker}] BUY signals on: {', '.join(buys)}\n"
            if sells:
                body += f"🔴 [{ticker}] SELL signals on: {', '.join(sells)}\n"
        
        body += f"\nTotal signals: {len(new_signals)} ticker(s)\n"
        body += f"Check time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S ET')}"
        
        print(body)
        send_email_notification("🚨 Stock Signal Alert", body)
    else:
        print("No new signals detected.")

    # Save updated history
    save_history(history)

def run_agent():
    """Main agent loop"""
    print("Stock Signal Agent starting...")
    print(f"Check interval: {CHECK_INTERVAL/60} minutes")
    print(f"Trading hours: {MARKET_OPEN} - {MARKET_CLOSE} ET, Monday-Friday")
    
    while True:
        current_time = datetime.now()
        print(f"\n[{current_time.strftime('%Y-%m-%d %H:%M:%S')}] Agent cycle starting...")
        
        if is_trading_hours():
            print("✅ Within trading hours - checking signals...")
            try:
                check_signals()
            except Exception as e:
                print(f"❌ Error during signal check: {e}")
        else:
            print("⏰ Outside trading hours - skipping signal check")
        
        print(f"💤 Sleeping for {CHECK_INTERVAL/60} minutes...")
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    # Verify environment variables
    if not os.getenv("EMAIL_ADDRESS") or not os.getenv("EMAIL_PASSWORD"):
        print("⚠️  Warning: EMAIL_ADDRESS and EMAIL_PASSWORD environment variables not set")
        print("Email notifications will not work without these credentials")
    
    try:
        run_agent()
    except KeyboardInterrupt:
        print("\n🛑 Agent stopped by user")
    except Exception as e:
        print(f"💥 Agent crashed with error: {e}")