# app/routes/stocks.py
from flask import Blueprint, request, jsonify
import logging
import yfinance as yf
import json
import csv
import io
from urllib.request import Request, urlopen

stocks_bp = Blueprint('stocks', __name__)


def _build_response_from_rows(symbol, rows):
    """Build API response shape from a normalized list of OHLCV rows."""
    if not rows:
        return None

    time_series = {}
    for row in rows:
        date_str = row['date']
        time_series[date_str] = {
            "1. open": f"{row['open']:.4f}",
            "2. high": f"{row['high']:.4f}",
            "3. low": f"{row['low']:.4f}",
            "4. close": f"{row['close']:.4f}",
            "5. volume": str(int(row['volume']))
        }

    last_refreshed = max(time_series.keys())
    return {
        "Meta Data": {
            "1. Information": "Daily Prices (open, high, low, close) and Volumes",
            "2. Symbol": symbol,
            "3. Last Refreshed": last_refreshed,
            "4. Output Size": "Compact",
            "5. Time Zone": "US/Eastern"
        },
        "Time Series (Daily)": time_series
    }


def _fetch_with_yfinance(symbol):
    """Primary data source: yfinance history(), then yf.download() fallback."""
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period='6mo', interval='1d', auto_adjust=False, actions=False)

    if hist.empty:
        hist = yf.download(
            symbol,
            period='6mo',
            interval='1d',
            auto_adjust=False,
            progress=False,
            threads=False,
        )

    if hist.empty:
        return None

    # yf.download can return multi-index columns depending on backend behavior.
    if hasattr(hist.columns, 'nlevels') and hist.columns.nlevels > 1:
        hist.columns = hist.columns.get_level_values(0)

    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if any(col not in hist.columns for col in required_cols):
        return None

    hist = hist.sort_index(ascending=False)
    rows = []
    for date, row in hist.iterrows():
        rows.append({
            'date': date.strftime('%Y-%m-%d'),
            'open': float(row['Open']),
            'high': float(row['High']),
            'low': float(row['Low']),
            'close': float(row['Close']),
            'volume': float(row['Volume']),
        })
    return rows


def _fetch_with_stooq(symbol):
    """Secondary source: Stooq CSV endpoint, useful when Yahoo returns empty data."""
    suffixes = ['', '.us']
    user_agent = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'

    for suffix in suffixes:
        stooq_symbol = f"{symbol.lower()}{suffix}"
        url = f"https://stooq.com/q/d/l/?s={stooq_symbol}&i=d"
        try:
            req = Request(url, headers={'User-Agent': user_agent})
            with urlopen(req, timeout=12) as resp:
                csv_text = resp.read().decode('utf-8', errors='ignore')

            reader = csv.DictReader(io.StringIO(csv_text))
            rows = []
            for r in reader:
                # Stooq returns "N/D" for unavailable rows.
                if not r.get('Date') or r.get('Close') in (None, '', 'N/D'):
                    continue
                rows.append({
                    'date': r['Date'],
                    'open': float(r['Open']),
                    'high': float(r['High']),
                    'low': float(r['Low']),
                    'close': float(r['Close']),
                    'volume': float(r.get('Volume') or 0),
                })

            if rows:
                rows.sort(key=lambda x: x['date'], reverse=True)
                return rows[:180]
        except Exception:
            continue

    return None

@stocks_bp.route('/signals', methods=['GET'])
def get_signals():
    tickers = request.args.get('tickers')
    if not tickers:
        return jsonify({"error": "No tickers provided."}), 400

    ticker_list = tickers.split(',')
    results = {}

    for ticker in ticker_list:
        logging.info(f'Processing ticker: {ticker}')
        try:
            data = yf.download(ticker, period='1mo', interval='1d')
            if data.empty:
                results[ticker] = {"error": "No data found."}
                continue

            # Calculate moving averages
            data['SMA_5'] = data['Close'].rolling(window=5).mean()
            data['SMA_20'] = data['Close'].rolling(window=20).mean()

            buy_signals = []
            sell_signals = []

            for i in range(1, len(data)):
                if data['SMA_5'].iloc[i] > data['SMA_20'].iloc[i] and data['SMA_5'].iloc[i - 1] <= data['SMA_20'].iloc[i - 1]:
                    buy_signals.append(data.index[i].strftime('%Y-%m-%d'))
                elif data['SMA_5'].iloc[i] < data['SMA_20'].iloc[i] and data['SMA_5'].iloc[i - 1] >= data['SMA_20'].iloc[i - 1]:
                    sell_signals.append(data.index[i].strftime('%Y-%m-%d'))

            results[ticker] = {
                "buy_signals": buy_signals,
                "sell_signals": sell_signals
            }

        except Exception as e:
            logging.error(f'Error processing ticker {ticker}: {e}')
            results[ticker] = {"error": str(e)}

    return jsonify(results)



@stocks_bp.route('/stock', methods=['GET'])
def get_stock_data():
    symbol = request.args.get('symbol', '').upper()

    if not symbol:
        return jsonify({'error': 'Missing stock symbol'}), 400

    try:
        rows = _fetch_with_yfinance(symbol)
        if not rows:
            rows = _fetch_with_stooq(symbol)

        if not rows:
            return jsonify({'error': 'No data returned'}), 404

        response = _build_response_from_rows(symbol, rows)
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500



import os

TICKERS_FILE = os.path.join(os.path.dirname(__file__), '../../tickers.json')


@stocks_bp.route('/tickers', methods=['GET', 'POST'])
def manage_tickers():
    if request.method == 'GET':
        try:
            with open(TICKERS_FILE, 'r') as f:
                tickers = json.load(f)
            return jsonify(tickers)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    elif request.method == 'POST':
        data = request.get_json()
        tickers = data.get('tickers')
        if not isinstance(tickers, list):
            return jsonify({'error': 'Tickers must be a list'}), 400

        try:
            with open(TICKERS_FILE, 'w') as f:
                json.dump({'tickers': tickers}, f, indent=2)
            return jsonify({'message': 'Tickers updated successfully'})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
