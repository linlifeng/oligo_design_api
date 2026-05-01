# app/routes/stocks.py
from flask import Blueprint, request, jsonify
import logging
import yfinance as yf
import json

stocks_bp = Blueprint('stocks', __name__)

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
        # yfinance occasionally returns empty frames from one path; try both APIs.
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
            return jsonify({'error': f'No data returned for {symbol}'}), 404

        # yf.download can return multi-index columns depending on settings.
        if hasattr(hist.columns, 'nlevels') and hist.columns.nlevels > 1:
            hist.columns = hist.columns.get_level_values(0)

        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [c for c in required_cols if c not in hist.columns]
        if missing_cols:
            return jsonify({'error': f'Missing columns for {symbol}: {", ".join(missing_cols)}'}), 500

        hist = hist.sort_index(ascending=False)

        time_series = {}

        for date, row in hist.iterrows():
            date_str = date.strftime('%Y-%m-%d')
            time_series[date_str] = {
                "1. open": f"{row['Open']:.4f}",
                "2. high": f"{row['High']:.4f}",
                "3. low": f"{row['Low']:.4f}",
                "4. close": f"{row['Close']:.4f}",
                "5. volume": str(int(row['Volume']))
            }

        last_refreshed = max(time_series.keys())

        response = {
            "Meta Data": {
                "1. Information": "Daily Prices (open, high, low, close) and Volumes",
                "2. Symbol": symbol,
                "3. Last Refreshed": last_refreshed,
                "4. Output Size": "Compact",
                "5. Time Zone": "US/Eastern"
            },
            "Time Series (Daily)": time_series
        }

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
