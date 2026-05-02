# agents/signals.py
# Pure-Python port of the signal engine in linlifeng.net/sandbox/stock_checker.html
# Produces identical ENTER / WATCH / IGNORE verdicts to the browser UI.

def _process_api_data(time_series):
    """Convert raw /api/stocks/stock Time Series to sorted OHLCV list (last 60 days)."""
    data = []
    for date, day in time_series.items():
        data.append({
            'date': date,
            'open': float(day['1. open']),
            'high': float(day['2. high']),
            'low': float(day['3. low']),
            'close': float(day['4. close']),
            'volume': int(day['5. volume']),
        })
    data.sort(key=lambda d: d['date'])
    return data[-60:]


# ---------------------------------------------------------------------------
# Indicator calculations (1:1 port from JS)
# ---------------------------------------------------------------------------

def _calculate_sma(data, period):
    sma = []
    for i in range(period - 1, len(data)):
        total = sum(d['close'] for d in data[i - period + 1:i + 1])
        sma.append(total / period)
    return sma


def _calculate_ema(data, period):
    if not data:
        return []
    multiplier = 2 / (period + 1)
    ema = [None] * len(data)
    ema[0] = data[0]['close']
    for i in range(1, len(data)):
        ema[i] = data[i]['close'] * multiplier + ema[i - 1] * (1 - multiplier)
    return ema


def _calculate_macd(data):
    ema12 = _calculate_ema(data, 12)
    ema26 = _calculate_ema(data, 26)
    macd_line = [ema12[i] - ema26[i] for i in range(len(ema12))]
    macd_data = [{'close': v} for v in macd_line]
    signal_line = _calculate_ema(macd_data, 9)
    histogram = [macd_line[i] - signal_line[i] for i in range(len(macd_line))]
    return {'macdLine': macd_line, 'signalLine': signal_line, 'histogram': histogram}


def _calculate_rsi(data, period=14):
    gains, losses = [], []
    for i in range(1, len(data)):
        change = data[i]['close'] - data[i - 1]['close']
        gains.append(change if change > 0 else 0)
        losses.append(abs(change) if change < 0 else 0)
    rsi = []
    for i in range(period - 1, len(gains)):
        avg_gain = sum(gains[i - period + 1:i + 1]) / period
        avg_loss = sum(losses[i - period + 1:i + 1]) / period
        if avg_loss == 0:
            rsi.append(100)
        else:
            rs = avg_gain / avg_loss
            rsi.append(100 - (100 / (1 + rs)))
    return rsi


def _calculate_cci(data, period=20):
    typical = [(d['high'] + d['low'] + d['close']) / 3 for d in data]
    tp_data = [{'close': v} for v in typical]
    tp_sma = _calculate_sma(tp_data, period)
    cci = []
    for i in range(period - 1, len(data)):
        tp = typical[i]
        sma = tp_sma[i - (period - 1)]
        mean_dev = sum(abs(typical[i - j] - sma) for j in range(period)) / period
        cci.append((tp - sma) / (0.015 * mean_dev) if mean_dev != 0 else 0)
    return cci


def _calculate_ao(data, fast=5, slow=34):
    ao = []
    for i in range(slow - 1, len(data)):
        fast_sum = sum((data[j]['high'] + data[j]['low']) / 2 for j in range(i - fast + 1, i + 1))
        slow_sum = sum((data[j]['high'] + data[j]['low']) / 2 for j in range(i - slow + 1, i + 1))
        ao.append(fast_sum / fast - slow_sum / slow)
    return ao


def _calculate_true_range(data):
    tr = []
    for i in range(1, len(data)):
        h, l, pc = data[i]['high'], data[i]['low'], data[i - 1]['close']
        tr.append(max(h - l, abs(h - pc), abs(l - pc)))
    return tr


def _calculate_stochastic(data):
    highs = [d['high'] for d in data]
    lows = [d['low'] for d in data]
    highest, lowest = max(highs), min(lows)
    if highest == lowest:
        return 50
    return (data[-1]['close'] - lowest) / (highest - lowest) * 100


def _calculate_williams_r(data):
    highs = [d['high'] for d in data]
    lows = [d['low'] for d in data]
    highest, lowest = max(highs), min(lows)
    if highest == lowest:
        return -50
    return (highest - data[-1]['close']) / (highest - lowest) * -100


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _normalize(value, min_val, max_val):
    if max_val == min_val:
        return 0
    return max(-1, min(1, (value - min_val) / (max_val - min_val) * 2 - 1))


def _detect_market_regime(data):
    prices = [d['close'] for d in data[-50:]]
    tr = _calculate_true_range(data[-20:])
    avg_vol = sum(tr) / len(tr) if tr else 1
    current_vol = tr[-1] if tr else avg_vol
    trend = (prices[-1] - prices[0]) / prices[0]
    vol_ratio = current_vol / avg_vol if avg_vol else 1
    if abs(trend) > 0.1 and vol_ratio < 1.2:
        return 'TRENDING'
    if abs(trend) < 0.05 and vol_ratio < 0.8:
        return 'RANGING'
    if vol_ratio > 1.5:
        return 'VOLATILE'
    return 'MIXED'


def _sma_alignment(sma5, sma20, sma50, price):
    tol = 0.001
    score = 0
    if sma5 > sma20 * (1 + tol):
        score += 0.33
    elif sma5 < sma20 * (1 - tol):
        score -= 0.33
    if sma20 > sma50 * (1 + tol):
        score += 0.33
    elif sma20 < sma50 * (1 - tol):
        score -= 0.33
    if price > sma5 * (1 + tol):
        score += 0.34
    elif price < sma5 * (1 - tol):
        score -= 0.34
    return max(-1, min(1, score))


def _price_momentum(price, price_history):
    short_ma = sum(price_history[-3:]) / 3
    long_ma = sum(price_history[-10:]) / 10
    return _normalize((short_ma - long_ma) / long_ma, -0.02, 0.02)


def _get_adaptive_rsi_zones(volatility):
    adj = volatility * 100
    return {'oversold': max(20, 30 - adj), 'overbought': min(80, 70 + adj)}


def _rsi_score(rsi, zones):
    if rsi < zones['oversold']:
        return _normalize(zones['oversold'] - rsi, 0, 20)
    if rsi > zones['overbought']:
        return _normalize(zones['overbought'] - rsi, 0, -20)
    return 0


def _analyze_macd(macd):
    ml = macd['macdLine'][-4:]
    sl = macd['signalLine'][-4:]
    hist = macd['histogram'][-4:]
    if len(ml) < 4:
        return {'score': 0}
    crossover = 0
    if ml[-2] <= sl[-2] and ml[-1] > sl[-1]:
        crossover = 1
    elif ml[-2] >= sl[-2] and ml[-1] < sl[-1]:
        crossover = -1
    hist_mom = (hist[-1] - hist[-2]) / abs(hist[-2]) if hist[-2] != 0 else 0
    hist_boost = _normalize(hist_mom, -0.2, 0.2) * 0.4 if abs(hist_mom) > 0.03 else 0
    score = crossover * 0.6 + hist_boost
    if ml[-1] > 0 and crossover > 0:
        score += 0.15
    if ml[-1] < 0 and crossover < 0:
        score -= 0.15
    return {'score': max(-1, min(1, score))}


def _oscillator_consensus(rsi, stochastic, williams):
    rsi_sig = 1 if rsi < 30 else (-1 if rsi > 70 else 0)
    stoch_sig = 1 if stochastic < 20 else (-1 if stochastic > 80 else 0)
    will_sig = 1 if williams < -80 else (-1 if williams > -20 else 0)
    return (rsi_sig + stoch_sig + will_sig) / 3


def _classify_volatility(volatility):
    if volatility > 0.04:
        return 'HIGH'
    if volatility > 0.02:
        return 'MEDIUM'
    return 'LOW'


def _volatility_score(vol_regime, market_regime):
    scores = {
        'HIGH':   {'TRENDING': -0.3, 'RANGING': -0.5, 'VOLATILE': -0.7, 'MIXED': -0.4},
        'MEDIUM': {'TRENDING':  0.2, 'RANGING':  0.1, 'VOLATILE': -0.2, 'MIXED':  0.0},
        'LOW':    {'TRENDING':  0.4, 'RANGING':  0.3, 'VOLATILE':  0.1, 'MIXED':  0.2},
    }
    return scores.get(vol_regime, {}).get(market_regime, 0)


def _volume_score(volume_ratio, trend_score):
    if volume_ratio > 1.5:
        return 0.5 if trend_score > 0 else -0.3
    if volume_ratio < 0.7 and abs(trend_score) > 0.3:
        return -0.3
    return 0


def _detect_pullback(data):
    if len(data) < 5:
        return {'isPullback': False, 'pullbackLow': 0}
    recent = [d['close'] for d in data[-20:]]
    highest = max(recent)
    current = recent[-1]
    is_pb = current < highest * 0.95 and current < recent[-2]
    return {'isPullback': is_pb, 'pullbackLow': min(recent[-10:])}


def _indicator_agreement(scores):
    valid = [s for s in scores if s == s]
    if len(valid) < 2:
        return 0.5
    avg = sum(valid) / len(valid)
    avg_dev = sum(abs(s - avg) for s in valid) / len(valid)
    return max(0, 1 - avg_dev)


def _position_params(price, volatility, confidence):
    stop_pct = 0.03 if volatility > 0.03 else 0.02
    stop = price * (1 - stop_pct)
    target_pct = 0.01 + (confidence / 100) * 0.005
    target = price * (1 + target_pct)
    risk = abs(price - stop)
    reward = abs(target - price)
    rr = f"{reward / risk:.2f}" if risk > 0 else "0.00"
    expected = f"{abs(target - price) / price * 100:.1f}"
    return {
        'entryPrice': round(price, 2),
        'stopLoss': round(stop, 2),
        'priceTarget': round(target, 2),
        'riskReward': rr,
        'expectedReturn': expected,
    }


# ---------------------------------------------------------------------------
# Main signal generator
# ---------------------------------------------------------------------------

def _generate_signals(data, strategy=None):
    """Port of JS generateTradingSignals(). data is processed OHLCV list."""
    # --- resolve strategy params with defaults ---
    s = strategy or {}
    w = s.get('weights', {})
    t = s.get('thresholds', {})
    setups_on = s.get('setups', {})

    W_SMA_ALIGN = float(w.get('sma_align', 3.0))
    W_PRICE_MOM = float(w.get('price_mom', 2.0))
    W_RSI       = float(w.get('rsi', 2.0))
    W_MACD      = float(w.get('macd', 1.5))

    WATCH_CONF_MIN = float(t.get('watch_confidence_min', 40))
    ENTER_CONF_MIN = float(t.get('enter_confidence_min', 45))
    RSI_OVERSOLD   = float(t.get('rsi_oversold', 35))
    RSI_OVERBOUGHT = float(t.get('rsi_overbought', 70))
    CCI_OVERSOLD   = float(t.get('cci_oversold', -100))

    TREND_ON    = setups_on.get('TREND', True)
    MEANREV_ON  = setups_on.get('MEAN_REVERSION', True)
    PULLBACK_ON = setups_on.get('PULLBACK', True)
    def ignore(reasons, confidence=0, strength=0, regime='UNKNOWN'):
        return {
            'signal': 'IGNORE', 'setupType': 'NONE',
            'confidence': round(confidence), 'strength': strength,
            'reasons': reasons, 'entryPrice': 0, 'stopLoss': 0,
            'priceTarget': 0, 'riskReward': '0.00', 'marketRegime': regime,
        }

    if not data or len(data) < 50:
        return ignore(['Insufficient data'])

    sma5_list  = _calculate_sma(data, 5)
    sma20_list = _calculate_sma(data, 20)
    sma50_list = _calculate_sma(data, 50)
    rsi_list   = _calculate_rsi(data)
    macd       = _calculate_macd(data)
    ao         = _calculate_ao(data)
    cci        = _calculate_cci(data)

    if not all([sma5_list, sma20_list, sma50_list, rsi_list, ao, cci]):
        return ignore(['Invalid indicator data'])

    sma5  = sma5_list[-1]
    sma20 = sma20_list[-1]
    sma50 = sma50_list[-1]
    rsi   = rsi_list[-1]
    price = data[-1]['close']
    vol   = data[-1].get('volume', 0)

    price_history  = [d['close'] for d in data[-20:]]
    volume_history = [d.get('volume', 0) for d in data[-20:]]
    avg_volume     = sum(volume_history) / len(volume_history) if volume_history else 1

    market_regime = _detect_market_regime(data)
    tr  = _calculate_true_range(data[-14:])
    atr = sum(tr) / len(tr) if tr else 0
    volatility = atr / price if price else 0

    # Scores
    sma_align    = _sma_alignment(sma5, sma20, sma50, price)
    price_mom    = _price_momentum(price, price_history)
    trend_score  = sma_align * W_SMA_ALIGN + price_mom * W_PRICE_MOM

    rsi_zones      = _get_adaptive_rsi_zones(volatility)
    rsi_sc         = _rsi_score(rsi, rsi_zones)
    macd_analysis  = _analyze_macd(macd)
    momentum_score = rsi_sc * W_RSI + macd_analysis['score'] * W_MACD

    vol_regime   = _classify_volatility(volatility)
    vol_score    = _volatility_score(vol_regime, market_regime)
    vol_ratio    = vol / avg_volume if avg_volume else 1
    volume_score = _volume_score(vol_ratio, trend_score)

    stochastic = _calculate_stochastic(data[-14:])
    williams   = _calculate_williams_r(data[-14:])
    osc_score  = _oscillator_consensus(rsi, stochastic, williams)

    agreement    = _indicator_agreement([trend_score, momentum_score, osc_score])
    data_quality = min(len(data) / 100, 1) * 0.3 + (0.3 if avg_volume > 0 else 0) + 0.4
    confidence   = (agreement * 0.7 + data_quality * 0.3) * 100
    strength     = max(0, min(5, round(confidence / 20)))

    # Market gate
    if volatility > 0.08:
        return ignore(['Extreme volatility detected'], confidence, strength, market_regime)
    if momentum_score < -0.6 and trend_score < -0.4:
        return ignore(['Strong bearish momentum'], confidence, strength, market_regime)

    # Setup classification
    setup_type = 'NONE'
    reasons    = []
    pullback   = _detect_pullback(data)

    if TREND_ON and trend_score > 0.4 and price > sma50 and sma20 > sma50:
        setup_type = 'TREND'
        reasons = ['Uptrend: Price > SMA50, SMA20 > SMA50']
    elif MEANREV_ON and abs(trend_score) < 0.2 and osc_score > 0.5:
        setup_type = 'MEAN_REVERSION'
        reasons = ['Mean reversion: Oscillators in extreme territory']
    elif PULLBACK_ON and trend_score > 0.3 and pullback['isPullback']:
        setup_type = 'PULLBACK'
        reasons = ['Pullback in uptrend']

    if setup_type == 'NONE':
        return ignore(['No valid setup detected'], confidence, strength, market_regime)

    state = 'IGNORE' if confidence < WATCH_CONF_MIN else 'WATCH'
    vol_too_high = volatility > 0.05

    ml   = macd['macdLine']
    sl_m = macd['signalLine']
    hist = macd['histogram']

    mean_rev_entry = trend_entry = pullback_entry = False
    entry_reasons = []

    if setup_type == 'MEAN_REVERSION' and not vol_too_high:
        cci_val  = cci[-1] if cci else 0
        ao_turn  = len(ao) >= 2 and ao[-1] > ao[-2]
        rsi_os   = rsi < RSI_OVERSOLD
        macd_pos = (len(ml) >= 3 and ml[-1] > sl_m[-1] and
                    len(hist) >= 2 and hist[-1] > hist[-2])
        if cci_val < CCI_OVERSOLD: entry_reasons.append('CCI reversal from oversold')
        if ao_turn:                entry_reasons.append('AO turning positive')
        if rsi_os:                 entry_reasons.append(f'RSI oversold ({round(rsi)})')
        if macd_pos:               entry_reasons.append('MACD bullish crossover')
        mean_rev_entry = (cci_val < CCI_OVERSOLD and ao_turn and rsi_os) or (macd_pos and ao_turn)
        if mean_rev_entry:
            reasons.append('Mean reversion entry: ' + ', '.join(entry_reasons))

    if setup_type == 'TREND' and not vol_too_high:
        ao_pos   = bool(ao) and ao[-1] > 0
        rsi_ok   = rsi < RSI_OVERBOUGHT
        macd_pos = len(ml) >= 3 and ml[-1] > sl_m[-1]
        if ao_pos:   entry_reasons.append('AO positive')
        if macd_pos: entry_reasons.append('MACD positive')
        if rsi_ok:   entry_reasons.append(f'RSI healthy ({round(rsi)})')
        trend_entry = ao_pos and macd_pos and rsi_ok
        if trend_entry:
            reasons.append('Trend entry: ' + ', '.join(entry_reasons))

    if setup_type == 'PULLBACK' and not vol_too_high:
        bounce      = price > pullback['pullbackLow']
        ao_pos      = bool(ao) and ao[-1] > 0
        hist_rising = len(hist) >= 2 and hist[-1] > hist[-2]
        if bounce:       entry_reasons.append('Price bouncing')
        if ao_pos:       entry_reasons.append('AO positive')
        if hist_rising:  entry_reasons.append('MACD histogram rising')
        pullback_entry = bounce and ao_pos and hist_rising
        if pullback_entry:
            reasons.append('Pullback entry: ' + ', '.join(entry_reasons))

    any_entry = mean_rev_entry or trend_entry or pullback_entry
    if any_entry and confidence >= ENTER_CONF_MIN:
        state = 'ENTER'
        reasons.append(f'High confidence ({round(confidence)}%)')
    elif any_entry:
        state = 'WATCH'
        reasons.append(f'Moderate confidence ({round(confidence)}%)')
    elif state == 'WATCH':
        reasons.append('Setup detected but entry conditions not fully met')

    params = _position_params(price, volatility, confidence)
    return {
        'signal':      state,
        'setupType':   setup_type,
        'confidence':  round(confidence),
        'strength':    strength,
        'reasons':     reasons,
        'entryPrice':  params['entryPrice'],
        'stopLoss':    params['stopLoss'],
        'priceTarget': params['priceTarget'],
        'riskReward':  params['riskReward'],
        'expectedReturn': params['expectedReturn'],
        'marketRegime': market_regime,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def analyze(api_response, strategy=None):
    """
    Pass the raw JSON from /api/stocks/stock?symbol=X.
    Returns a signal dict with keys: signal, confidence, reasons, entryPrice, stopLoss, priceTarget, ...
    Optionally pass a strategy dict to override default weights/thresholds/setups.
    """
    time_series = api_response.get('Time Series (Daily)', {})
    if not time_series:
        return {'signal': 'IGNORE', 'reasons': ['No data returned'], 'confidence': 0}
    data = _process_api_data(time_series)
    return _generate_signals(data, strategy=strategy)
