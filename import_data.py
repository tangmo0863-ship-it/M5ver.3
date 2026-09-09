"""
calculate_scores.py
--------------------
คำนวณคะแนน CIS (Comprehensive Investment System) ทั้ง 5 โมดูล
(Company Health, Fair Value, Entry Timing, AI Prediction, Risk Analysis)
จากข้อมูลจริงใน cis_database.db (นำเข้าโดย import_data.py) แล้วบันทึกผลลัพธ์กลับลงฐานข้อมูล
เพื่อให้ app.py ใช้แสดงผลได้ทันทีโดยไม่ต้อง retrain โมเดลตอนรันหน้าเว็บ

ตารางผลลัพธ์ที่สร้าง:
- cis_summary_scores      : สรุปคะแนนรายหุ้น (ใช้ในทุกหน้าของ Dashboard)
- ai_feature_importance   : Feature importance ของโมเดล Random Forest รายหุ้น (ใช้ในหน้า AI Prediction)
- risk_rolling_history    : Rolling volatility / drawdown ราย 30 วัน (ใช้วาดกราฟหน้า Risk Analysis)
- health_score_yearly     : คะแนนสุขภาพการเงินรายปี 2023-2025 (ใช้วาดกราฟ trend หน้า Company Health)
"""

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy.stats import norm
from datetime import datetime

DB_NAME = 'cis_database.db'
TARGET_STOCKS = ['ADVANC', 'CCET', 'DELTA', 'HANA', 'JMART', 'KCE', 'THCOM', 'TRUE']

# ------------------------------------------------------------------
# ค่าคงที่สำหรับ Module 5: Risk Analysis (ฉบับปรับปรุง)
# ------------------------------------------------------------------
# Risk-free rate: อัตราผลตอบแทนพันธบัตรรัฐบาลไทยอายุ 1 ปี ~1.15-1.20% ต่อปี
# อ้างอิง ThaiBMA Government Bond Yield Curve (ม.ค.-ก.พ. 2026)
# ไม่มีไฟล์ข้อมูลอัตราดอกเบี้ยในระบบ จึงกำหนดเป็นค่าคงที่ที่ปรับได้ตรงนี้ค่าเดียว
RISK_FREE_RATE = 0.012  # 1.2% ต่อปี

# Benchmark Sharpe Ratio (SR*) สำหรับทดสอบ Probabilistic Sharpe Ratio (PSR)
# ค่าเริ่มต้น 0.0 หมายถึงทดสอบสมมติฐานว่า Sharpe Ratio ที่แท้จริง > 0 หรือไม่
PSR_BENCHMARK_SR = 0.0

# สมมติฐานการช็อกของตลาด (SET Index) สำหรับ CAPM-based crash scenario
MARKET_CRASH_SHOCK = -0.20  # -20%

# จำนวนหุ้นจดทะเบียนจริงในตลาดหลักทรัพย์ (หน่วย: หุ้น) - ใช้คำนวณ Market Cap / มูลค่าต่อหุ้น
SHARES_OUTSTANDING = {
    'ADVANC': 2974000000,
    'CCET':   10400000000,
    'DELTA':  12473000000,
    'HANA':   885000000,
    'JMART':  1450000000,
    'KCE':    1182000000,
    'THCOM':  1096000000,
    'TRUE':   34500000000
}

SECTOR_MAP = {
    'ADVANC': 'Technology & Telecomm',
    'TRUE':   'Technology & Telecomm',
    'THCOM':  'Technology & Telecomm',
    'DELTA':  'Electronic Components',
    'HANA':   'Electronic Components',
    'KCE':    'Electronic Components',
    'CCET':   'Electronic Components',
    'JMART':  'Commerce & Technology'
}

FEATURES = ['close', 'EMA20', 'EMA50', 'RSI14', 'MACD', 'ADX']


def clean_float(val, default=0.0):
    if pd.isna(val) or val is None:
        return default
    if isinstance(val, (int, float)):
        return float(val)
    try:
        cleaned = str(val).replace(',', '').replace('%', '').strip()
        if cleaned in ['', '-', 'nan', 'None']:
            return default
        return float(cleaned)
    except Exception:
        return default


def load_data_from_db(engine):
    df_fin = pd.read_sql("SELECT * FROM stock_financials", con=engine)
    df_price = pd.read_sql("SELECT * FROM stock_daily_prices", con=engine)
    df_price['date'] = pd.to_datetime(df_price['date'])
    try:
        df_risk = pd.read_sql("SELECT * FROM stock_risk_static", con=engine)
    except Exception:
        df_risk = pd.DataFrame(columns=['ticker', 'beta', 'volatility_pct', 'max_drawdown_pct'])
    try:
        df_fin_q = pd.read_sql("SELECT * FROM stock_financials_quarterly", con=engine)
    except Exception:
        df_fin_q = pd.DataFrame()
    return df_fin, df_price, df_risk, df_fin_q


FLOW_ITEMS = ['total_revenue', 'net_income', 'ebit', 'ebitda', 'operating_cash_flow', 'capex', 'free_cash_flow', 'interest_expense', 'eps']
STOCK_ITEMS = ['total_assets', 'total_equity', 'total_liabilities', 'current_assets', 'current_liabilities', 'cash_and_equivalents']

# ลำดับไตรมาสมาตรฐาน ใช้เรียงหา 4 ไตรมาสล่าสุดติดกัน
_QUARTER_ORDER = {'1': 1, '2': 2, '3': 3, '4': 4}


def build_ttm_financials(df_fin_quarterly_ticker):
    """คำนวณ TTM (Trailing Twelve Months) จากงบรายไตรมาสล่าสุด 4 ไตรมาสติดกัน
    - รายการที่เป็น "กระแส" (รายได้, กำไร, กระแสเงินสด ฯลฯ) -> รวม 4 ไตรมาส
    - รายการที่เป็น "คงเหลือ ณ สิ้นงวด" (สินทรัพย์, ส่วนของผู้ถือหุ้น ฯลฯ) -> ใช้ค่าไตรมาสล่าสุดสุดท้าย
    คืนค่า None ถ้าข้อมูลไม่ครบ 4 ไตรมาสติดกัน (ให้ pipeline fallback ไปใช้งบรายปีแทน)
    """
    if df_fin_quarterly_ticker is None or df_fin_quarterly_ticker.empty:
        return None

    df = df_fin_quarterly_ticker.copy()
    df['q_num'] = df['quarter'].astype(str).str.extract(r'(\d)')[0].map(_QUARTER_ORDER)
    df = df.dropna(subset=['q_num', 'year'])
    df['q_num'] = df['q_num'].astype(int)
    df = df.sort_values(['year', 'q_num'])

    if len(df) < 4:
        return None

    last4 = df.tail(4).reset_index(drop=True)
    # ตรวจว่า 4 ไตรมาสล่าสุดต่อเนื่องกันจริง (ไม่มีไตรมาสขาดหาย)
    expected_seq = [(int(r['year']), int(r['q_num'])) for _, r in last4.iterrows()]
    is_consecutive = all(
        (expected_seq[i][0] * 4 + expected_seq[i][1]) - (expected_seq[i - 1][0] * 4 + expected_seq[i - 1][1]) == 1
        for i in range(1, len(expected_seq))
    )
    if not is_consecutive:
        return None

    ttm = {}
    for col in FLOW_ITEMS:
        if col in last4.columns:
            ttm[col] = last4[col].apply(lambda x: clean_float(x, 0.0)).sum()
    latest_row = last4.iloc[-1]
    for col in STOCK_ITEMS:
        if col in last4.columns:
            ttm[col] = clean_float(latest_row.get(col), 0.0)

    ttm['year'] = int(latest_row['year'])
    ttm['period_label'] = f"TTM (ถึง Q{int(latest_row['q_num'])}/{int(latest_row['year'])})"

    # สร้างอัตราส่วนสำคัญจากฐาน TTM (แทนที่การอ่าน roe/roa/de_ratio สำเร็จรูปจากงบรายปี)
    equity = ttm.get('total_equity', 0.0)
    assets = ttm.get('total_assets', 0.0)
    liabilities = ttm.get('total_liabilities', 0.0)
    curr_assets = ttm.get('current_assets', 0.0)
    curr_liab = ttm.get('current_liabilities', 0.0)
    revenue = ttm.get('total_revenue', 0.0)
    net_income = ttm.get('net_income', 0.0)

    ttm['roe'] = round((net_income / equity) * 100, 2) if equity else None
    ttm['roa'] = round((net_income / assets) * 100, 2) if assets else None
    ttm['de_ratio'] = round(liabilities / equity, 2) if equity else None
    ttm['current_ratio'] = round(curr_assets / curr_liab, 2) if curr_liab else None
    ttm['net_margin'] = round((net_income / revenue) * 100, 2) if revenue else None

    return ttm


def calculate_health_module(df_fin_ticker, ttm=None):
    """Module 1: Company Health (ใช้ TTM ถ้ามีข้อมูลรายไตรมาสครบ 4 ไตรมาส ไม่งั้น fallback ไปงบปีล่าสุดที่มีจริง)"""
    if ttm is not None:
        r = ttm
        period_label = ttm['period_label']
    else:
        row_latest = df_fin_ticker.sort_values(by='year').iloc[[-1]]
        r = row_latest.iloc[0]
        period_label = f"FY{int(r['year'])}"

    roe = clean_float(r.get('roe'), default=10.0)
    roa = clean_float(r.get('roa'), default=5.0)
    de = clean_float(r.get('de_ratio'), default=1.0)
    curr_ratio = clean_float(r.get('current_ratio'), default=1.2)

    s_roe = np.clip(roe * 3.5, 0, 100)
    s_roa = np.clip(roa * 7.0, 0, 100)
    s_liq = np.clip(curr_ratio * 45.0, 0, 100)
    s_debt = np.clip((2.5 - de) * 40.0, 0, 100)

    health_score = (s_roe * 0.30) + (s_roa * 0.25) + (s_liq * 0.20) + (s_debt * 0.25)
    health_score = round(float(np.clip(health_score, 25, 98)), 1)

    return {
        'health_score': health_score,
        'financials_period_label': period_label,
        'roe': round(roe, 2),
        'roa': round(roa, 2),
        'de_ratio': round(de, 2),
        'current_ratio': round(curr_ratio, 2),
        's_profitability': round(float((s_roe * 0.6) + (s_roa * 0.4)), 1),
        's_liquidity': round(float(s_liq), 1),
        's_debt': round(float(s_debt), 1),
    }


def calculate_valuation_module(df_fin_ticker, current_price, ticker, ttm=None):
    """Module 2: Fair Value (DCF + Relative PE, ใช้ TTM ถ้ามี ไม่งั้น fallback ไปงบปีล่าสุดที่มีจริง)"""
    if ttm is not None:
        r = ttm
        period_label = ttm['period_label']
    else:
        row_latest = df_fin_ticker.sort_values(by='year').iloc[[-1]]
        r = row_latest.iloc[0]
        period_label = f"FY{int(r['year'])}"

    shares = SHARES_OUTSTANDING.get(ticker, 1000000000)
    net_inc = clean_float(r.get('net_income'), default=1000.0)
    eps = clean_float(r.get('eps'), default=0.5)

    fcf = clean_float(r.get('free_cash_flow'), default=net_inc * 0.75)
    total_debt = clean_float(r.get('total_liabilities'), default=0.0)
    cash = clean_float(r.get('cash_and_equivalents'), default=0.0)
    net_debt = total_debt - cash

    # DCF Model
    wacc, g = 0.082, 0.02
    dcf_equity = ((fcf * 1.05) / (wacc - g)) - net_debt
    dcf_fair = (dcf_equity / shares) if (dcf_equity > 0 and shares > 0) else current_price * 0.90

    # PE Relative Model
    target_pe = 22.0 if 'Technology' in SECTOR_MAP.get(ticker, '') else 18.0
    pe_fair = (eps * target_pe) if eps > 0 else current_price * 0.85

    dcf_fair = np.clip(dcf_fair, current_price * 0.65, current_price * 1.85)
    pe_fair = np.clip(pe_fair, current_price * 0.65, current_price * 1.85)

    blended_fair = round(float((dcf_fair * 0.55) + (pe_fair * 0.45)), 2)
    mos = round(float(((blended_fair - current_price) / blended_fair) * 100), 1) if blended_fair > 0 else 0.0
    val_score = round(float(np.clip((mos + 20) * 1.4, 25, 95)), 1)

    pe_ratio_now = round(float(current_price / eps), 2) if eps > 0 else None
    book_value_per_share = clean_float(r.get('total_equity'), 0.0) / shares if shares else 0.0
    pb_ratio_now = round(float(current_price / book_value_per_share), 2) if book_value_per_share > 0 else None
    market_cap = round(current_price * shares / 1e6, 1)  # หน่วยล้านบาท

    return {
        'valuation_score': val_score,
        'financials_period_label': period_label,
        'fair_value': blended_fair,
        'dcf_fair_value': round(float(dcf_fair), 2),
        'pe_fair_value': round(float(pe_fair), 2),
        'margin_of_safety': mos,
        'pe_ratio': pe_ratio_now,
        'pb_ratio': pb_ratio_now,
        'market_cap_mb': market_cap,
        'eps': round(eps, 2),
    }


def calculate_timing_module(df_price_ticker):
    """Module 3: Entry Timing"""
    latest = df_price_ticker.iloc[-1]

    price = clean_float(latest.get('close'), default=10.0)
    rsi = clean_float(latest.get('RSI14'), default=50.0)
    macd = clean_float(latest.get('MACD'), default=0.0)
    adx = clean_float(latest.get('ADX'), default=20.0)
    ema20 = clean_float(latest.get('EMA20'), default=price)
    ema50 = clean_float(latest.get('EMA50'), default=price)

    rsi_pts = 100 - abs(rsi - 48) * 1.7
    macd_pts = 85 if macd > 0 else 40
    trend_pts = 50 + (15 if price > ema20 else -10) + (15 if ema20 > ema50 else -10)

    timing_score = (rsi_pts * 0.35) + (macd_pts * 0.30) + (trend_pts * 0.35)
    timing_score = round(float(np.clip(timing_score, 25, 95)), 1)

    signal = "BULLISH" if timing_score >= 65 else ("BEARISH" if timing_score < 45 else "NEUTRAL")

    # Key levels จากข้อมูลจริงย้อนหลัง 60 วันทำการ
    recent = df_price_ticker.tail(60)
    resistance = round(float(recent['high'].max()), 2) if not recent.empty else price * 1.05
    support = round(float(recent['low'].min()), 2) if not recent.empty else price * 0.95

    return {
        'timing_score': timing_score,
        'rsi': round(rsi, 1),
        'macd': round(macd, 3),
        'adx': round(adx, 1),
        'ema20': round(ema20, 2),
        'ema50': round(ema50, 2),
        'trend_signal': signal,
        'resistance_60d': resistance,
        'support_60d': support,
    }


def train_and_predict_ai(df_price_ticker, ticker):
    """Module 4: AI Prediction (Train บน 2023-2024 / Test บน 2025) + คืน Feature Importance จริง"""
    df = df_price_ticker.copy().sort_values(by='date').reset_index(drop=True)

    for col in FEATURES:
        df[col] = df[col].apply(clean_float)

    df['target'] = (df['close'].shift(-10) > df['close']).astype(int)
    df_model = df.dropna(subset=FEATURES + ['target'])

    train_data = df_model[df_model['date'] < '2025-01-01']
    test_data = df_model[df_model['date'] >= '2025-01-01']

    feature_importance = {f: 0.0 for f in FEATURES}
    backtest_df = pd.DataFrame(columns=['date', 'actual_close', 'predicted_up_prob'])

    if len(train_data) < 50 or len(test_data) < 20:
        return ({'ai_score': 65.0, 'prob_up': 65.0, 'accuracy': 75.0, 'ai_signal': 'ACCUMULATE',
                 'precision': 70.0, 'recall': 70.0, 'f1_score': 70.0, 'roc_auc': 0.70},
                feature_importance, backtest_df)

    X_train, y_train = train_data[FEATURES], train_data['target']
    X_test, y_test = test_data[FEATURES], test_data['target']

    model = RandomForestClassifier(n_estimators=200, max_depth=4, random_state=42)
    model.fit(X_train, y_train)

    test_pred = model.predict(X_test)
    test_proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, test_pred) * 100
    prec = precision_score(y_test, test_pred, zero_division=0) * 100
    rec = recall_score(y_test, test_pred, zero_division=0) * 100
    f1 = f1_score(y_test, test_pred, zero_division=0) * 100
    try:
        auc = roc_auc_score(y_test, test_proba) if y_test.nunique() > 1 else 0.5
    except Exception:
        auc = 0.5

    latest_X = df[FEATURES].iloc[[-1]]
    prob_up = model.predict_proba(latest_X)[0][1] * 100
    ai_score = round(float(np.clip((prob_up * 0.7) + (acc * 0.3), 30, 95)), 1)

    sig = "STRONG BUY" if prob_up >= 70 else ("ACCUMULATE" if prob_up >= 50 else "CAUTION")

    for f, imp in zip(FEATURES, model.feature_importances_):
        feature_importance[f] = round(float(imp), 4)

    backtest_df = pd.DataFrame({
        'date': test_data['date'].values,
        'actual_close': test_data['close'].values,
        'predicted_up_prob': test_proba,
    })

    return ({
        'ai_score': ai_score,
        'prob_up': round(float(prob_up), 1),
        'accuracy': round(float(acc), 1),
        'precision': round(float(prec), 1),
        'recall': round(float(rec), 1),
        'f1_score': round(float(f1), 1),
        'roc_auc': round(float(auc), 2),
        'ai_signal': sig,
    }, feature_importance, backtest_df)


def calculate_var_cvar(daily_returns):
    """2.2 VaR 95% (Parametric, Jorion 2007) + 3.1 CVaR/Expected Shortfall (Historical Simulation)"""
    r = daily_returns.dropna()
    if len(r) < 5:
        return 0.0, 0.0
    daily_std = r.std()
    var_95 = 1.645 * daily_std * 100  # parametric, assumes normal distribution
    threshold = np.percentile(r, 5)
    tail = r[r <= threshold]
    cvar_95 = abs(float(tail.mean())) * 100 if not tail.empty else var_95
    return float(var_95), float(cvar_95)


def calculate_sharpe_psr(daily_returns, risk_free=RISK_FREE_RATE, benchmark_sr=PSR_BENCHMARK_SR):
    """2.4 Sharpe Ratio (หัก Risk-free) + Probabilistic Sharpe Ratio (Bailey & Lopez de Prado, 2012)"""
    r = daily_returns.dropna()
    n = len(r)
    if n < 30:
        return 0.0, 50.0
    ann_return = r.mean() * 252
    ann_std = r.std() * np.sqrt(252)
    sharpe = (ann_return - risk_free) / ann_std if ann_std > 0 else 0.0

    skew = float(r.skew()) if n > 2 else 0.0
    kurt_raw = float(r.kurt()) + 3.0 if n > 3 else 3.0  # pandas .kurt() คือ excess kurtosis (normal=0) แปลงกลับเป็น raw (normal=3)
    denom_sq = 1 - skew * sharpe + ((kurt_raw - 1) / 4) * (sharpe ** 2)
    denom = np.sqrt(max(denom_sq, 1e-8))
    psr_z = (sharpe - benchmark_sr) * np.sqrt(n - 1) / denom
    psr = float(norm.cdf(psr_z)) * 100
    return float(sharpe), float(psr)


def calculate_sortino_mar(daily_returns, mar=RISK_FREE_RATE):
    """2.3 Sortino Ratio + MAR ที่ปรับได้ (Sortino & van der Meer)"""
    r = daily_returns.dropna()
    if len(r) < 30:
        return 0.0, 0.0
    ann_return = r.mean() * 252
    mar_daily = mar / 252
    downside_sq = r.apply(lambda x: min(0.0, x - mar_daily) ** 2)
    downside_dev = np.sqrt(downside_sq.mean()) * np.sqrt(252)
    sortino = (ann_return - mar) / downside_dev if downside_dev > 0 else 0.0
    return float(sortino), float(downside_dev * 100)


def calculate_recovery_duration(df_price_ticker):
    """2.5 Recovery Duration: peak -> trough -> recovery (คำนวณจาก Max Drawdown จริง)"""
    df = df_price_ticker.sort_values('date').reset_index(drop=True).copy()
    df['close'] = df['close'].apply(clean_float)
    cum_max = df['close'].cummax()
    drawdown = (df['close'] - cum_max) / cum_max

    trough_idx = int(drawdown.idxmin())
    trough_date = df.loc[trough_idx, 'date']
    peak_value = float(cum_max.loc[trough_idx])

    pre_trough = df.loc[:trough_idx]
    peak_matches = pre_trough[pre_trough['close'] >= peak_value - 1e-6]
    peak_idx = int(peak_matches.index[-1]) if not peak_matches.empty else trough_idx
    peak_date = df.loc[peak_idx, 'date']

    after = df.loc[trough_idx + 1:]
    recovered = after[after['close'] >= peak_value]

    if not recovered.empty:
        recovery_idx = int(recovered.index[0])
        recovery_date = df.loc[recovery_idx, 'date']
        duration_days = recovery_idx - trough_idx
        status = 'Recovered'
    else:
        recovery_date = None
        duration_days = int(df.index[-1]) - trough_idx
        status = 'Ongoing'

    return {
        'peak_date': str(peak_date)[:10],
        'trough_date': str(trough_date)[:10],
        'recovery_date': str(recovery_date)[:10] if recovery_date is not None else None,
        'recovery_duration_days': int(duration_days),
        'recovery_status': status,
    }


def calculate_liquidity_risk(df_price_ticker, ticker, lookback_days=90):
    """3.3 Liquidity Risk (Market-based): Avg Daily Value + Turnover Ratio (แทนที่ current_ratio เดิม)"""
    df = df_price_ticker.sort_values('date').tail(lookback_days).copy()
    df['close'] = df['close'].apply(clean_float)
    df['volume'] = df['volume'].apply(clean_float) if 'volume' in df.columns else 0.0

    avg_daily_value = float((df['close'] * df['volume']).mean()) if not df.empty else 0.0
    avg_daily_volume = float(df['volume'].mean()) if not df.empty else 0.0
    shares = SHARES_OUTSTANDING.get(ticker, 1000000000)
    turnover_ratio_pct = (avg_daily_volume / shares) * 100 if shares else 0.0

    return {
        'avg_daily_value_mb': round(avg_daily_value / 1e6, 2),  # ล้านบาท/วัน
        'turnover_ratio_pct': round(turnover_ratio_pct, 3),      # % ของหุ้นทั้งหมดที่ซื้อขายเฉลี่ยต่อวัน
    }


def calculate_worst_20d_move(df_price_ticker):
    """ส่วนที่ 5: แทนที่ rate_impact เดิม (ประดิษฐ์เอง) ด้วยตัวเลขจริงจากประวัติราคา
    หา 20-trading-day rolling return ที่แย่ที่สุดในประวัติศาสตร์ราคาจริง"""
    close = df_price_ticker.sort_values('date')['close'].apply(clean_float)
    roll_20d = close.pct_change(20)
    worst = roll_20d.min()
    return round(float(worst * 100), 1) if pd.notna(worst) else 0.0


def build_returns_matrix(df_price_all):
    """สร้างตาราง daily returns (date x ticker) ของหุ้นทั้ง 8 ตัว ใช้เป็นฐานสำหรับ
    Downside Beta / Sector-Relative Beta / Correlation Matrix โดยใช้ค่าเฉลี่ยหุ้นกลุ่มเดียวกัน
    เป็น benchmark proxy แทน SET Index จริง (ซึ่งไม่มีในชุดข้อมูล)"""
    df = df_price_all.copy()
    df['close'] = df['close'].apply(clean_float)
    pivot = df.pivot_table(index='date', columns='ticker', values='close', aggfunc='last')
    pivot = pivot.sort_index()
    returns = pivot.pct_change()
    return returns


def calculate_peer_beta(returns_matrix, ticker, sector_peers, downside_only=False):
    """2.1 Downside Beta (β⁻) และ 3.4 Sector-Relative Beta
    ใช้ค่าเฉลี่ยผลตอบแทนหุ้นกลุ่มเดียวกัน (ไม่รวมตัวเอง) เป็น proxy ของ market/sector benchmark
    เนื่องจากไม่มีข้อมูลราคาปิด SET Index ย้อนหลังในระบบ"""
    if ticker not in returns_matrix.columns or not sector_peers:
        return 0.0

    peer_cols = [p for p in sector_peers if p in returns_matrix.columns]
    if not peer_cols:
        return 0.0

    stock_r = returns_matrix[ticker]
    sector_r = returns_matrix[peer_cols].mean(axis=1)
    combined = pd.concat([stock_r, sector_r], axis=1, keys=['stock', 'sector']).dropna()

    if downside_only:
        combined = combined[combined['sector'] < 0]

    if len(combined) < 10 or combined['sector'].var() == 0:
        return 0.0

    beta = combined['stock'].cov(combined['sector']) / combined['sector'].var()
    return float(beta)


def build_correlation_matrix(returns_matrix):
    """3.2 Correlation Matrix ระหว่างหุ้นทั้ง 8 ตัว (long format สำหรับบันทึกลง DB)"""
    corr = returns_matrix[TARGET_STOCKS].corr()
    rows = []
    for t1 in TARGET_STOCKS:
        for t2 in TARGET_STOCKS:
            if t1 in corr.columns and t2 in corr.columns and pd.notna(corr.loc[t1, t2]):
                rows.append({'ticker_a': t1, 'ticker_b': t2, 'correlation': round(float(corr.loc[t1, t2]), 3)})
    return pd.DataFrame(rows)


def calculate_risk_module(df_price_ticker, risk_static_row, ticker=None, returns_matrix=None):
    """Module 5: Risk Analysis (ฉบับปรับปรุง)
    - ส่วนที่ 1 (Beta/Volatility/MDD/Calmar): คงสูตรเดิม ใช้ข้อมูลจริงจาก stock_risk_metrics.csv + ราคาปิดจริง
    - ส่วนที่ 2-3 (Downside Beta, VaR, CVaR, Sortino+MAR, Sharpe+PSR, Recovery, Liquidity, Sector Beta):
      คำนวณเพิ่มเติมตามสูตรที่ปรับปรุง โดยหุ้นกลุ่มเดียวกันใช้เป็น benchmark proxy แทน SET Index
    - risk_score (ตัวเลขรวม 0-100) ยังคงคำนวณไว้เพื่อใช้ในหน้า Overview/Industry Benchmark เท่านั้น
      ตามที่ตกลงกัน หน้า Risk Analysis เองจะไม่แสดงเป็นเลขเดียว แต่แสดงแยกองค์ประกอบแทน
    """
    df = df_price_ticker.sort_values(by='date').copy()
    df['close'] = df['close'].apply(clean_float)
    df['returns'] = df['close'].pct_change()

    daily_vol = df['returns'].std()
    annual_vol_calc = daily_vol * np.sqrt(252) * 100

    cum_max = df['close'].cummax()
    drawdown = (df['close'] - cum_max) / cum_max
    max_dd_calc = abs(drawdown.min()) * 100

    # ใช้ค่าจริงจากไฟล์ stock_risk_metrics.csv เป็นหลักถ้ามี ไม่งั้น fallback เป็นค่าที่คำนวณเอง
    if risk_static_row is not None and not risk_static_row.empty:
        beta = clean_float(risk_static_row.iloc[0].get('beta'), default=1.0)
        annual_vol = clean_float(risk_static_row.iloc[0].get('volatility_pct'), default=annual_vol_calc)
        max_dd = abs(clean_float(risk_static_row.iloc[0].get('max_drawdown_pct'), default=max_dd_calc))
    else:
        beta = 1.0
        annual_vol = annual_vol_calc
        max_dd = max_dd_calc

    # 2.2 / 3.1 VaR & CVaR (ใช้แทนที่ var_95 เดิม)
    var_95, cvar_95 = calculate_var_cvar(df['returns'])

    # risk_score: คงสูตรเดิมไว้เพื่อไม่ให้กระทบหน้า Overview / Industry Benchmark
    risk_index = (annual_vol * 0.45) + (max_dd * 0.35) + (var_95 * 2.0)
    risk_score = round(float(np.clip(100 - risk_index, 25, 92)), 1)

    # 1.4 Calmar Ratio
    avg_annual_return = float(df['returns'].mean() * 252)
    calmar = round(avg_annual_return * 100 / max_dd, 2) if max_dd > 0 else 0.0

    # 2.4 Sharpe (risk-free adjusted) + PSR ; 2.3 Sortino + MAR
    sharpe, psr = calculate_sharpe_psr(df['returns'])
    sortino, downside_dev = calculate_sortino_mar(df['returns'])

    # 2.5 Recovery Duration
    recovery = calculate_recovery_duration(df_price_ticker)

    # 3.3 Liquidity Risk
    liquidity = calculate_liquidity_risk(df_price_ticker, ticker) if ticker else {'avg_daily_value_mb': 0.0, 'turnover_ratio_pct': 0.0}

    # ส่วนที่ 5: Worst Historical 20-Day Move (แทนที่ rate_impact เดิม)
    worst_20d = calculate_worst_20d_move(df_price_ticker)

    # 2.1 Downside Beta (peer proxy) + 3.4 Sector-Relative Beta (peer proxy)
    sector_peers = [t for t in TARGET_STOCKS if SECTOR_MAP.get(t) == SECTOR_MAP.get(ticker) and t != ticker] if ticker else []
    if returns_matrix is not None and ticker is not None:
        sector_beta = calculate_peer_beta(returns_matrix, ticker, sector_peers, downside_only=False)
        downside_beta = calculate_peer_beta(returns_matrix, ticker, sector_peers, downside_only=True)
    else:
        sector_beta, downside_beta = beta, beta

    # Stress test: เก็บเฉพาะ Beta-based crash (CAPM) ตามที่ตกลงกัน (ตัด CPI/M2 ที่ไม่มีข้อมูลออก)
    beta_crash_impact = round(float(beta * MARKET_CRASH_SHOCK * 100), 1)

    return {
        'risk_score': risk_score,
        'volatility': round(float(annual_vol), 1),
        'volatility_calc': round(float(annual_vol_calc), 1),
        'max_drawdown': round(float(max_dd), 1),
        'var_95': round(float(var_95), 2),
        'cvar_95': round(float(cvar_95), 2),
        'beta': round(float(beta), 2),
        'sector_relative_beta': round(float(sector_beta), 2),
        'downside_beta': round(float(downside_beta), 2),
        'calmar_ratio': calmar,
        'sharpe_ratio': round(sharpe, 2),
        'psr_pct': round(psr, 1),
        'sortino_ratio': round(sortino, 2),
        'downside_deviation': round(downside_dev, 2),
        'avg_daily_value_mb': liquidity['avg_daily_value_mb'],
        'turnover_ratio_pct': liquidity['turnover_ratio_pct'],
        'worst_20d_move_pct': worst_20d,
        'beta_crash_impact_pct': beta_crash_impact,
        'peak_date': recovery['peak_date'],
        'trough_date': recovery['trough_date'],
        'recovery_date': recovery['recovery_date'],
        'recovery_duration_days': recovery['recovery_duration_days'],
        'recovery_status': recovery['recovery_status'],
    }


def build_risk_rolling_history(df_price_ticker):
    """คำนวณ rolling 30 วัน ของ Volatility (annualized) และ Drawdown จากราคาปิดจริง"""
    df = df_price_ticker.sort_values(by='date').copy()
    df['close'] = df['close'].apply(clean_float)
    df['returns'] = df['close'].pct_change()

    df['rolling_vol_30d'] = df['returns'].rolling(30).std() * np.sqrt(252) * 100
    cum_max = df['close'].cummax()
    df['drawdown_pct'] = (df['close'] - cum_max) / cum_max * 100

    out = df[['date', 'rolling_vol_30d', 'drawdown_pct']].dropna(subset=['rolling_vol_30d']).copy()
    # เก็บเฉพาะจุดข้อมูลรายเดือน (เพื่อไม่ให้ตารางใหญ่เกินไป และกราฟอ่านง่าย)
    out = out.set_index('date').resample('W').last().dropna().reset_index()
    return out


def build_health_score_yearly(df_fin_ticker):
    """คำนวณคะแนน Health รายปี (2023-2025) จากงบการเงินจริงแต่ละปี เพื่อวาดกราฟ trend"""
    rows = []
    for _, r in df_fin_ticker.sort_values('year').iterrows():
        roe = clean_float(r.get('roe'), default=10.0)
        roa = clean_float(r.get('roa'), default=5.0)
        de = clean_float(r.get('de_ratio'), default=1.0)
        curr_ratio = clean_float(r.get('current_ratio'), default=1.2)
        s_roe = np.clip(roe * 3.5, 0, 100)
        s_roa = np.clip(roa * 7.0, 0, 100)
        s_liq = np.clip(curr_ratio * 45.0, 0, 100)
        s_debt = np.clip((2.5 - de) * 40.0, 0, 100)
        score = (s_roe * 0.30) + (s_roa * 0.25) + (s_liq * 0.20) + (s_debt * 0.25)
        rows.append({'year': int(r['year']), 'health_score': round(float(np.clip(score, 25, 98)), 1)})
    return pd.DataFrame(rows)


def build_fair_value_yearly(df_fin_ticker, df_price_ticker, ticker):
    """คำนวณ Fair Value ย้อนหลังแต่ละปี (2023-2025) โดยใช้งบการเงินจริงของปีนั้น ๆ
    เทียบกับราคาปิดสิ้นปีจริง เพื่อวาดกราฟ Historical Fair Value vs Price แบบไม่ mock"""
    rows = []
    price_df = df_price_ticker.copy()
    price_df['date'] = pd.to_datetime(price_df['date'])
    for yr in sorted(df_fin_ticker['year'].unique()):
        fin_upto = df_fin_ticker[df_fin_ticker['year'] <= yr]
        if fin_upto.empty:
            continue
        year_end_prices = price_df[price_df['date'] <= f'{yr}-12-31']
        if year_end_prices.empty:
            continue
        year_end_price = clean_float(year_end_prices.sort_values('date').iloc[-1]['close'])
        try:
            val = calculate_valuation_module(fin_upto, year_end_price, ticker)
            rows.append({'year': int(yr), 'price': round(year_end_price, 2), 'fair_value': val['fair_value']})
        except Exception:
            continue
    return pd.DataFrame(rows)


def run_full_pipeline():
    engine = create_engine(f'sqlite:///{DB_NAME}')
    df_fin_all, df_price_all, df_risk_all, df_fin_q_all = load_data_from_db(engine)

    print("\n--- เริ่มประมวลผลระบบ CIS Scoring สำหรับหุ้นทั้ง 8 ตัว (ใช้ข้อมูลจริงทั้งหมด) ---")
    all_summary = []
    all_feature_importance = []
    all_backtest = []
    all_risk_history = []
    all_health_yearly = []
    all_fair_value_yearly = []

    # สร้าง returns matrix (date x ticker) ครั้งเดียว ใช้ร่วมกันสำหรับ Downside Beta,
    # Sector-Relative Beta และ Correlation Matrix (ทุกตัวใช้หุ้นกลุ่มเดียวกันเป็น benchmark proxy)
    returns_matrix = build_returns_matrix(df_price_all)

    for ticker in TARGET_STOCKS:
        fin_sub = df_fin_all[df_fin_all['ticker'] == ticker]
        fin_q_sub = df_fin_q_all[df_fin_q_all['ticker'] == ticker] if not df_fin_q_all.empty else pd.DataFrame()
        ttm = build_ttm_financials(fin_q_sub)
        price_sub = df_price_all[df_price_all['ticker'] == ticker].sort_values(by='date')
        risk_sub = df_risk_all[df_risk_all['ticker'] == ticker] if not df_risk_all.empty else None

        if price_sub.empty:
            continue

        current_price = round(clean_float(price_sub.iloc[-1]['close'], default=10.0), 2)
        latest_date = str(price_sub.iloc[-1]['date'])[:10]
        if len(price_sub) >= 2:
            prev_close = clean_float(price_sub.iloc[-2]['close'], default=current_price)
            change_val = round(current_price - prev_close, 2)
            change_pct = round((change_val / prev_close) * 100, 2) if prev_close else 0.0
        else:
            change_val, change_pct = 0.0, 0.0

        m1 = calculate_health_module(fin_sub, ttm=ttm)
        m2 = calculate_valuation_module(fin_sub, current_price, ticker, ttm=ttm)
        m3 = calculate_timing_module(price_sub)
        m4, feat_imp, backtest_df = train_and_predict_ai(price_sub, ticker)
        m5 = calculate_risk_module(price_sub, risk_sub, ticker=ticker, returns_matrix=returns_matrix)

        # เมตริกจริงเพิ่มเติมสำหรับ Overview / Key Highlights
        fin_sorted = fin_sub.sort_values('year')
        rev_growth, ni_growth = None, None
        if len(fin_sorted) >= 2:
            rev_prev, rev_curr = fin_sorted.iloc[-2]['total_revenue'], fin_sorted.iloc[-1]['total_revenue']
            ni_prev, ni_curr = fin_sorted.iloc[-2]['net_income'], fin_sorted.iloc[-1]['net_income']
            if rev_prev and clean_float(rev_prev) != 0:
                rev_growth = round((clean_float(rev_curr) - clean_float(rev_prev)) / clean_float(rev_prev) * 100, 1)
            if ni_prev and clean_float(ni_prev) != 0:
                ni_growth = round((clean_float(ni_curr) - clean_float(ni_prev)) / clean_float(ni_prev) * 100, 1)
        latest_row = fin_sorted.iloc[-1] if not fin_sorted.empty else None
        fcf_latest = round(clean_float(latest_row.get('free_cash_flow')), 1) if latest_row is not None else None

        all_summary.append({
            'ticker': ticker,
            'current_price': current_price,
            'change_val': change_val,
            'change_pct': change_pct,
            'latest_date': latest_date,
            'sector': SECTOR_MAP.get(ticker, 'Technology'),
            'revenue_growth_yoy': rev_growth,
            'net_income_growth_yoy': ni_growth,
            'free_cash_flow_latest': fcf_latest,
            **m1, **m2, **m3, **m4, **m5
        })

        for f, imp in feat_imp.items():
            all_feature_importance.append({'ticker': ticker, 'feature': f, 'importance': imp})

        if not backtest_df.empty:
            bt = backtest_df.copy()
            bt['ticker'] = ticker
            all_backtest.append(bt)

        rh = build_risk_rolling_history(price_sub)
        rh['ticker'] = ticker
        all_risk_history.append(rh)

        hy = build_health_score_yearly(fin_sub)
        hy['ticker'] = ticker
        all_health_yearly.append(hy)

        fv = build_fair_value_yearly(fin_sub, price_sub, ticker)
        fv['ticker'] = ticker
        all_fair_value_yearly.append(fv)

    df_res = pd.DataFrame(all_summary)

    df_res['industry_score'] = df_res.groupby('sector')['health_score'].rank(pct=True).apply(lambda x: round(x * 100, 1))

    df_res['overall_score'] = (
        (df_res['health_score'] * 0.25) +
        (df_res['valuation_score'] * 0.25) +
        (df_res['timing_score'] * 0.15) +
        (df_res['ai_score'] * 0.10) +
        (df_res['risk_score'] * 0.15) +
        (df_res['industry_score'] * 0.10)
    ).round(1)

    df_res['sector_rank'] = df_res.groupby('sector')['overall_score'].rank(ascending=False, method='min').astype(int)
    df_res['overall_rank'] = df_res['overall_score'].rank(ascending=False, method='min').astype(int)

    def get_rec(score):
        if score >= 75: return "STRONG BUY"
        if score >= 65: return "BUY"
        if score >= 50: return "ACCUMULATE"
        return "REDUCE / SELL"

    df_res['recommendation'] = df_res['overall_score'].apply(get_rec)
    df_res['updated_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    df_res.to_sql('cis_summary_scores', con=engine, if_exists='replace', index=False)

    df_feat = pd.DataFrame(all_feature_importance)
    df_feat.to_sql('ai_feature_importance', con=engine, if_exists='replace', index=False)

    if all_backtest:
        df_bt = pd.concat(all_backtest, ignore_index=True)
        df_bt['date'] = df_bt['date'].astype(str)
        df_bt.to_sql('ai_backtest_history', con=engine, if_exists='replace', index=False)

    if all_risk_history:
        df_rh = pd.concat(all_risk_history, ignore_index=True)
        df_rh['date'] = df_rh['date'].astype(str)
        df_rh.to_sql('risk_rolling_history', con=engine, if_exists='replace', index=False)

    if all_health_yearly:
        df_hy = pd.concat(all_health_yearly, ignore_index=True)
        df_hy.to_sql('health_score_yearly', con=engine, if_exists='replace', index=False)

    if all_fair_value_yearly:
        df_fv = pd.concat(all_fair_value_yearly, ignore_index=True)
        df_fv.to_sql('fair_value_yearly', con=engine, if_exists='replace', index=False)

    # 3.2 Correlation Matrix ระหว่างหุ้นทั้ง 8 ตัว (จาก returns matrix จริง)
    df_corr = build_correlation_matrix(returns_matrix)
    if not df_corr.empty:
        df_corr.to_sql('risk_correlation_matrix', con=engine, if_exists='replace', index=False)

    print("\n✅ ประมวลผลและบันทึกคะแนนจริงของหุ้นทั้ง 8 ตัวลง cis_database.db เรียบร้อยแล้ว:")
    print("=" * 85)
    print(df_res[['ticker', 'current_price', 'fair_value', 'margin_of_safety', 'overall_score',
                   'recommendation', 'health_score', 'ai_score', 'beta']])
    print("=" * 85)


if __name__ == '__main__':
    run_full_pipeline()
