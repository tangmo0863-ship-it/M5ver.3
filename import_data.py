"""
import_data.py
----------------
นำเข้าข้อมูลจากไฟล์ Dataset ทั้ง 5 ไฟล์ (งบการเงิน, ราคาหุ้นรายวัน, ความเสี่ยง)
เข้า SQLite (cis_database.db) เพื่อให้ calculate_scores.py และ app.py ใช้งานต่อ

ตาราง (tables) ที่จะถูกสร้าง:
- stock_financials      : งบการเงินปี 2023-2025 ของ 8 หุ้นเป้าหมาย (ตัวเลขถูกแปลงเป็น float แล้ว)
- stock_daily_prices    : ราคาหุ้นรายวัน + technical indicators ปี 2023-2025
- stock_risk_static     : Beta / Volatility / Max Drawdown รายหุ้น จาก stock_risk_metrics.csv
"""

import os
import glob
import pandas as pd
from sqlalchemy import create_engine

DB_NAME = 'cis_database.db'
TARGET_STOCKS = ['ADVANC', 'CCET', 'DELTA', 'HANA', 'JMART', 'KCE', 'THCOM', 'TRUE']
TARGET_YEARS = [2023, 2024, 2025]

SEARCH_DIRS = ['.', 'Dataset', 'Dataset/train_test', 'dataset', 'dataset/train_test']


def find_file(pattern):
    """ค้นหาไฟล์ตาม pattern ในหลาย ๆ โฟลเดอร์ที่เป็นไปได้"""
    for d in SEARCH_DIRS:
        matches = glob.glob(os.path.join(d, pattern))
        if matches:
            return sorted(matches)[0]
    return None


def find_all_files(pattern):
    found = []
    for d in SEARCH_DIRS:
        found.extend(glob.glob(os.path.join(d, pattern)))
    # unique, keep order
    seen = set()
    out = []
    for f in found:
        rp = os.path.abspath(f)
        if rp not in seen:
            seen.add(rp)
            out.append(f)
    return out


def read_csv_smart(path):
    """อ่าน CSV โดยลองหลาย encoding (ไฟล์งบการเงินเป็น cp874 เพราะมีหัวคอลัมน์ภาษาไทย)"""
    for enc in ['utf-8', 'cp874', 'cp1252', 'latin1']:
        try:
            return pd.read_csv(path, encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    # last resort
    return pd.read_csv(path, encoding='utf-8', errors='replace')


def clean_numeric_series(s):
    """แปลงคอลัมน์ตัวเลขที่มี comma / % / ช่องว่าง ให้เป็น float"""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    return (
        s.astype(str)
        .str.replace(',', '', regex=False)
        .str.replace('%', '', regex=False)
        .str.strip()
        .replace({'': None, '-': None, 'nan': None, 'None': None})
        .astype(float)
    )


def import_financial_data(engine):
    print("\n--- 1. กำลังประมวลผลไฟล์งบการเงิน (Financials) ---")

    # ไฟล์หลัก master_all_8_stocks_financials + ไฟล์ train/test (โครงสร้างคอลัมน์เดียวกัน)
    # ใช้ master เป็นหลักเพราะครอบคลุมทั้ง 2023-2025 อยู่แล้ว แต่รวมทุกไฟล์ที่เจอไว้กันตกหล่น
    candidate_files = find_all_files("*master_all_8_stocks_financials*.csv")
    train_test_files = find_all_files("*financials_train*.csv") + find_all_files("*financials_test*.csv")

    if not candidate_files and not train_test_files:
        print("❌ ไม่พบไฟล์งบการเงิน")
        return

    frames = []
    for fp in (candidate_files or train_test_files):
        try:
            frames.append(read_csv_smart(fp))
        except Exception as e:
            print(f"⚠️ อ่านไฟล์ {fp} ไม่ได้: {e}")

    if not frames:
        print("❌ ไม่สามารถอ่านไฟล์งบการเงินได้เลย")
        return

    df = pd.concat(frames, ignore_index=True)
    df.columns = df.columns.str.strip()

    stock_col = [c for c in df.columns if 'Stock' in c][0]
    # หาปี ค.ศ. โดยตรงก่อน ถ้าไม่เจอค่อย fallback ไปแปลงจาก พ.ศ.
    year_ad_col = [c for c in df.columns if 'ค.ศ' in c or c.lower().strip() == 'year']
    year_be_col = [c for c in df.columns if 'พ.ศ' in c]

    df['stock_symbol'] = df[stock_col].astype(str).str.strip().str.upper().str.replace('.BK', '', regex=False)

    def clean_year_generic(val):
        try:
            y = int(float(str(val).split('.')[0].strip()))
            return y - 543 if y > 2500 else y
        except Exception:
            return None

    if year_ad_col:
        df['year_clean'] = df[year_ad_col[0]].apply(clean_year_generic)
    elif year_be_col:
        df['year_clean'] = df[year_be_col[0]].apply(clean_year_generic)
    else:
        print("❌ ไม่พบคอลัมน์ปีในไฟล์งบการเงิน")
        return

    df = df.drop_duplicates(subset=['stock_symbol', 'year_clean'], keep='first')

    filtered_df = df[
        (df['stock_symbol'].isin(TARGET_STOCKS)) &
        (df['year_clean'].isin(TARGET_YEARS))
    ].copy()

    # แปลงคอลัมน์ตัวเลข (ที่มี comma/% ปนอยู่) ให้เป็น float ทั้งหมด
    non_numeric_cols = {stock_col, 'stock_symbol'}
    if year_ad_col:
        non_numeric_cols.add(year_ad_col[0])
    if year_be_col:
        non_numeric_cols.add(year_be_col[0])

    for col in filtered_df.columns:
        if col in non_numeric_cols or col == 'year_clean':
            continue
        try:
            filtered_df[col] = clean_numeric_series(filtered_df[col])
        except Exception:
            pass  # คอลัมน์ที่แปลงไม่ได้ ปล่อยไว้เป็น string เดิม

    rename_mapping = {
        'stock_symbol': 'ticker',
        'year_clean': 'year',
        'Total Revenue': 'total_revenue',
        'Operating Revenue': 'operating_revenue',
        'COGS': 'cogs',
        'Gross Profit': 'gross_profit',
        'EBIT': 'ebit',
        'EBITDA': 'ebitda',
        'Gross Margin (%)': 'gross_margin',
        'Operating Margin (%)': 'operating_margin',
        'EBITDA Margin (%)': 'ebitda_margin',
        'Net Margin (%)': 'net_margin',
        'ROE (%)': 'roe',
        'ROA (%)': 'roa',
        'D/E (x)': 'de_ratio',
        'Current Ratio (x)': 'current_ratio',
        'Quick Ratio (x)': 'quick_ratio',
        'Interest Coverage (x)': 'interest_coverage',
        'OCF_to_NI': 'ocf_to_ni',
        'Free Cash Flow': 'free_cash_flow',
        'Operating CF': 'operating_cash_flow',
        'Capital Expenditure': 'capex',
        'Total Assets': 'total_assets',
        'Total Equity': 'total_equity',
        'Total Liabilities': 'total_liabilities',
        'Current Assets': 'current_assets',
        'Current Liabilities': 'current_liabilities',
        'Cash Equivalent': 'cash_and_equivalents',
        'Inventory': 'inventory',
        'Accounts Receivable': 'accounts_receivable',
        'Accounts Payable': 'accounts_payable',
        'Fixed Assets': 'fixed_assets',
        'INT': 'interest_expense',
        'EPS': 'eps',
        'NI (Parent)': 'net_income',
    }

    save_df = filtered_df.rename(columns=rename_mapping)
    keep_cols = ['ticker', 'year'] + [v for v in rename_mapping.values() if v not in ('ticker', 'year') and v in save_df.columns]
    save_df = save_df[keep_cols].sort_values(['ticker', 'year']).reset_index(drop=True)

    save_df.to_sql('stock_financials', con=engine, if_exists='replace', index=False)
    print(f"✅ นำเข้าข้อมูลงบการเงินสำเร็จ: {len(save_df)} แถว "
          f"(ครอบคลุม {save_df['ticker'].nunique()} หุ้น, ปี {sorted(save_df['year'].unique())})")


def import_price_data(engine):
    print("\n--- 2. กำลังประมวลผลไฟล์ราคาและดัชนีเทคนิคอล (Stock Prices) ---")
    file_path = find_file("*stock_cleaned*.csv")
    if not file_path:
        print("❌ ไม่พบไฟล์ราคาหุ้น")
        return

    df = read_csv_smart(file_path)
    df.columns = df.columns.str.strip()

    ticker_col = [c for c in df.columns if c.lower() in ['ticker', 'stock', 'symbol']][0]
    df['clean_ticker'] = df[ticker_col].astype(str).str.strip().str.upper().str.replace('.BK', '', regex=False)

    date_col = [c for c in df.columns if c.lower() in ['date', 'datetime', 'วันที่']][0]
    # ไฟล์นี้เป็นรูปแบบ M/D/YYYY (สากล) ไม่ใช่ D/M/YYYY
    df['parsed_date'] = pd.to_datetime(df[date_col], errors='coerce', dayfirst=False)

    if df['parsed_date'].notna().any() and df['parsed_date'].dt.year.max() > 2500:
        df['parsed_date'] = df['parsed_date'].apply(
            lambda x: x.replace(year=x.year - 543) if pd.notnull(x) and x.year > 2500 else x
        )

    mask_stock = df['clean_ticker'].isin(TARGET_STOCKS)
    mask_date = (df['parsed_date'] >= '2023-01-01') & (df['parsed_date'] <= '2025-12-31')

    filtered_df = df[mask_stock & mask_date].copy()

    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'EMA20', 'EMA50', 'Volume Avg', 'RSI14', 'MACD', 'ADX']
    for col in numeric_cols:
        if col in filtered_df.columns:
            filtered_df[col] = clean_numeric_series(filtered_df[col])

    filtered_df = filtered_df.sort_values(by=['clean_ticker', 'parsed_date']).reset_index(drop=True)
    filtered_df['date'] = filtered_df['parsed_date'].dt.strftime('%Y-%m-%d')
    filtered_df['ticker'] = filtered_df['clean_ticker']

    drop_cols = ['clean_ticker', 'parsed_date']
    if ticker_col != 'ticker':
        drop_cols.append(ticker_col)
    if date_col != 'date':
        drop_cols.append(date_col)
    filtered_df = filtered_df.drop(columns=drop_cols, errors='ignore')

    filtered_df.to_sql('stock_daily_prices', con=engine, if_exists='replace', index=False)

    if len(filtered_df) > 0:
        print(f"✅ นำเข้าข้อมูลราคาหุ้นสำเร็จ: {len(filtered_df)} แถว "
              f"(ครอบคลุม {filtered_df['ticker'].nunique()} หุ้น, "
              f"วันที่ {filtered_df['date'].min()} ถึง {filtered_df['date'].max()})")
    else:
        print("⚠️ ไม่พบแถวที่ตรงตามเงื่อนไข ลองตรวจตัวอย่างข้อมูล:")
        print("ตัวอย่าง Ticker ในไฟล์:", df[ticker_col].unique()[:5])
        print("ตัวอย่าง Date ในไฟล์:", df[date_col].head(3).tolist())


def import_quarterly_financial_data(engine):
    """(Optional) นำเข้างบการเงินรายไตรมาส สำหรับคำนวณ TTM (Trailing Twelve Months)
    ใช้ให้ Fair Value / Company Health สดขึ้น แทนการพึ่งงบรายปีล่าสุดที่อาจเก่าถึง 15 เดือน

    ค้นหาไฟล์ pattern: *financials_quarterly*.csv หรือ *quarterly_financials*.csv ใน SEARCH_DIRS
    ถ้าไม่พบไฟล์ -> ข้ามไปเงียบๆ (ไม่ error) ระบบจะ fallback ไปใช้งบรายปีตามปกติ

    Schema ที่คาดหวัง (คอลัมน์ต้องมีอย่างน้อยเท่านี้ ชื่อคอลัมน์ตรงหรือใกล้เคียงกับที่ระบุ):
        Stock, Year (ค.ศ.), Quarter (Q1/Q2/Q3/Q4), Total Revenue, NI (Parent), EPS,
        Operating CF, Capital Expenditure, Free Cash Flow, Total Assets, Total Equity,
        Total Liabilities, Current Assets, Current Liabilities, Cash Equivalent, EBIT, EBITDA
    (คอลัมน์อื่นที่มีในไฟล์รายปีอยู่แล้วก็ใส่มาได้ จะถูก map ชื่อแบบเดียวกัน)
    """
    print("\n--- 1b. กำลังตรวจสอบไฟล์งบการเงินรายไตรมาส (สำหรับ TTM, ไม่บังคับ) ---")
    candidate_files = find_all_files("*financials_quarterly*.csv") + find_all_files("*quarterly_financials*.csv")
    if not candidate_files:
        print("  ⏭️  ไม่พบไฟล์งบรายไตรมาส -> ข้าม (ระบบจะใช้งบรายปีล่าสุดตามปกติ ไม่ error)")
        return False

    frames = []
    for fp in candidate_files:
        try:
            frames.append(read_csv_smart(fp))
        except Exception as e:
            print(f"  ⚠️ อ่านไฟล์ {fp} ไม่ได้: {e}")
    if not frames:
        return False

    df = pd.concat(frames, ignore_index=True)
    df.columns = df.columns.str.strip()

    stock_col = [c for c in df.columns if 'Stock' in c][0]
    quarter_col = [c for c in df.columns if c.strip().lower() in ['quarter', 'ไตรมาส', 'q']]
    year_ad_col = [c for c in df.columns if 'ค.ศ' in c or c.lower().strip() == 'year']
    year_be_col = [c for c in df.columns if 'พ.ศ' in c]

    if not quarter_col:
        print("  ⚠️ ไม่พบคอลัมน์ Quarter ในไฟล์งบรายไตรมาส -> ข้าม")
        return False

    df['stock_symbol'] = df[stock_col].astype(str).str.strip().str.upper().str.replace('.BK', '', regex=False)
    df['quarter'] = df[quarter_col[0]].astype(str).str.strip().str.upper().str.extract(r'(Q?[1-4])')[0].str.replace('Q', '', regex=False)

    def clean_year_generic(val):
        try:
            y = int(float(str(val).split('.')[0].strip()))
            return y - 543 if y > 2500 else y
        except Exception:
            return None

    if year_ad_col:
        df['year_clean'] = df[year_ad_col[0]].apply(clean_year_generic)
    elif year_be_col:
        df['year_clean'] = df[year_be_col[0]].apply(clean_year_generic)
    else:
        print("  ⚠️ ไม่พบคอลัมน์ปีในไฟล์งบรายไตรมาส -> ข้าม")
        return False

    filtered_df = df[df['stock_symbol'].isin(TARGET_STOCKS)].copy()
    filtered_df = filtered_df.drop_duplicates(subset=['stock_symbol', 'year_clean', 'quarter'], keep='first')

    non_numeric_cols = {stock_col, 'stock_symbol', 'quarter'}
    if year_ad_col:
        non_numeric_cols.add(year_ad_col[0])
    if year_be_col:
        non_numeric_cols.add(year_be_col[0])
    if quarter_col:
        non_numeric_cols.add(quarter_col[0])

    for col in filtered_df.columns:
        if col in non_numeric_cols or col == 'year_clean':
            continue
        try:
            filtered_df[col] = clean_numeric_series(filtered_df[col])
        except Exception:
            pass

    rename_mapping = {
        'stock_symbol': 'ticker', 'year_clean': 'year', 'quarter': 'quarter',
        'Total Revenue': 'total_revenue', 'NI (Parent)': 'net_income', 'EPS': 'eps',
        'EBIT': 'ebit', 'EBITDA': 'ebitda', 'Operating CF': 'operating_cash_flow',
        'Capital Expenditure': 'capex', 'Free Cash Flow': 'free_cash_flow',
        'Total Assets': 'total_assets', 'Total Equity': 'total_equity', 'Total Liabilities': 'total_liabilities',
        'Current Assets': 'current_assets', 'Current Liabilities': 'current_liabilities',
        'Cash Equivalent': 'cash_and_equivalents', 'INT': 'interest_expense',
    }
    save_df = filtered_df.rename(columns=rename_mapping)
    keep_cols = ['ticker', 'year', 'quarter'] + [v for v in rename_mapping.values() if v not in ('ticker', 'year', 'quarter') and v in save_df.columns]
    save_df = save_df[keep_cols].sort_values(['ticker', 'year', 'quarter']).reset_index(drop=True)

    save_df.to_sql('stock_financials_quarterly', con=engine, if_exists='replace', index=False)
    print(f"  ✅ นำเข้างบรายไตรมาสสำเร็จ: {len(save_df)} แถว ({save_df['ticker'].nunique()} หุ้น) -> ใช้คำนวณ TTM ได้")
    return True


def import_risk_metrics(engine):
    print("\n--- 3. กำลังประมวลผลไฟล์ Risk Metrics (Beta / Volatility / Max Drawdown) ---")
    file_path = find_file("*stock_risk_metrics*.csv")
    if not file_path:
        print("❌ ไม่พบไฟล์ stock_risk_metrics")
        return

    df = read_csv_smart(file_path)
    df.columns = df.columns.str.strip()

    stock_col = [c for c in df.columns if c.lower() in ['stock', 'ticker', 'symbol']][0]
    df['ticker'] = df[stock_col].astype(str).str.strip().str.upper().str.replace('.BK', '', regex=False)

    beta_col = [c for c in df.columns if 'beta' in c.lower()][0]
    vol_col = [c for c in df.columns if 'volatility' in c.lower()][0]
    dd_col = [c for c in df.columns if 'drawdown' in c.lower()][0]

    out = pd.DataFrame({
        'ticker': df['ticker'],
        'beta': clean_numeric_series(df[beta_col]),
        'volatility_pct': clean_numeric_series(df[vol_col]),
        'max_drawdown_pct': clean_numeric_series(df[dd_col]),
    })
    out = out[out['ticker'].isin(TARGET_STOCKS)].drop_duplicates(subset=['ticker']).reset_index(drop=True)

    out.to_sql('stock_risk_static', con=engine, if_exists='replace', index=False)
    print(f"✅ นำเข้าข้อมูลความเสี่ยงสำเร็จ: {len(out)} หุ้น")


if __name__ == '__main__':
    engine = create_engine(f'sqlite:///{DB_NAME}')
    import_financial_data(engine)
    import_quarterly_financial_data(engine)
    import_price_data(engine)
    import_risk_metrics(engine)
    print("\n" + "=" * 60)
    print("นำเข้าข้อมูลทั้งหมดเสร็จสมบูรณ์ -> รันต่อด้วย: python calculate_scores.py")
