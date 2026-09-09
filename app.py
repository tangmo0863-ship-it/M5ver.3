import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sqlalchemy import create_engine, text
from datetime import datetime

# ==========================================
# 1. ตั้งค่าหน้าเว็บ & ธีม Dark Navy
# ==========================================
st.set_page_config(
    page_title="CIS - Comprehensive Investment System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    html, body, [class*="css"] { font-size: 16px; }
    .stApp { background-color: #0B1120; }
    .metric-card { background-color: #151E2F; padding: 16px; border-radius: 12px; border: 1px solid #1E293B; text-align: center; height: 100%; }
    .hero-card { background-color: #0F172A; padding: 20px; border-radius: 16px; border: 1px solid #1E293B; }
    .dim-card { background-color: #151E2F; border: 1px solid #1E293B; border-radius: 10px; padding: 12px; text-align: center; }

    /* ปุ่มทั้งหมดในแอป (Previous/Home/Next) ให้เข้ากับธีมมืด อ่านง่าย กดง่าย */
    .stButton > button {
        background-color: #151E2F !important;
        border: 1px solid #2A3A55 !important;
        color: #F1F5F9 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 14.5px !important;
        padding: 10px 18px !important;
        transition: all 0.2s ease-in-out !important;
    }
    .stButton > button:hover {
        border-color: #10B981 !important;
        color: #10B981 !important;
        background-color: rgba(16,185,129,0.10) !important;
    }
    .stButton > button p { font-size: 14.5px !important; font-weight: 600 !important; }

    /* --- Sidebar: พื้นหลังเข้ม ตัวหนังสือต้องสว่างพอให้อ่านออก --- */
    [data-testid="stSidebar"] { background-color: #0F172A; }
    [data-testid="stSidebar"] * { font-size: 15px; }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label > div:first-child,
    [data-testid="stSidebar"] [data-testid="stRadio"] input[type="radio"],
    [data-testid="stSidebar"] [data-testid="stRadio"] svg { display: none !important; width: 0 !important; height: 0 !important; margin: 0 !important; }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] { gap: 7px !important; }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label {
        background-color: #151E2F !important; border: 1px solid #1E293B !important; border-radius: 8px !important;
        padding: 11px 14px !important; margin: 0 !important; cursor: pointer !important; width: 100% !important;
        display: flex !important; align-items: center !important; transition: all 0.2s ease-in-out !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:hover {
        background-color: rgba(45, 212, 191, 0.10) !important; border-color: rgba(45, 212, 191, 0.35) !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
        background: rgba(16, 185, 129, 0.22) !important; border: 1.5px solid #10B981 !important; box-shadow: 0 0 10px rgba(16, 185, 129, 0.18) !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label p {
        font-size: 15px !important; color: #E2E8F0 !important; font-weight: 600 !important; margin: 0 !important; line-height: 1.4 !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) p {
        color: #10B981 !important; font-weight: 800 !important;
    }
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] .stCaption, [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #CBD5E1 !important;
    }
    [data-testid="stSidebar"] [data-testid="stSelectbox"] label p { color: #CBD5E1 !important; font-size: 14px !important; font-weight: 600 !important; }
</style>
""", unsafe_allow_html=True)

TARGET_STOCKS = ['ADVANC', 'CCET', 'DELTA', 'HANA', 'JMART', 'KCE', 'THCOM', 'TRUE']
SECTOR_MAP = {
    'ADVANC': 'Technology & Telecomm', 'TRUE': 'Technology & Telecomm', 'THCOM': 'Technology & Telecomm',
    'DELTA': 'Electronic Components', 'HANA': 'Electronic Components', 'KCE': 'Electronic Components',
    'CCET': 'Electronic Components', 'JMART': 'Commerce & Technology'
}
COMPANY_NAMES = {
    'ADVANC': 'Advanced Info Service PCL', 'CCET': 'Cal-Comp Electronics PCL', 'DELTA': 'Delta Electronics (Thailand) PCL',
    'HANA': 'Hana Microelectronics PCL', 'JMART': 'Jaymart Group Holdings PCL', 'KCE': 'KCE Electronics PCL',
    'THCOM': 'Thaicom PCL', 'TRUE': 'True Corporation PCL'
}

# ค่าคงที่แสดงผลสำหรับหน้า Risk Analysis (ต้องตรงกับค่าที่ใช้จริงใน calculate_scores.py)
RISK_FREE_RATE_PCT = 1.2          # risk-free rate ต่อปี (%) — ThaiBMA 1Y Gov Bond, ดู calculate_scores.RISK_FREE_RATE
PSR_BENCHMARK_SR_DISPLAY = "0"    # Benchmark Sharpe Ratio (SR*) สำหรับทดสอบ PSR
MARKET_CRASH_SHOCK_PCT = -20      # สมมติฐานการช็อกตลาด (SET Index) สำหรับ Beta-based crash scenario


def fmt_mb(x, unit="MB"):
    """แปลงตัวเลขบาทดิบ ให้เป็นหน่วยล้านบาท พร้อม comma"""
    try:
        return f"{float(x)/1e6:,.1f} {unit}"
    except Exception:
        return "-"


def safe(v, default=0.0):
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return default
        return float(v)
    except Exception:
        return default


def create_gauge(score, title, color_hex):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        number={'font': {'size': 38, 'color': 'white'}},
        title={'text': f"<br><span style='font-size:15px;color:#94A3B8'>{title}</span>", 'font': {'size': 14}},
        gauge={'axis': {'range': [None, 100], 'visible': False}, 'bar': {'color': color_hex, 'thickness': 0.85},
               'bgcolor': "rgba(255,255,255,0.05)", 'borderwidth': 0}
    ))
    fig.update_layout(height=170, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)")
    return fig


@st.dialog("ขยายกราฟ", width="large")
def _open_chart_dialog(fig, expand_height):
    big_fig = go.Figure(fig)
    big_fig.update_layout(height=expand_height)
    try:
        big_fig.update_xaxes(tickfont=dict(size=13))
        big_fig.update_yaxes(tickfont=dict(size=13))
    except Exception:
        pass
    st.plotly_chart(big_fig, use_container_width=True, config={'displayModeBar': True}, key=f"dlg_{id(fig)}")


def show_chart(fig, key, expand_height=680):
    """แสดงกราฟ Plotly พร้อมปุ่ม '🔍 ขยายกราฟ' ที่เปิดกราฟเวอร์ชันใหญ่ในหน้าต่างลอย (dialog)"""
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}, key=f"{key}_small")
    if st.button("🔍 ขยายกราฟ", key=f"{key}_expand_btn", use_container_width=True):
        _open_chart_dialog(fig, expand_height)


def render_nav_footer(key_prefix, prev_page=None, next_page=None):
    """แถบปุ่มนำทาง (หน้าก่อนหน้า / หน้าหลัก / หน้าถัดไป) แสดงท้ายทุกหน้าโมดูล จัด layout ให้มีระยะห่างจากกรอบด้านบนพอดี"""
    st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)
    st.markdown("""<div style="border-top:1px solid #1E293B; padding-top:16px; margin-bottom:6px;"></div>""", unsafe_allow_html=True)
    col_prev, col_home, col_next, col_disc = st.columns([1.3, 1.3, 1.3, 3.3])
    with col_prev:
        if prev_page:
            if st.button("⬅ หน้าก่อนหน้า", key=f"btn_prev_{key_prefix}", use_container_width=True):
                st.session_state["pending_nav"] = prev_page
                st.rerun()
    with col_home:
        if st.button("🏠 หน้าหลัก", key=f"btn_home_{key_prefix}", use_container_width=True):
            st.session_state["pending_nav"] = " 🏠 Overview"
            st.rerun()
    with col_next:
        if next_page:
            if st.button("หน้าถัดไป ➡", key=f"btn_next_{key_prefix}", use_container_width=True):
                st.session_state["pending_nav"] = next_page
                st.rerun()
    with col_disc:
        st.markdown(
            '<div style="font-size:15.5px; color:#94A3B8; text-align:right; padding-top:11px; line-height:1.5;">'
            'หมายเหตุ: การประเมินนี้ไม่ใช่คำแนะนำในการลงทุน ผู้ลงทุนควรศึกษาข้อมูลเพิ่มเติม</div>',
            unsafe_allow_html=True
        )


# ==========================================
# 2. เชื่อมต่อฐานข้อมูล SQLite (ข้อมูลจริงทั้งหมด จาก import_data.py + calculate_scores.py)
# ==========================================
@st.cache_resource
def get_connection():
    return create_engine("sqlite:///cis_database.db")


def database_is_ready():
    """เช็คว่าฐานข้อมูลมีตารางผลลัพธ์พร้อมใช้งานหรือไม่"""
    try:
        engine = get_connection()
        with engine.connect() as conn:
            conn.execute(text("SELECT ticker FROM cis_summary_scores LIMIT 1"))
        return True
    except Exception:
        return False


def build_database():
    """รัน import_data.py + calculate_scores.py อัตโนมัติ กรณีฐานข้อมูลหาย/ว่างเปล่า
    (เช่น ตอน deploy บน Streamlit Cloud แล้ว .db ไม่ได้ถูกอัปโหลดไปด้วย หรือ container ถูกสร้างใหม่)"""
    import os
    import import_data
    import calculate_scores

    # ตรวจสอบก่อนว่ามีไฟล์ Dataset ครบไหม เพื่อให้ error ชัดเจนว่าปัญหาอยู่ตรงไหนจริง ๆ
    missing = []
    if not import_data.find_file("*master_all_8_stocks_financials*.csv") and not import_data.find_all_files("*financials_train*.csv"):
        missing.append("งบการเงิน (master_all_8_stocks_financials*.csv หรือ financials_train/test*.csv)")
    if not import_data.find_file("*stock_cleaned*.csv"):
        missing.append("ราคาหุ้นรายวัน (stock_cleaned_data*.csv)")
    if not import_data.find_file("*stock_risk_metrics*.csv"):
        missing.append("ความเสี่ยง (stock_risk_metrics*.csv)")

    if missing:
        cwd_listing = "\n".join(f"  - {f}" for f in sorted(os.listdir("."))) if os.path.isdir(".") else "(อ่านโฟลเดอร์ปัจจุบันไม่ได้)"
        ds_listing = "\n".join(f"  - {f}" for f in sorted(os.listdir("Dataset"))) if os.path.isdir("Dataset") else "  (ไม่พบโฟลเดอร์ Dataset/ เลย)"
        raise FileNotFoundError(
            "ไม่พบไฟล์ข้อมูลต่อไปนี้ใน repo: " + "; ".join(missing) +
            f"\n\nไฟล์/โฟลเดอร์ใน working directory ปัจจุบัน:\n{cwd_listing}" +
            f"\n\nไฟล์ใน Dataset/ ที่เจอ:\n{ds_listing}" +
            "\n\n➡️ กรุณาตรวจสอบว่าโฟลเดอร์ Dataset/ (พร้อมไฟล์ CSV ทั้ง 5 ไฟล์) ถูก push ขึ้น GitHub จริง "
            "(เช็คได้จากหน้า repo บน GitHub ว่าเห็นโฟลเดอร์ Dataset/ และไฟล์ .csv ข้างในหรือไม่ "
            "และเช็ค .gitignore ว่าไม่ได้ ignore *.csv หรือทั้งโฟลเดอร์ Dataset/ อยู่)"
        )

    engine = get_connection()
    import_data.import_financial_data(engine)
    import_data.import_quarterly_financial_data(engine)
    import_data.import_price_data(engine)
    import_data.import_risk_metrics(engine)
    calculate_scores.run_full_pipeline()


@st.cache_data(ttl=600)
def load_all_data():
    engine = get_connection()
    scores_df = pd.read_sql("SELECT * FROM cis_summary_scores ORDER BY ticker", engine)
    daily_df = pd.read_sql("SELECT * FROM stock_daily_prices ORDER BY ticker, date", engine)
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    fin_df = pd.read_sql("SELECT * FROM stock_financials ORDER BY ticker, year", engine)
    feat_imp_df = pd.read_sql("SELECT * FROM ai_feature_importance", engine)
    try:
        backtest_df = pd.read_sql("SELECT * FROM ai_backtest_history", engine)
        backtest_df['date'] = pd.to_datetime(backtest_df['date'])
    except Exception:
        backtest_df = pd.DataFrame()
    try:
        risk_hist_df = pd.read_sql("SELECT * FROM risk_rolling_history", engine)
        risk_hist_df['date'] = pd.to_datetime(risk_hist_df['date'])
    except Exception:
        risk_hist_df = pd.DataFrame()
    try:
        health_yearly_df = pd.read_sql("SELECT * FROM health_score_yearly", engine)
    except Exception:
        health_yearly_df = pd.DataFrame()
    try:
        fair_value_yearly_df = pd.read_sql("SELECT * FROM fair_value_yearly", engine)
    except Exception:
        fair_value_yearly_df = pd.DataFrame()
    try:
        risk_static_df = pd.read_sql("SELECT * FROM stock_risk_static", engine)
    except Exception:
        risk_static_df = pd.DataFrame()
    try:
        corr_matrix_df = pd.read_sql("SELECT * FROM risk_correlation_matrix", engine)
    except Exception:
        corr_matrix_df = pd.DataFrame()
    return (scores_df, daily_df, fin_df, feat_imp_df, backtest_df, risk_hist_df, health_yearly_df,
            fair_value_yearly_df, risk_static_df, corr_matrix_df)


if not database_is_ready():
    with st.spinner("⏳ กำลังประมวลผลข้อมูลครั้งแรก (import + calculate scores)... อาจใช้เวลาสักครู่"):
        try:
            build_database()
            st.cache_data.clear()
        except Exception as build_err:
            st.error(f"⚠️ สร้างฐานข้อมูลอัตโนมัติไม่สำเร็จ: {build_err}\n\n"
                     f"กรุณาตรวจสอบว่าโฟลเดอร์ `Dataset/` ถูกอัปโหลดขึ้น GitHub ครบถ้วน "
                     f"หรือรัน `python import_data.py` แล้วตามด้วย `python calculate_scores.py` เองก่อนเปิด Dashboard")
            st.stop()

try:
    (scores_df, daily_df, fin_df, feat_imp_df, backtest_df, risk_hist_df,
     health_yearly_df, fair_value_yearly_df, risk_static_df, corr_matrix_df) = load_all_data()
    if scores_df.empty:
        raise ValueError("cis_summary_scores ว่างเปล่า")
except Exception as e:
    st.error(f"⚠️ ไม่สามารถโหลดข้อมูลจากฐานข้อมูลได้: {e}\n\nกรุณารัน `python import_data.py` แล้วตามด้วย `python calculate_scores.py` ก่อนเปิด Dashboard")
    st.stop()

# ==========================================
# 3. Sidebar Navigation
# ==========================================
st.sidebar.markdown("""
<div style="padding: 8px 0 14px 0;">
    <div style="display: flex; align-items: flex-start; gap: 10px; min-width: 0;">
        <span style="font-size: 26px; flex-shrink: 0; line-height:1.3;">🧠</span>
        <span style="font-size: 15px; font-weight: 800; color: #F8FAFC; letter-spacing: -0.1px; line-height: 1.3; min-width: 0; word-break: break-word; overflow-wrap: break-word;">
            Comprehensive Investment System
        </span>
    </div>
    <div style="font-size: 12px; color: #94A3B8; margin-top: 4px; padding-left: 36px; letter-spacing: 0.3px; font-weight: 600;">
        Investment Decision Support
    </div>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("---")

PAGES = [
    " 🏠 Overview", " 💚 Company Health", " ⚖️ Fair Value",
    " ⏱️ Entry Timing", " 🔮 AI Prediction", " 🛡️ Risk Analysis", " 📊 Industry Benchmark"
]

if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = PAGES[0]

# ปุ่ม "หน้าก่อนหน้า/หน้าหลัก/หน้าถัดไป" จะฝากคำขอเปลี่ยนหน้าไว้ใน pending_nav แล้ว rerun
# ต้องอัปเดตค่าให้ widget ตรงนี้ (ก่อน radio ถูกสร้าง) เพราะ Streamlit ไม่อนุญาตให้แก้ session_state
# ของ key ที่ widget ใช้อยู่ หลังจาก widget ถูกสร้างไปแล้วในรอบเดียวกัน
if "pending_nav" in st.session_state:
    st.session_state["nav_page"] = st.session_state.pop("pending_nav")

st.sidebar.radio("Navigation", PAGES, key="nav_page")
nav_page = st.session_state["nav_page"]

st.sidebar.markdown("---")
selected_ticker = st.sidebar.selectbox("Search Company...", scores_df['ticker'].unique())
stock_info = scores_df[scores_df['ticker'] == selected_ticker].iloc[0].to_dict()
st.sidebar.caption(f"📅 ราคาหุ้น ณ วันที่ล่าสุดในชุดข้อมูล: **{scores_df['latest_date'].max()}**\n\n📑 งบการเงินที่ใช้คำนวณล่าสุด: **{stock_info.get('financials_period_label','-')}**\n\n(ราคาทั้งหมดอ้างอิงจากไฟล์ Dataset ไม่ใช่ราคาตลาดสด)")

stock_daily = daily_df[daily_df['ticker'] == selected_ticker].sort_values('date').reset_index(drop=True)
fin_stock = fin_df[fin_df['ticker'] == selected_ticker].sort_values('year').reset_index(drop=True)
sector_peers = scores_df[scores_df['sector'] == stock_info['sector']]

current_price = safe(stock_info.get('current_price'))
change_pct = safe(stock_info.get('change_pct'))
change_val = safe(stock_info.get('change_val'))
change_color = "#10B981" if change_pct >= 0 else "#EF4444"
change_sign = "+" if change_pct >= 0 else ""
arrow_sign = "▲" if change_pct >= 0 else "▼"

# ==========================================
# HEADER BAR
# ==========================================
st.markdown(f"""
<div style="display:flex; justify-content:space-between; align-items:center; background-color:#0F172A; padding:14px 24px; border-radius:12px; border:1px solid #1E293B; margin-bottom:20px;">
    <div>
        <span style="font-size:24px; font-weight:bold; color:white;">{selected_ticker}</span>
        <span style="color:#94A3B8; font-size:16px; margin-left:8px;">{stock_info.get('sector','-')} (SET) ☆</span>
    </div>
    <div style="font-size:16.5px; color:#94A3B8;">
        Price: <b style="color:white; font-size:19px;">{current_price:.2f}</b> THB
        <span style="color:{change_color}; font-weight:bold; margin-left:6px;">({change_sign}{change_pct:.2f}%) {arrow_sign}</span>
        <span style="margin: 0 12px; color:#334155;">|</span>
        P/E: <b style="color:white;">{stock_info.get('pe_ratio','-')}x</b>
        <span style="margin: 0 12px; color:#334155;">|</span>
        ROE: <b style="color:white;">{stock_info.get('roe','-')}%</b>
        <span style="margin: 0 12px; color:#334155;">|</span>
        Data as of: <b style="color:#F59E0B;">{stock_info.get('latest_date','-')}</b>
        <span style="margin: 0 12px; color:#334155;">|</span>
        งบการเงิน: <b style="color:#38BDF8;">{stock_info.get('financials_period_label','-')}</b>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# PAGE 1: OVERVIEW DASHBOARD
# ============================================================================
if "Overview" in nav_page:
    st.markdown(f"""<div style="display:flex; justify-content:space-between; align-items:flex-end; margin-bottom:15px;">
<div>
<h2 style="margin:0; font-size:23px; font-weight:bold; color:#F8FAFC; letter-spacing:0.5px;">OVERVIEW DASHBOARD</h2>
<div style="font-size:15px; color:#94A3B8; margin-top:2px;">AI-Powered Investment Decision Support System</div>
</div>
<div style="display:flex; align-items:center; gap:15px;">
<div style="font-size:14.5px; color:#94A3B8;">Data as of: <b style="color:#CBD5E1;">{stock_info.get('latest_date','-')}</b></div>
<div style="background-color:#151E2F; border:1px solid #1E293B; border-radius:8px; padding:4px 12px; font-size:14.5px; color:#F8FAFC; display:flex; align-items:center; gap:6px;">
<span>🇹🇭</span> <b>Thai Stock Market</b>
</div>
</div>
</div>""", unsafe_allow_html=True)

    col_left, col_center, col_right = st.columns([1.1, 2.3, 1.2])

    with col_left:
        st.markdown(f"""<div style="background-color:#0F172A; border:1px solid #1E293B; border-radius:12px 12px 0 0; padding:16px 16px 8px 16px;">
<div style="display:flex; justify-content:space-between; align-items:center;">
<span style="font-size:23px; font-weight:bold; color:#FFFFFF;">{selected_ticker}</span>
<span style="color:#64748B; font-size:18.5px;">☆</span>
</div>
<div style="font-size:14.5px; color:#94A3B8; margin-top:2px;">{COMPANY_NAMES.get(selected_ticker,'-')}</div>
<div style="display:flex; align-items:baseline; gap:8px; margin-top:10px;">
<span style="font-size:28px; font-weight:bold; color:#FFFFFF; line-height:1;">{current_price:.2f}</span>
<span style="font-size:14.5px; color:#94A3B8;">THB</span>
</div>
<div style="font-size:15px; font-weight:bold; color:{change_color}; margin-top:4px;">{change_sign}{change_val:.2f} ({change_sign}{change_pct:.2f}%) {arrow_sign}</div>
<div style="font-size:13px; color:#64748B; margin-top:4px;">Dataset close &bull; {stock_info.get('latest_date','-')}</div>
</div>""", unsafe_allow_html=True)

        # Sparkline จากราคาปิดจริงย้อนหลัง 90 วันทำการ
        spark = stock_daily.tail(90)
        fig_mini = go.Figure()
        fig_mini.add_trace(go.Scatter(
            x=spark['date'], y=spark['close'], mode='lines',
            line=dict(color='#10B981', width=1.5), fill='tozeroy', fillcolor='rgba(16, 185, 129, 0.08)', hoverinfo='skip'
        ))
        fig_mini.update_layout(
            height=125, margin=dict(l=8, r=8, t=0, b=0), paper_bgcolor="#0F172A", plot_bgcolor="#0F172A",
            xaxis=dict(showgrid=False, showticklabels=True, tickfont=dict(size=11, color="#64748B"), nticks=4, linecolor="#1E293B"),
            yaxis=dict(showgrid=False, showticklabels=False)
        )
        st.plotly_chart(fig_mini, use_container_width=True, config={'displayModeBar': False})

        n_sector = len(sector_peers)
        fcf_yield = (safe(stock_info.get('free_cash_flow_latest')) / (safe(stock_info.get('market_cap_mb')) * 1e6) * 100) if safe(stock_info.get('market_cap_mb')) > 0 else 0
        st.markdown(f"""<div style="background-color:#0F172A; border:1px solid #1E293B; border-top:none; border-radius:0 0 12px 12px; padding:8px 16px 16px 16px;">
<div style="display:grid; grid-template-columns: 1fr 1fr; gap:10px; border-top:1px solid #1E293B; padding-top:10px;">
<div><div style="font-size:13px; color:#64748B;">Market Cap</div><div style="font-size:15px; font-weight:bold; color:#F8FAFC; margin-top:2px;">{fmt_mb(safe(stock_info.get('market_cap_mb'))*1e6)}</div></div>
<div><div style="font-size:13px; color:#64748B;">P/E (TTM)</div><div style="font-size:15px; font-weight:bold; color:#F8FAFC; margin-top:2px;">{stock_info.get('pe_ratio','-')}x</div></div>
<div><div style="font-size:13px; color:#64748B;">Sector</div><div style="font-size:15px; font-weight:bold; color:#F8FAFC; margin-top:2px;">{stock_info.get('sector','-').split(' ')[0]}</div></div>
<div><div style="font-size:13px; color:#64748B;">P/B (TTM)</div><div style="font-size:15px; font-weight:bold; color:#F8FAFC; margin-top:2px;">{stock_info.get('pb_ratio','-')}x</div></div>
<div><div style="font-size:13px; color:#64748B;">Industry</div><div style="font-size:14.5px; font-weight:bold; color:#F8FAFC; margin-top:2px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{stock_info.get('sector','-')}</div></div>
<div><div style="font-size:13px; color:#64748B;">FCF Yield</div><div style="font-size:15px; font-weight:bold; color:#10B981; margin-top:2px;">{fcf_yield:.2f}%</div></div>
</div>
</div>""", unsafe_allow_html=True)

    with col_center:
        m1_s = int(round(safe(stock_info.get('health_score'), 50)))
        m2_s = int(round(safe(stock_info.get('valuation_score'), 50)))
        m3_s = int(round(safe(stock_info.get('timing_score'), 50)))
        m4_s = int(round(safe(stock_info.get('ai_score'), 50)))
        m5_s = int(round(safe(stock_info.get('risk_score'), 50)))
        m6_s = int(round(safe(stock_info.get('industry_score'), 50)))

        if m1_s >= 70: m1_badge, m1_desc = "EXCELLENT", "Strong balance sheet and sustainable quality"
        elif m1_s >= 45: m1_badge, m1_desc = "MODERATE", "Stable financial position with sound liquidity"
        else: m1_badge, m1_desc = "WEAK", "Elevated debt leverage or margin pressure"

        if m2_s >= 70: m2_badge, m2_desc = "UNDERVALUED", "Attractive valuation with high margin of safety"
        elif m2_s >= 45: m2_badge, m2_desc = "FAIR VALUE", "Trading near assessed fundamental value"
        else: m2_badge, m2_desc = "OVERVALUED", "Price trades at premium to fair valuation"

        if m3_s >= 65: m3_badge, m3_desc = "BULLISH", "Strong upward momentum across moving averages"
        elif m3_s >= 45: m3_badge, m3_desc = "NEUTRAL", "Consolidating near key technical support"
        else: m3_badge, m3_desc = "BEARISH", "Downtrend momentum; elevated pullback risk"

        if m4_s >= 65: m4_badge, m4_desc = "POSITIVE", "AI model forecasts favorable upside probability"
        elif m4_s >= 45: m4_badge, m4_desc = "NEUTRAL", "AI predicts range-bound price consolidation"
        else: m4_badge, m4_desc = "CAUTION", "Low upside probability under current features"

        if m5_s >= 65: m5_badge, m5_desc = "LOW RISK", "High resilience with stable volatility"
        elif m5_s >= 45: m5_badge, m5_desc = "MODERATE", "Balanced market risk profile"
        else: m5_badge, m5_desc = "HIGH RISK", "Higher volatility and deeper drawdown risk"

        if m6_s >= 70: m6_badge, m6_desc = "OUTPERFORM", "Leading peer group across key industry metrics"
        elif m6_s >= 45: m6_badge, m6_desc = "PARITY", "Performing on par with sectoral median"
        else: m6_badge, m6_desc = "LAGGING", "Trailing behind sectoral benchmark"

        def module_card(num, icon, label, score, color, badge, desc, badge_bg):
            return f"""<div style="background-color:#151E2F; border:1px solid #1E293B; border-radius:10px; padding:16px 14px; text-align:center; position:relative;">
<div style="position:absolute; top:10px; left:10px; background:{badge_bg}; color:{color}; font-size:14px; font-weight:bold; padding:3px 7px; border-radius:5px;">{num}</div>
<div style="display:flex; justify-content:center; align-items:center; gap:6px; margin-bottom:10px;">
<span style="font-size:18px;">{icon}</span><span style="font-size:15px; font-weight:bold; color:#F8FAFC;">{label}</span>
</div>
<div style="margin:0 auto 10px auto; width:88px; height:88px; border-radius:50%; background:conic-gradient({color} 0% {score}%, #1E293B {score}% 100%); display:flex; align-items:center; justify-content:center;">
<div style="width:72px; height:72px; border-radius:50%; background-color:#151E2F; display:flex; flex-direction:column; align-items:center; justify-content:center;">
<span style="font-size:20px; font-weight:bold; color:#FFFFFF; line-height:1;">{score}</span><span style="font-size:13.5px; color:#94A3B8;">100</span>
</div></div>
<div style="color:{color}; font-size:15.5px; font-weight:bold; margin-bottom:5px;">{badge}</div>
<div style="font-size:14.5px; color:#CBD5E1; line-height:1.4;">{desc}</div>
</div>"""

        cards_html = "".join([
            module_card("01", "💚", "COMPANY HEALTH", m1_s, "#34D399", m1_badge, m1_desc, "rgba(16,185,129,0.2)"),
            module_card("02", "⚖️", "FAIR VALUE", m2_s, "#FBBF24", m2_badge, m2_desc, "rgba(245,158,11,0.2)"),
            module_card("03", "⏱️", "ENTRY TIMING", m3_s, "#38BDF8", m3_badge, m3_desc, "rgba(56,189,248,0.2)"),
            module_card("04", "🔮", "AI PREDICTION", m4_s, "#C084FC", m4_badge, m4_desc, "rgba(168,85,247,0.2)"),
            module_card("05", "🛡️", "RISK ANALYSIS", m5_s, "#FB923C", m5_badge, m5_desc, "rgba(249,115,22,0.2)"),
            module_card("06", "📊", "INDUSTRY BENCHMARK", m6_s, "#2DD4BF", m6_badge, m6_desc, "rgba(20,184,166,0.2)"),
        ])
        st.markdown(f"""<div style="background-color:#0F172A; border:1px solid #1E293B; border-radius:12px; padding:16px; min-height:900px;">
<div style="font-size:15px; font-weight:bold; color:#F1F5F9; letter-spacing:0.5px; margin-bottom:12px;">INVESTMENT DECISION OVERVIEW ({selected_ticker})</div>
<div style="display:grid; grid-template-columns: repeat(2, 1fr); gap:12px;">{cards_html}</div>
</div>""", unsafe_allow_html=True)

    with col_right:
        overall = safe(stock_info.get('overall_score'), 50)
        rec = stock_info.get('recommendation', 'ACCUMULATE')
        rec_color = {"STRONG BUY": "#10B981", "BUY": "#10B981", "ACCUMULATE": "#84CC16", "REDUCE / SELL": "#EF4444"}.get(rec, "#F59E0B")
        stars = min(5, max(1, round(overall / 20)))
        arc_frac = min(1.0, overall / 100)
        dash_len = round(119.38 * arc_frac, 2)
        label = "ATTRACTIVE" if overall >= 65 else ("FAIR" if overall >= 45 else "CAUTION")
        top_strength = "financial health" if m1_s == max(m1_s, m2_s, m3_s, m4_s, m5_s, m6_s) else "fair value"
        st.markdown(f"""<div style="background-color:#0F172A; border:1px solid #1E293B; border-radius:12px; padding:16px; min-height:900px; display:flex; flex-direction:column; justify-content:space-between; text-align:center;">
<div>
<div style="font-size:14.5px; font-weight:bold; color:#94A3B8; letter-spacing:0.5px; text-align:left; margin-bottom:4px;">AI INVESTMENT SUMMARY</div>
<div style="margin:6px auto 0 auto; width:140px;">
<svg viewBox="0 0 100 58" style="width:130px; height:83px; display:block; margin:0 auto;">
<path d="M 12 50 A 38 38 0 0 1 88 50" fill="none" stroke="#1E293B" stroke-width="10" stroke-linecap="round" />
<path d="M 12 50 A 38 38 0 0 1 88 50" fill="none" stroke="#10B981" stroke-width="10" stroke-linecap="round" stroke-dasharray="{dash_len} 119.38" />
<text x="50" y="38" text-anchor="middle" font-size="21" font-weight="bold" fill="#FFFFFF">{overall:.0f}</text>
<text x="50" y="49" text-anchor="middle" font-size="11.5" fill="#64748B">100</text>
</svg>
</div>
<div style="font-size:13px; font-weight:bold; color:#94A3B8; margin-top:2px;">OVERALL SCORE</div>
<div style="color:#F59E0B; font-size:14.5px; letter-spacing:2px; margin:2px 0;">{'★'*stars}{'☆'*(5-stars)}</div>
<div style="color:{rec_color}; font-size:16px; font-weight:bold; margin-top:2px;">{label}</div>
<div style="font-size:12.5px; color:#CBD5E1; line-height:1.35; margin-top:4px; padding:0 2px;">
<b>{selected_ticker}</b> ได้คะแนนภาพรวม {overall:.1f}/100 จุดเด่นหลักอยู่ที่ {top_strength} อันดับ {int(stock_info.get('sector_rank',1))} จาก {n_sector} บริษัทในกลุ่ม {stock_info.get('sector','-')}
</div>
<div style="background-color:#151E2F; border:1px solid #1E293B; border-radius:8px; padding:8px 10px; margin-top:8px; text-align:left;">
<div style="font-size:12.5px; color:#94A3B8; font-weight:bold; margin-bottom:2px;">RECOMMENDATION</div>
<div style="display:flex; justify-content:space-between; align-items:center;">
<div style="display:flex; align-items:center; gap:6px;">
<span style="color:{rec_color}; font-size:16.5px;">📈</span>
<div><b style="color:{rec_color}; font-size:16px; line-height:1;">{rec}</b><div style="color:#64748B; font-size:11.5px;">Based on Overall Score</div></div>
</div>
<div style="text-align:right;"><div style="color:#64748B; font-size:11.5px;">Sector Rank:</div><b style="color:{rec_color}; font-size:13px;">{int(stock_info.get('sector_rank',1))} / {n_sector}</b></div>
</div></div>
</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:22px;'></div>", unsafe_allow_html=True)

    rev_g = stock_info.get('revenue_growth_yoy')
    ni_g = stock_info.get('net_income_growth_yoy')
    fcf_val = stock_info.get('free_cash_flow_latest')
    de_val = safe(stock_info.get('de_ratio'))
    roe_val = safe(stock_info.get('roe'))
    industry_rank_txt = f"{int(stock_info.get('sector_rank',1))} / {n_sector}"

    def hl_card(icon, bg, label, value, sub, val_color="#F8FAFC"):
        return f"""<div style="background-color:#151E2F; border:1px solid #1E293B; border-radius:8px; padding:10px 12px; display:flex; align-items:center; gap:10px;">
<div style="background:{bg}; width:40px; height:40px; border-radius:8px; display:flex; align-items:center; justify-content:center; font-size:17.5px;">{icon}</div>
<div><div style="font-size:13px; color:#94A3B8;">{label}</div><div style="font-size:16px; font-weight:bold; color:{val_color}; margin-top:1px;">{value}</div><div style="font-size:12px; color:#64748B;">{sub}</div></div>
</div>"""

    hl_html = "".join([
        hl_card("📊", "rgba(16,185,129,0.15)", "Revenue Growth", f"{'+' if (rev_g or 0)>=0 else ''}{rev_g if rev_g is not None else 0:.1f}%", "YoY (latest FY)", "#10B981" if (rev_g or 0) >= 0 else "#EF4444"),
        hl_card("💰", "rgba(245,158,11,0.15)", "Net Profit Growth", f"{'+' if (ni_g or 0)>=0 else ''}{ni_g if ni_g is not None else 0:.1f}%", "YoY (latest FY)", "#10B981" if (ni_g or 0) >= 0 else "#EF4444"),
        hl_card("⏱️", "rgba(56,189,248,0.15)", "ROE (TTM)", f"{roe_val:.1f}%", "Return on Equity", "#38BDF8"),
        hl_card("💵", "rgba(168,85,247,0.15)", "Free Cash Flow", fmt_mb(safe(fcf_val)), "Latest FY", "#F8FAFC"),
        hl_card("🛡️", "rgba(249,115,22,0.15)", "Debt to Equity", f"{de_val:.2f}", "Lower is safer", "#FB923C"),
        hl_card("🏆", "rgba(20,184,166,0.15)", "Sector Rank", industry_rank_txt, f"In {stock_info.get('sector','-')}", "#2DD4BF"),
    ])
    st.markdown(f"""<div style="background-color:#0F172A; border:1px solid #1E293B; border-radius:12px; padding:14px 16px;">
<div style="font-size:14.5px; font-weight:bold; color:#94A3B8; margin-bottom:10px; letter-spacing:0.5px;">KEY HIGHLIGHTS</div>
<div style="display:grid; grid-template-columns: repeat(6, 1fr); gap:10px;">{hl_html}</div>
</div>
<div style="font-size:12.5px; color:#475569; text-align:center; margin-top:10px;">
Disclaimer: This dashboard is for informational purposes only and not intended as investment advice. Please conduct your own research before making investment decisions.
</div>""", unsafe_allow_html=True)

# ============================================================================
# PAGE 2: MODULE 1 - COMPANY HEALTH
# ============================================================================
elif nav_page == " 💚 Company Health":

    def get_fin_val(target_yr, col_name, default="-", fmt="{:.1f}"):
        match = fin_stock[fin_stock['year'] == target_yr]
        if not match.empty:
            v = match.iloc[0].get(col_name)
            if v is not None and str(v).strip() not in ['', '-', 'nan', 'None']:
                try:
                    return fmt.format(float(v))
                except Exception:
                    return str(v)
        return str(default)

    roe_23, roe_24, roe_25 = get_fin_val(2023, 'roe'), get_fin_val(2024, 'roe'), get_fin_val(2025, 'roe')
    roa_23, roa_24, roa_25 = get_fin_val(2023, 'roa'), get_fin_val(2024, 'roa'), get_fin_val(2025, 'roa')
    npm_23, npm_24, npm_25 = get_fin_val(2023, 'net_margin'), get_fin_val(2024, 'net_margin'), get_fin_val(2025, 'net_margin')
    de_23, de_24, de_25 = get_fin_val(2023, 'de_ratio', fmt="{:.2f}"), get_fin_val(2024, 'de_ratio', fmt="{:.2f}"), get_fin_val(2025, 'de_ratio', fmt="{:.2f}")
    cr_23, cr_24, cr_25 = get_fin_val(2023, 'current_ratio', fmt="{:.2f}"), get_fin_val(2024, 'current_ratio', fmt="{:.2f}"), get_fin_val(2025, 'current_ratio', fmt="{:.2f}")

    st.markdown(f"""<div style="display:flex; justify-content:space-between; align-items:flex-end; margin-bottom:15px;">
<div><div style="display:flex; align-items:center; gap:8px;"><h2 style="margin:0; font-size:23px; font-weight:bold; color:#F8FAFC; letter-spacing:0.5px;">COMPANY HEALTH</h2></div>
<div style="font-size:15px; color:#94A3B8; margin-top:2px;">ประเมินสุขภาพทางการเงินของบริษัทจากมิติสำคัญตามงบการเงินจริง</div></div>
<div style="text-align:right;"><span style="font-size:13px; color:#64748B;">คำนวณจากงบ</span><br><b style="color:#38BDF8; font-size:15px;">{stock_info.get('financials_period_label','-')}</b></div>
</div>""", unsafe_allow_html=True)

    r1_c1, r1_c2, r1_c3 = st.columns([1.1, 1.4, 1.5])

    h_score = int(round(safe(stock_info.get('health_score'), 75)))
    h_badge = "EXCELLENT" if h_score >= 75 else ("MODERATE" if h_score >= 50 else "WEAK")
    h_color = "#10B981" if h_score >= 75 else ("#F59E0B" if h_score >= 50 else "#EF4444")
    h_stars = min(5, max(1, round(h_score / 20)))

    with r1_c1:
        st.markdown(f"""<div style="background-color:#0F172A; border:1px solid #1E293B; border-radius:12px; padding:16px; min-height:235px; display:flex; flex-direction:column; justify-content:space-between;">
<div style="font-size:14.5px; font-weight:bold; color:#94A3B8; letter-spacing:0.5px;">COMPANY HEALTH SCORE</div>
<div style="display:flex; align-items:center; gap:16px; margin:auto 0;">
<div style="width:92px; height:92px; border-radius:50%; background:conic-gradient({h_color} 0% {h_score}%, #1E293B {h_score}% 100%); display:flex; align-items:center; justify-content:center; flex-shrink:0;">
<div style="width:76px; height:76px; border-radius:50%; background-color:#0F172A; display:flex; flex-direction:column; align-items:center; justify-content:center;">
<span style="font-size:23px; font-weight:bold; color:#FFFFFF; line-height:1;">{h_score}</span><span style="font-size:13px; color:#64748B;">/100</span></div></div>
<div><div style="color:{h_color}; font-size:18.5px; font-weight:bold; line-height:1.2;">{h_badge}</div>
<div style="font-size:14.5px; color:#CBD5E1; line-height:1.4; margin-top:4px;">ประเมินจากอัตราส่วนทางการเงินจริงปี 2023-2025</div>
<div style="color:{h_color}; font-size:15px; letter-spacing:2px; margin-top:6px;">{'★'*h_stars}{'☆'*(5-h_stars)}</div></div>
</div></div>""", unsafe_allow_html=True)

    with r1_c2:
        st.markdown(f"""<div style="background-color:#0F172A; border:1px solid #1E293B; border-radius:12px; padding:16px; min-height:235px; display:flex; flex-direction:column; justify-content:space-between;">
<div><div style="font-size:14.5px; font-weight:bold; color:#94A3B8; letter-spacing:0.5px; margin-bottom:8px;">EXPLAINABLE FINANCIAL SUMMARY ({selected_ticker})</div>
<p style="font-size:15px; color:#CBD5E1; line-height:1.6; margin:0;">
ผลการวิเคราะห์สุขภาพการเงินของ <b>{selected_ticker}</b> พบว่ามีอัตราส่วนผลตอบแทนต่อส่วนของผู้ถือหุ้น (ROE) ล่าสุดอยู่ที่ {roe_25}% และความสามารถในการทำกำไรสุทธิ (Net Margin) อยู่ที่ {npm_25}% ในขณะที่ภาระหนี้สินต่อทุน (D/E Ratio) อยู่ที่ {de_25} เท่า และสภาพคล่องหมุนเวียน (Current Ratio) อยู่ที่ {cr_25} เท่า
</p></div>
<div><span style="display:inline-flex; align-items:center; gap:6px; background-color:#151E2F; border:1px solid #1E293B; color:#38BDF8; font-size:14px; padding:5px 12px; border-radius:6px;">
Financial Health Benchmark: {stock_info.get('sector','-')}</span></div>
</div>""", unsafe_allow_html=True)

    with r1_c3:
        st.markdown("""<div style="background-color:#0F172A; border:1px solid #1E293B; border-radius:12px 12px 0 0; padding:12px 16px 0 16px;">
<div style="font-size:14.5px; font-weight:bold; color:#94A3B8; letter-spacing:0.5px;">COMPANY HEALTH SCORE TREND (Actual, 2023-2025)</div></div>""", unsafe_allow_html=True)

        hy = health_yearly_df[health_yearly_df['ticker'] == selected_ticker].sort_values('year') if not health_yearly_df.empty else pd.DataFrame()
        trend_x = hy['year'].astype(str).tolist() if not hy.empty else ['2023', '2024', '2025']
        trend_y = hy['health_score'].tolist() if not hy.empty else [h_score, h_score, h_score]

        fig_health_trend = go.Figure()
        fig_health_trend.add_trace(go.Scatter(
            x=trend_x, y=trend_y, mode='lines+markers+text', text=trend_y, textposition='top center',
            textfont=dict(size=12.5, color='#F8FAFC'), line=dict(color='#10B981', width=2),
            marker=dict(size=10, color='#10B981', line=dict(width=1.5, color='#FFFFFF'))
        ))
        fig_health_trend.update_layout(
            height=168, margin=dict(l=25, r=15, t=10, b=20), paper_bgcolor="#0F172A", plot_bgcolor="#0F172A",
            yaxis=dict(range=[0, 110], tickvals=[0, 25, 50, 75, 100], tickfont=dict(size=11.5, color="#64748B"), gridcolor="#1E293B", zeroline=False),
            xaxis=dict(tickfont=dict(size=12, color="#94A3B8"), gridcolor="#1E293B"), showlegend=False
        )
        show_chart(fig_health_trend, key="health_trend", expand_height=650)

    st.markdown("<div style='margin-top:22px;'></div>", unsafe_allow_html=True)
    st.markdown("""<div style="font-size:15px; font-weight:bold; color:#F8FAFC; letter-spacing:0.5px; margin-bottom:8px;">
7 DIMENSIONS OVERVIEW <span style="font-size:14.5px; color:#94A3B8; font-weight:normal; margin-left:6px;">ผลการประเมินสุขภาพทางการเงินในแต่ละมิติ (คำนวณจากอัตราส่วนจริง)</span></div>""", unsafe_allow_html=True)

    latest_fin_row = fin_stock.iloc[-1] if not fin_stock.empty else pd.Series(dtype=float)
    ocf_ni = safe(latest_fin_row.get('ocf_to_ni'), 1.0)
    int_cov = safe(latest_fin_row.get('interest_coverage'), 5.0)
    rev_growth = safe(stock_info.get('revenue_growth_yoy'), 0.0)

    dim_profit = int(round(safe(stock_info.get('s_profitability'), 50)))
    dim_growth = int(round(np.clip(50 + rev_growth * 2, 0, 100)))
    dim_stability = int(round(safe(stock_info.get('s_debt'), 50)))
    dim_liquidity = int(round(safe(stock_info.get('s_liquidity'), 50)))
    dim_cashflow = int(round(np.clip(50 + ocf_ni * 5, 0, 100)))
    dim_efficiency = int(round(np.clip(safe(stock_info.get('roa'), 5) * 7, 0, 100)))
    dim_earnings = int(round(np.clip(50 + int_cov * 0.3, 0, 100)))

    def label_for(score):
        if score >= 75: return "EXCELLENT"
        if score >= 55: return "GOOD"
        if score >= 35: return "MODERATE"
        return "WEAK"

    dims = [
        ("1", "📊", "PROFITABILITY", "30%", dim_profit, "#10B981"),
        ("2", "📈", "GROWTH", "15%", dim_growth, "#3B82F6"),
        ("3", "🛡️", "FIN. STABILITY", "20%", dim_stability, "#EAB308"),
        ("4", "💧", "LIQUIDITY", "10%", dim_liquidity, "#06B6D4"),
        ("5", "💵", "CASH FLOW", "10%", dim_cashflow, "#8B5CF6"),
        ("6", "⚙️", "EFFICIENCY", "10%", dim_efficiency, "#F97316"),
        ("7", "🎖️", "EARNINGS Q.", "5%", dim_earnings, "#10B981"),
    ]
    dim_html = "".join([f"""<div style="background-color:#0F172A; border:1px solid #1E293B; border-radius:10px; padding:10px 8px; text-align:center;">
<div style="display:flex; align-items:center; justify-content:center; gap:4px;"><span style="font-size:14.5px;">{icon}</span><span style="font-size:13px; font-weight:bold; color:#CBD5E1;">{n}. {label}</span></div>
<div style="font-size:12.5px; color:#64748B; margin-top:1px;">Weight {w}</div>
<div style="margin:8px auto; width:60px; height:60px; border-radius:50%; background:conic-gradient({color} 0% {score}%, #1E293B {score}% 100%); display:flex; align-items:center; justify-content:center;">
<div style="width:48px; height:48px; border-radius:50%; background-color:#0F172A; display:flex; flex-direction:column; align-items:center; justify-content:center;">
<span style="font-size:16px; font-weight:bold; color:#FFFFFF; line-height:1;">{score}</span><span style="font-size:11.5px; color:#64748B;">/100</span></div></div>
<div style="color:{color}; font-size:13.5px; font-weight:bold;">{label_for(score)}</div>
</div>""" for n, icon, label, w, score, color in dims])
    st.markdown(f"""<div style="display:grid; grid-template-columns: repeat(7, 1fr); gap:8px;">{dim_html}</div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:22px;'></div>", unsafe_allow_html=True)
    r3_c1, r3_c2, r3_c3 = st.columns([1.5, 1.25, 1.25])

    with r3_c1:
        st.markdown(f"""<div style="background-color:#0F172A; border:1px solid #1E293B; border-radius:12px; padding:14px; min-height:360px; display:flex; flex-direction:column; justify-content:space-between;">
<div><div style="font-size:14.5px; font-weight:bold; color:#94A3B8; letter-spacing:0.5px; margin-bottom:8px;">KEY FINANCIAL HIGHLIGHTS ({selected_ticker})</div>
<table style="width:100%; text-align:left; font-size:14px; color:#CBD5E1; border-collapse:collapse;">
<tr style="border-bottom:1px solid #1E293B; color:#64748B; font-size:13px;"><th style="padding:4px 0;">Metric</th><th>2023</th><th>2024</th><th>2025</th></tr>
<tr style="border-bottom:1px solid #1E293B;"><td style="padding:5px 0; font-weight:bold; color:#F8FAFC;">ROE (%)</td><td>{roe_23}</td><td>{roe_24}</td><td style="font-weight:bold; color:#F8FAFC;">{roe_25}</td></tr>
<tr style="border-bottom:1px solid #1E293B;"><td style="padding:5px 0; font-weight:bold; color:#F8FAFC;">ROA (%)</td><td>{roa_23}</td><td>{roa_24}</td><td style="font-weight:bold; color:#F8FAFC;">{roa_25}</td></tr>
<tr style="border-bottom:1px solid #1E293B;"><td style="padding:5px 0; font-weight:bold; color:#F8FAFC;">Net Profit Margin (%)</td><td>{npm_23}</td><td>{npm_24}</td><td style="font-weight:bold; color:#F8FAFC;">{npm_25}</td></tr>
<tr style="border-bottom:1px solid #1E293B;"><td style="padding:5px 0; font-weight:bold; color:#F8FAFC;">Debt to Equity (x)</td><td>{de_23}</td><td>{de_24}</td><td style="font-weight:bold; color:#F8FAFC;">{de_25}</td></tr>
<tr><td style="padding:5px 0; font-weight:bold; color:#F8FAFC;">Current Ratio (x)</td><td>{cr_23}</td><td>{cr_24}</td><td style="font-weight:bold; color:#F8FAFC;">{cr_25}</td></tr>
</table></div>
<div style="font-size:12px; color:#64748B; margin-top:6px;">* ข้อมูลทางการเงินดึงตรงจาก stock_financials.csv สำหรับปี 2023-2025 จริงทุกค่า</div>
</div>""", unsafe_allow_html=True)

    with r3_c2:
        strengths, watch = [], []
        if safe(roe_25 if roe_25 != '-' else 0) > 15: strengths.append(f"ROE ล่าสุดอยู่ในเกณฑ์ดีที่ {roe_25}%")
        if safe(cr_25 if cr_25 != '-' else 0) >= 1.0: strengths.append(f"สภาพคล่อง Current Ratio อยู่ที่ {cr_25} เท่า เพียงพอต่อภาระหนี้ระยะสั้น")
        if safe(npm_25 if npm_25 != '-' else 0) > 10: strengths.append(f"Net Margin ระดับ {npm_25}% สะท้อนความสามารถทำกำไรที่ดี")
        if safe(de_25 if de_25 != '-' else 0) < 1.5: strengths.append(f"โครงสร้างเงินทุนมี D/E เพียง {de_25} เท่า ความเสี่ยงหนี้สินต่ำ")
        if not strengths: strengths.append("ผลประกอบการโดยรวมยังอยู่ระหว่างการฟื้นตัว")
        if safe(de_25 if de_25 != '-' else 0) > 1.5: watch.append(f"ภาระหนี้สินต่อทุนค่อนข้างสูงที่ {de_25} เท่า ควรติดตามใกล้ชิด")
        if safe(cr_25 if cr_25 != '-' else 0) < 1.0: watch.append(f"Current Ratio ต่ำกว่า 1 เท่า ({cr_25}) สภาพคล่องระยะสั้นควรเฝ้าระวัง")
        if rev_growth < 0: watch.append(f"รายได้หดตัว {rev_growth:.1f}% YoY ควรติดตามแนวโน้มปีถัดไป")
        if not watch: watch.append("ยังไม่พบสัญญาณความเสี่ยงเชิงโครงสร้างที่ชัดเจนในงบล่าสุด")

        st.markdown(f"""<div style="background-color:#0F172A; border:1px solid #1E293B; border-radius:12px; padding:14px; height:360px; overflow-y:auto;">
<div style="font-size:14px; font-weight:bold; color:#10B981; margin-bottom:6px;">STRENGTHS ({selected_ticker})</div>
<div style="font-size:13px; color:#CBD5E1; line-height:1.45; margin-bottom:10px;">
{''.join([f'<div style="display:flex; gap:6px; margin-bottom:3px;"><span style="color:#10B981;">✔</span><span>{s}</span></div>' for s in strengths])}
</div>
<div style="font-size:14px; font-weight:bold; color:#F59E0B; margin-bottom:6px; border-top:1px dashed #1E293B; padding-top:8px;">WATCH OUT</div>
<div style="font-size:13px; color:#CBD5E1; line-height:1.45;">
{''.join([f'<div style="display:flex; gap:6px; margin-bottom:3px;"><span style="color:#F59E0B;">⚠️</span><span>{w}</span></div>' for w in watch])}
</div></div>""", unsafe_allow_html=True)

    with r3_c3:
        # ค่าเฉลี่ยอุตสาหกรรม (sector) จากข้อมูลจริงของปีล่าสุดที่มี ในกลุ่มเดียวกัน
        sector_fin = fin_df[(fin_df['ticker'].isin(sector_peers['ticker'])) & (fin_df['year'] == fin_stock['year'].max())]
        ind_roe = sector_fin['roe'].mean() if not sector_fin.empty else safe(roe_25 if roe_25 != '-' else 0)
        ind_roa = sector_fin['roa'].mean() if not sector_fin.empty else safe(roa_25 if roa_25 != '-' else 0)
        ind_npm = sector_fin['net_margin'].mean() if not sector_fin.empty else safe(npm_25 if npm_25 != '-' else 0)
        ind_de = sector_fin['de_ratio'].mean() if not sector_fin.empty else safe(de_25 if de_25 != '-' else 0)
        ind_cr = sector_fin['current_ratio'].mean() if not sector_fin.empty else safe(cr_25 if cr_25 != '-' else 0)

        def pct_bar(stock_val, ind_val, higher_better=True):
            if ind_val == 0: return 50
            ratio = (stock_val / ind_val) if higher_better else (ind_val / max(stock_val, 0.01))
            return int(np.clip(ratio * 50, 5, 100))

        rows_cmp = [
            ("ROE (%)", roe_25, f"{ind_roe:.1f}", pct_bar(safe(roe_25 if roe_25 != '-' else 0), ind_roe)),
            ("ROA (%)", roa_25, f"{ind_roa:.1f}", pct_bar(safe(roa_25 if roa_25 != '-' else 0), ind_roa)),
            ("Net Margin (%)", npm_25, f"{ind_npm:.1f}", pct_bar(safe(npm_25 if npm_25 != '-' else 0), ind_npm)),
            ("Debt to Equity (x)", de_25, f"{ind_de:.2f}", pct_bar(safe(de_25 if de_25 != '-' else 0), ind_de, higher_better=False)),
            ("Current Ratio (x)", cr_25, f"{ind_cr:.2f}", pct_bar(safe(cr_25 if cr_25 != '-' else 0), ind_cr)),
        ]
        rows_html = "".join([f"""<tr style="border-bottom:1px solid #1E293B;">
<td style="padding:4px 0;">{name}</td><td style="font-weight:bold; color:#F8FAFC;">{v}</td><td style="color:#64748B;">{avg}</td>
<td><div style="display:flex; align-items:center; gap:6px;"><div style="background:#1E293B; width:60px; height:9px; border-radius:4px; overflow:hidden;"><div style="background:#10B981; width:{pct}%; height:100%;"></div></div><span style="font-size:12px; color:#10B981; font-weight:bold;">{pct}%</span></div></td>
</tr>""" for name, v, avg, pct in rows_cmp])

        st.markdown(f"""<div style="background-color:#0F172A; border:1px solid #1E293B; border-radius:12px; padding:14px; height:360px;">
<div style="font-size:14.5px; font-weight:bold; color:#94A3B8; letter-spacing:0.5px;">INDUSTRY COMPARISON</div>
<div style="font-size:12.5px; color:#64748B; margin-bottom:8px;">เทียบกับค่าเฉลี่ยจริงของกลุ่ม ({stock_info.get('sector','-')}, ปีล่าสุด)</div>
<table style="width:100%; text-align:left; font-size:13.5px; color:#CBD5E1; border-collapse:collapse;">
<tr style="border-bottom:1px solid #1E293B; color:#64748B; font-size:12.5px;"><th style="padding:3px 0;">Metric</th><th>{selected_ticker}</th><th>Sector Avg</th><th>vs Avg</th></tr>
{rows_html}
</table></div>""", unsafe_allow_html=True)

    render_nav_footer("m1", prev_page=" 🏠 Overview", next_page=" ⚖️ Fair Value")
# ============================================================================
# PAGE 3: MODULE 2 - FAIR VALUE ASSESSMENT
# ============================================================================
elif nav_page == " ⚖️ Fair Value":
    val_cur_price = current_price
    val_fair_value = safe(stock_info.get('fair_value'), val_cur_price * 1.1)
    val_mos = safe(stock_info.get('margin_of_safety'), 10.0)
    val_score = int(round(safe(stock_info.get('valuation_score'), 75)))
    val_status = "UNDERVALUED" if val_mos > 10 else ("OVERVALUED" if val_mos < -10 else "FAIR VALUE")
    val_color = "#10B981" if val_mos > 10 else ("#EF4444" if val_mos < -10 else "#F59E0B")
    val_rec_label = "ATTRACTIVE" if val_mos > 10 else ("FAIR" if val_mos >= -5 else "CAUTION")

    val_bear = safe(stock_info.get('dcf_fair_value'), val_fair_value * 0.9)
    val_base = val_fair_value
    val_bull = safe(stock_info.get('pe_fair_value'), val_fair_value * 1.1)
    # เรียงให้ bear <= base <= bull เสมอเพื่อความสวยงามของภาพ
    lo, hi = min(val_bear, val_bull), max(val_bear, val_bull)
    val_bear, val_bull = lo, hi

    st.markdown(f"""<div style="display:flex; justify-content:space-between; align-items:flex-end; margin-bottom:15px;">
<div><div style="display:flex; align-items:center; gap:8px;"><h2 style="margin:0; font-size:23px; font-weight:bold; color:#F8FAFC; letter-spacing:0.5px;">FAIR VALUE ASSESSMENT</h2></div>
<div style="font-size:15px; color:#94A3B8; margin-top:2px;">ประเมินมูลค่าที่เหมาะสมของหุ้นโดยใช้แบบจำลอง DCF ผสาน P/E Relative</div></div>
<div style="text-align:right;"><span style="font-size:13px; color:#64748B;">คำนวณจากงบ</span><br><b style="color:#38BDF8; font-size:15px;">{stock_info.get('financials_period_label','-')}</b></div>
</div>""", unsafe_allow_html=True)

    r1_c1, r1_c2, r1_c3, r1_c4, r1_c5, r1_c6 = st.columns([1.5, 0.9, 0.9, 0.9, 0.9, 1.1])
    val_stars = min(5, max(1, round(val_score / 20)))

    with r1_c1:
        st.markdown(f"""<div style="background-color:#0F172A; border:1px solid #1E293B; border-radius:12px; padding:14px; min-height:170px; display:flex; flex-direction:column; justify-content:space-between;">
<div style="font-size:13.5px; font-weight:bold; color:#94A3B8; letter-spacing:0.5px;">FAIR VALUE SUMMARY</div>
<div style="display:flex; align-items:center; gap:12px;">
<div style="width:76px; height:76px; border-radius:50%; background:conic-gradient({val_color} 0% {val_score}%, #1E293B {val_score}% 100%); display:flex; align-items:center; justify-content:center; flex-shrink:0;">
<div style="width:62px; height:62px; border-radius:50%; background-color:#0F172A; display:flex; flex-direction:column; align-items:center; justify-content:center;">
<span style="font-size:19px; font-weight:bold; color:#FFFFFF; line-height:1;">{val_score}</span><span style="font-size:12px; color:#64748B;">/100</span></div></div>
<div><div style="color:{val_color}; font-size:16.5px; font-weight:bold; line-height:1.2;">{val_status}</div>
<div style="font-size:13px; color:#CBD5E1; line-height:1.35; margin-top:3px;">Margin of Safety อยู่ที่ {val_mos:.1f}% เมื่อเทียบกับมูลค่าพื้นฐานที่แท้จริง</div>
<div style="color:{val_color}; font-size:14.5px; letter-spacing:1px; margin-top:4px;">{'★'*val_stars}{'☆'*(5-val_stars)}</div></div>
</div></div>""", unsafe_allow_html=True)

    with r1_c2:
        st.markdown(f"""<div style="background-color:#0F172A; border:1px solid #1E293B; border-radius:12px; padding:12px; min-height:170px; display:flex; flex-direction:column; justify-content:space-between;">
<div style="font-size:13px; font-weight:bold; color:#94A3B8;">CURRENT PRICE</div>
<div><div style="font-size:21px; font-weight:bold; color:#FFFFFF; line-height:1;">{val_cur_price:.2f} <span style="font-size:13.5px; color:#94A3B8;">THB</span></div>
<div style="font-size:12px; color:#64748B; margin-top:2px;">({stock_info.get('latest_date','-')})</div></div>
</div>""", unsafe_allow_html=True)

    with r1_c3:
        st.markdown(f"""<div style="background-color:#0F172A; border:1px solid #1E293B; border-radius:12px; padding:12px; min-height:170px; display:flex; flex-direction:column; justify-content:space-between;">
<div style="font-size:13px; font-weight:bold; color:#94A3B8;">ESTIMATED FAIR VALUE<br><span style="font-size:12px; color:#64748B;">(BLENDED: 55% DCF + 45% P/E)</span></div>
<div><div style="font-size:21px; font-weight:bold; color:#FFFFFF; line-height:1;">{val_base:.2f} <span style="font-size:13.5px; color:#94A3B8;">THB</span></div></div>
</div>""", unsafe_allow_html=True)

    with r1_c4:
        st.markdown(f"""<div style="background-color:#0F172A; border:1px solid #1E293B; border-radius:12px; padding:12px; min-height:170px; display:flex; flex-direction:column; justify-content:space-between; text-align:center;">
<div style="font-size:13px; font-weight:bold; color:#94A3B8; text-align:left;">MARGIN OF SAFETY</div>
<div><div style="font-size:21px; font-weight:bold; color:{val_color}; line-height:1;">{val_mos:.1f}%</div></div>
<div style="margin-top:auto; display:flex; justify-content:center;"><div style="background:rgba(16,185,129,0.15); border:1px solid #10B981; border-radius:50%; width:36px; height:36px; display:flex; align-items:center; justify-content:center; font-size:15px; color:#10B981;">🛡️</div></div>
</div>""", unsafe_allow_html=True)

    with r1_c5:
        conf = "High" if abs(val_mos) > 15 else ("Medium" if abs(val_mos) > 5 else "Low")
        st.markdown(f"""<div style="background-color:#0F172A; border:1px solid #1E293B; border-radius:12px; padding:12px; min-height:170px; display:flex; flex-direction:column; justify-content:space-between; text-align:center;">
<div style="font-size:13px; font-weight:bold; color:#94A3B8; text-align:left;">CONFIDENCE LEVEL</div>
<div><div style="font-size:18.5px; font-weight:bold; color:#10B981; line-height:1;">{conf.upper()}</div></div>
</div>""", unsafe_allow_html=True)

    with r1_c6:
        st.markdown(f"""<div style="background-color:#0F172A; border:1px solid #1E293B; border-radius:12px; padding:12px; min-height:170px; display:flex; flex-direction:column; justify-content:space-between; text-align:center;">
<div style="font-size:13px; font-weight:bold; color:#94A3B8; text-align:left;">RECOMMENDATION</div>
<div><div style="font-size:17.5px; font-weight:bold; color:{val_color}; line-height:1.1;">{val_rec_label}</div></div>
</div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:22px;'></div>", unsafe_allow_html=True)
    r2_c1, r2_c2, r2_c3 = st.columns([1.3, 1.3, 1.4])

    with r2_c1:
        st.markdown(f"""<div style="background-color:#0F172A; border:1px solid #1E293B; border-radius:12px; padding:14px; min-height:315px; display:flex; flex-direction:column; justify-content:space-between;">
<div style="font-size:14px; font-weight:bold; color:#94A3B8; letter-spacing:0.5px;">FAIR VALUE RANGE — DCF vs P/E RELATIVE</div>
<div style="display:grid; grid-template-columns: 1fr 1.1fr 1fr; gap:6px; margin-top:6px;">
<div style="background:#151E2F; border:1px solid #1E293B; border-radius:8px; padding:8px 4px; text-align:center;">
<div style="color:#38BDF8; font-size:13.5px; font-weight:bold;">Lower Estimate</div><div style="color:#64748B; font-size:12px;">Min(DCF, P/E)</div>
<div style="color:#F8FAFC; font-size:16px; font-weight:bold; margin-top:4px;">{val_bear:.2f} <span style="font-size:12px; color:#64748B;">THB</span></div></div>
<div style="background:#151E2F; border:1.5px solid #8B5CF6; border-radius:8px; padding:8px 4px; text-align:center;">
<div style="color:#C084FC; font-size:13.5px; font-weight:bold;">Blended Fair Value</div><div style="color:#94A3B8; font-size:12px;">55% DCF + 45% P/E</div>
<div style="color:#FFFFFF; font-size:16.5px; font-weight:bold; margin-top:4px;">{val_base:.2f} <span style="font-size:12px; color:#94A3B8;">THB</span></div></div>
<div style="background:#151E2F; border:1px solid #1E293B; border-radius:8px; padding:8px 4px; text-align:center;">
<div style="color:#10B981; font-size:13.5px; font-weight:bold;">Upper Estimate</div><div style="color:#64748B; font-size:12px;">Max(DCF, P/E)</div>
<div style="color:#F8FAFC; font-size:16px; font-weight:bold; margin-top:4px;">{val_bull:.2f} <span style="font-size:12px; color:#64748B;">THB</span></div></div>
</div>
<div style="background:rgba(16,185,129,0.08); border-radius:6px; padding:6px 8px; display:flex; align-items:flex-start; gap:6px; margin-top:10px;">
<span style="color:#10B981; font-size:14.5px;">✔</span><div style="font-size:12.5px; color:#CBD5E1; line-height:1.3;">
<b>DCF Fair Value: {safe(stock_info.get('dcf_fair_value')):.2f} THB &nbsp;|&nbsp; P/E Fair Value: {safe(stock_info.get('pe_fair_value')):.2f} THB</b><br>
<span style="color:#94A3B8;">คำนวณจากงบการเงินปีล่าสุด (FY{int(fin_stock['year'].max()) if not fin_stock.empty else '-'}) เทียบราคาตลาดปัจจุบัน {val_cur_price:.2f} THB</span></div></div>
</div>""", unsafe_allow_html=True)

    with r2_c2:
        st.markdown(f"""<div style="background-color:#0F172A; border:1px solid #1E293B; border-radius:12px; padding:14px; min-height:315px; display:flex; flex-direction:column; justify-content:space-between;">
<div style="font-size:14px; font-weight:bold; color:#94A3B8; letter-spacing:0.5px;">VALUATION DRIVERS ({selected_ticker})</div>
<div style="font-size:13px; color:#CBD5E1; line-height:1.45; display:flex; flex-direction:column; gap:6px; margin:auto 0;">
<div style="display:flex; gap:6px;"><span style="color:#10B981;">✔</span><div><b>Fair Value (Blended): {val_base:.2f} THB</b><br><span style="color:#64748B; font-size:12.5px;">ประเมินแบบผสมผสาน DCF + Relative P/E</span></div></div>
<div style="display:flex; gap:6px;"><span style="color:#10B981;">✔</span><div><b>Margin of Safety: {val_mos:.1f}%</b><br><span style="color:#64748B; font-size:12.5px;">ส่วนต่างความปลอดภัยจากราคาตลาดปัจจุบัน</span></div></div>
<div style="display:flex; gap:6px;"><span style="color:#10B981;">✔</span><div><b>P/E Ratio ปัจจุบัน: {stock_info.get('pe_ratio','-')} เท่า</b><br><span style="color:#64748B; font-size:12.5px;">เทียบ EPS ล่าสุด {stock_info.get('eps','-')} บาท/หุ้น</span></div></div>
<div style="display:flex; gap:6px;"><span style="color:#10B981;">✔</span><div><b>สถานะมูลค่า: {val_status}</b><br><span style="color:#64748B; font-size:12.5px;">ระดับความน่าดึงดูดเชิงมูลค่าพื้นฐาน</span></div></div>
</div></div>""", unsafe_allow_html=True)

    with r2_c3:
        safety_score = int(min(100, max(20, int(val_mos + 50))))
        st.markdown(f"""<div style="background-color:#0F172A; border:1px solid #1E293B; border-radius:12px; padding:14px; min-height:315px; display:flex; flex-direction:column; justify-content:space-between;">
<div style="font-size:14px; font-weight:bold; color:#94A3B8; letter-spacing:0.5px;">FAIR VALUE SCORE BY DIMENSION</div>
<div style="display:flex; flex-direction:column; gap:10px; margin:auto 0;">
<div><div style="display:flex; justify-content:space-between; font-size:13px; color:#CBD5E1; margin-bottom:3px;"><span>📊 Relative Valuation (P/E)</span><span style="font-weight:bold; color:#F8FAFC;">{val_score} <span style="font-size:12px; color:#64748B;">/100</span></span></div>
<div style="background:#1E293B; height:9px; border-radius:4px; overflow:hidden;"><div style="background:#10B981; width:{val_score}%; height:100%;"></div></div></div>
<div><div style="display:flex; justify-content:space-between; font-size:13px; color:#CBD5E1; margin-bottom:3px;"><span>🎯 Intrinsic Valuation (DCF)</span><span style="font-weight:bold; color:#F8FAFC;">{val_score} <span style="font-size:12px; color:#64748B;">/100</span></span></div>
<div style="background:#1E293B; height:9px; border-radius:4px; overflow:hidden;"><div style="background:#10B981; width:{val_score}%; height:100%;"></div></div></div>
<div><div style="display:flex; justify-content:space-between; font-size:13px; color:#CBD5E1; margin-bottom:3px;"><span>🛡️ Margin of Safety</span><span style="font-weight:bold; color:#F8FAFC;">{safety_score} <span style="font-size:12px; color:#64748B;">/100</span></span></div>
<div style="background:#1E293B; height:9px; border-radius:4px; overflow:hidden;"><div style="background:#10B981; width:{safety_score}%; height:100%;"></div></div></div>
</div>
<div style="display:flex; justify-content:space-between; align-items:center; border-top:1px solid #1E293B; padding-top:8px;">
<span style="font-size:13.5px; font-weight:bold; color:#CBD5E1;">OVERALL FAIR VALUE SCORE</span><span style="font-size:18.5px; font-weight:bold; color:{val_color};">{val_score} <span style="font-size:13px; color:#64748B;">/100</span></span>
</div></div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:22px;'></div>", unsafe_allow_html=True)
    st.markdown("""<div style="font-size:14.5px; font-weight:bold; color:#94A3B8; letter-spacing:0.5px; margin-bottom:8px;">DETAIL BREAKDOWN</div>""", unsafe_allow_html=True)
    d_c1, d_c2, d_c3, d_c4 = st.columns(4)

    with d_c1:
        st.markdown(f"""<div style="background-color:#0F172A; border:1px solid #1E293B; border-radius:10px; padding:12px; min-height:190px; display:flex; flex-direction:column; justify-content:space-between;">
<div><div style="font-size:13px; font-weight:bold; color:#CBD5E1;">RELATIVE VALUATION (P/E)</div>
<table style="width:100%; font-size:13px; color:#CBD5E1; border-collapse:collapse; margin-top:6px;">
<tr style="border-bottom:1px solid #1E293B; color:#64748B; font-size:12.5px;"><th style="text-align:left; padding:2px 0;">Metric</th><th>Value</th></tr>
<tr style="border-bottom:1px solid #1E293B;"><td style="padding:3px 0;">P/E Ratio ปัจจุบัน</td><td>{stock_info.get('pe_ratio','-')}x</td></tr>
<tr><td style="padding:3px 0;">P/E Fair Value</td><td>{safe(stock_info.get('pe_fair_value')):.2f} THB</td></tr>
</table></div></div>""", unsafe_allow_html=True)

    with d_c2:
        st.markdown(f"""<div style="background-color:#0F172A; border:1px solid #1E293B; border-radius:10px; padding:12px; min-height:190px; display:flex; flex-direction:column; justify-content:space-between;">
<div><div style="font-size:13px; font-weight:bold; color:#CBD5E1;">INTRINSIC VALUATION (DCF)</div>
<table style="width:100%; font-size:13px; color:#CBD5E1; border-collapse:collapse; margin-top:6px;">
<tr style="border-bottom:1px solid #1E293B; color:#64748B; font-size:12.5px;"><th style="text-align:left; padding:2px 0;">Metric</th><th>Value</th></tr>
<tr style="border-bottom:1px solid #1E293B;"><td style="padding:3px 0;">WACC</td><td>8.2%</td></tr>
<tr><td style="padding:3px 0;">DCF Fair Value</td><td>{safe(stock_info.get('dcf_fair_value')):.2f} THB</td></tr>
</table></div></div>""", unsafe_allow_html=True)

    with d_c3:
        st.markdown(f"""<div style="background-color:#0F172A; border:1px solid #1E293B; border-radius:10px; padding:12px; min-height:190px; display:flex; flex-direction:column; justify-content:space-between;">
<div><div style="font-size:13px; font-weight:bold; color:#CBD5E1;">PRICE COMPARISON</div>
<table style="width:100%; font-size:13px; color:#CBD5E1; border-collapse:collapse; margin-top:6px;">
<tr style="border-bottom:1px solid #1E293B;"><td style="padding:3px 0;">Current</td><td>{val_cur_price:.2f}</td></tr>
<tr style="border-bottom:1px solid #1E293B;"><td style="padding:3px 0;">Fair Value</td><td>{val_base:.2f}</td></tr>
<tr><td style="padding:3px 0;">Difference</td><td>{(val_base - val_cur_price):+.2f}</td></tr>
</table></div></div>""", unsafe_allow_html=True)

    with d_c4:
        st.markdown(f"""<div style="background-color:#0F172A; border:1px solid #1E293B; border-radius:10px; padding:12px; min-height:190px; display:flex; flex-direction:column; justify-content:space-between;">
<div><div style="font-size:13px; font-weight:bold; color:#CBD5E1;">MARGIN OF SAFETY</div>
<table style="width:100%; font-size:13px; color:#CBD5E1; border-collapse:collapse; margin-top:6px;">
<tr><td style="padding:3px 0;">Margin of Safety</td><td style="color:{val_color}; font-weight:bold;">{val_mos:.1f}%</td></tr>
</table></div></div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:22px;'></div>", unsafe_allow_html=True)
    r4_c1, r4_c2 = st.columns([1.3, 1.7])

    with r4_c1:
        st.markdown("""<div style="background-color:#0F172A; border:1px solid #1E293B; border-radius:12px; padding:14px; min-height:220px; display:flex; flex-direction:column; justify-content:space-between;">
<div style="font-size:13.5px; font-weight:bold; color:#94A3B8; letter-spacing:0.5px;">DCF ASSUMPTIONS</div>
<div style="display:grid; grid-template-columns: 1fr 1fr; gap:6px; font-size:13px; color:#CBD5E1; margin:auto 0;">
<div><span style="color:#64748B;">WACC</span><br><b style="color:#F8FAFC;">8.2%</b></div>
<div><span style="color:#64748B;">Terminal Growth</span><br><b style="color:#F8FAFC;">2.0%</b></div>
<div><span style="color:#64748B;">Forecast Period</span><br><b style="color:#F8FAFC;">1 Year FCF x1.05</b></div>
<div><span style="color:#64748B;">Target P/E</span><br><b style="color:#F8FAFC;">18-22x (by sector)</b></div>
<div><span style="color:#64748B;">DCF Weight</span><br><b style="color:#F8FAFC;">55%</b></div>
<div><span style="color:#64748B;">P/E Weight</span><br><b style="color:#F8FAFC;">45%</b></div>
</div></div>""", unsafe_allow_html=True)

    with r4_c2:
        st.markdown("""<div style="background-color:#0F172A; border:1px solid #1E293B; border-radius:12px 12px 0 0; padding:12px 16px 0 16px;">
<div style="font-size:13.5px; font-weight:bold; color:#94A3B8; letter-spacing:0.5px;">HISTORICAL FAIR VALUE VS PRICE (Actual, year-end 2023-2025)</div></div>""", unsafe_allow_html=True)

        fv_hist = fair_value_yearly_df[fair_value_yearly_df['ticker'] == selected_ticker].sort_values('year') if not fair_value_yearly_df.empty else pd.DataFrame()
        if not fv_hist.empty:
            fig_hist_val = go.Figure()
            fig_hist_val.add_trace(go.Scatter(x=fv_hist['year'].astype(str), y=fv_hist['fair_value'], mode='lines+markers', name='Fair Value', line=dict(color='#A855F7', width=1.8, dash='dash')))
            fig_hist_val.add_trace(go.Scatter(x=fv_hist['year'].astype(str), y=fv_hist['price'], mode='lines+markers', name='Actual Price', line=dict(color='#38BDF8', width=2), marker=dict(size=9, color='#38BDF8')))
            fig_hist_val.update_layout(
                height=150, margin=dict(l=25, r=15, t=10, b=20), paper_bgcolor="#0F172A", plot_bgcolor="#0F172A",
                yaxis=dict(tickfont=dict(size=11.5, color="#64748B"), gridcolor="#1E293B", zeroline=False),
                xaxis=dict(tickfont=dict(size=11, color="#94A3B8"), gridcolor="#1E293B"),
                legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5, font=dict(size=11.5, color="#94A3B8"))
            )
            show_chart(fig_hist_val, key="fair_value_hist", expand_height=650)
        else:
            st.info("ไม่มีข้อมูลย้อนหลังเพียงพอ")

    render_nav_footer("m2", prev_page=" 💚 Company Health", next_page=" ⏱️ Entry Timing")

# ============================================================================
# PAGE 4: MODULE 3 - ENTRY TIMING ANALYSIS
# ============================================================================
elif "Entry Timing" in nav_page:
    st.markdown("""<div style="display:flex; justify-content:space-between; align-items:flex-end; margin-bottom:15px;">
<div><div style="display:flex; align-items:center; gap:8px;"><h2 style="margin:0; font-size:23px; font-weight:bold; color:#F8FAFC; letter-spacing:0.5px;">ENTRY TIMING ANALYSIS</h2></div>
<div style="font-size:15px; color:#94A3B8; margin-top:2px;">วิเคราะห์จังหวะเข้าลงทุนด้วย Technical Indicators จริงจากราคาปิดรายวัน</div></div>
</div>""", unsafe_allow_html=True)

    timing_score = safe(stock_info.get('timing_score'), 50)
    trend_signal = stock_info.get('trend_signal', 'NEUTRAL')
    sig_color = "#10B981" if trend_signal == "BULLISH" else ("#EF4444" if trend_signal == "BEARISH" else "#F59E0B")
    sig_icon = "🐂" if trend_signal == "BULLISH" else ("🐻" if trend_signal == "BEARISH" else "⚖️")
    suggested_action = "Wait for Pullback" if trend_signal == "BULLISH" else ("Avoid / Wait for Reversal" if trend_signal == "BEARISH" else "Watch & Wait")

    r1_c1, r1_c2, r1_c3, r1_c4 = st.columns([1.1, 2.5, 0.9, 0.95])

    with r1_c1:
        st.markdown(f"""<div style="background-color:#0F172A; border:1px solid #1E293B; border-radius:12px; padding:16px; min-height:495px; display:flex; flex-direction:column; justify-content:space-between; text-align:center;">
<div style="font-size:14.5px; font-weight:bold; color:#94A3B8; letter-spacing:0.5px; text-align:left;">OVERALL ENTRY SIGNAL</div>
<div style="margin:auto 0;">
<div style="margin:0 auto; width:150px;"><svg viewBox="0 0 100 58" style="width:140px; height:89px; display:block; margin:0 auto;">
<path d="M 12 50 A 38 38 0 0 1 88 50" fill="none" stroke="#1E293B" stroke-width="10" stroke-linecap="round" />
<path d="M 12 50 A 38 38 0 0 1 88 50" fill="none" stroke="{sig_color}" stroke-width="10" stroke-linecap="round" stroke-dasharray="{round(119.38*min(1,timing_score/100),2)} 119.38" />
<text x="50" y="44" text-anchor="middle" font-size="25" fill="{sig_color}">{sig_icon}</text></svg></div>
<div style="font-size:19px; font-weight:bold; color:{sig_color}; margin-top:4px;">{trend_signal}</div>
<div style="font-size:14px; color:#CBD5E1;">Score {timing_score:.0f}/100</div></div>
<div style="background-color:#151E2F; border:1px solid #1E293B; border-radius:8px; padding:10px 12px; margin-bottom:10px;">
<div style="font-size:12.5px; color:{sig_color}; font-weight:bold;">SUGGESTED ACTION</div>
<div style="font-size:16.5px; font-weight:bold; color:#FFFFFF; margin-top:2px;">{suggested_action}</div></div>
<div><div style="display:flex; justify-content:space-between; font-size:13px; color:#94A3B8; margin-bottom:4px;"><span>SIGNAL STRENGTH</span><span style="font-weight:bold; color:#CBD5E1;">{'High' if timing_score>=65 or timing_score<35 else 'Medium'}</span></div>
<div style="display:flex; align-items:center; gap:8px;"><span style="font-size:15px; font-weight:bold; color:#FFFFFF;">{timing_score:.0f}<span style="font-size:12.5px; color:#64748B;">/100</span></span>
<div style="background:#1E293B; height:10px; flex-grow:1; border-radius:5px; overflow:hidden;"><div style="background:{sig_color}; width:{timing_score:.0f}%; height:100%;"></div></div></div></div>
</div>""", unsafe_allow_html=True)

    with r1_c2:
        st.markdown("""
        <style>
        section.main div[data-testid="stRadio"] > div { display: flex; justify-content: flex-end; gap: 4px; flex-wrap: nowrap; background: transparent; margin-bottom: 2px; }
        section.main div[data-testid="stRadio"] label { background-color: #151E2F !important; border: 1px solid #1E293B !important; border-radius: 6px !important; padding: 4px 10px !important; margin: 0 !important; cursor: pointer !important; }
        section.main div[data-testid="stRadio"] label > div:first-child { display: none !important; }
        section.main div[data-testid="stRadio"] label div[data-testid="stMarkdownContainer"] p { font-size: 10px !important; color: #94A3B8 !important; font-weight: 600 !important; margin: 0 !important; }
        section.main div[data-testid="stRadio"] label:has(input:checked) { background-color: #2563EB !important; border-color: #2563EB !important; }
        section.main div[data-testid="stRadio"] label:has(input:checked) div[data-testid="stMarkdownContainer"] p { color: #FFFFFF !important; font-weight: bold !important; }
        </style>
        """, unsafe_allow_html=True)

        head_c1, head_c2 = st.columns([1, 2.2])
        with head_c1:
            st.markdown("""<div style="font-size:14.5px; font-weight:bold; color:#94A3B8; letter-spacing:0.5px; padding-top:4px;">PRICE CHART (Actual OHLC)</div>""", unsafe_allow_html=True)
        with head_c2:
            tf_selected = st.radio("Timeframe", ["1M", "3M", "6M", "1Y", "2Y", "ALL"], index=2, horizontal=True, label_visibility="collapsed", key="timing_timeframe_selector")

        tf_bars = {"1M": 22, "3M": 66, "6M": 132, "1Y": 252, "2Y": 504, "ALL": len(stock_daily)}
        n_bars = min(tf_bars.get(tf_selected, 132), len(stock_daily))
        chart_df = stock_daily.tail(n_bars).copy()
        chart_df['SMA100'] = stock_daily['close'].rolling(100, min_periods=1).mean().tail(n_bars).values

        fig_main = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.78, 0.22])
        fig_main.add_trace(go.Candlestick(
            x=chart_df['date'], open=chart_df['open'], high=chart_df['high'], low=chart_df['low'], close=chart_df['close'],
            name='Price', increasing_line_color='#10B981', increasing_fillcolor='#10B981',
            decreasing_line_color='#EF4444', decreasing_fillcolor='#EF4444', whiskerwidth=0.7, line=dict(width=1.2)
        ), row=1, col=1)
        fig_main.add_trace(go.Scatter(x=chart_df['date'], y=chart_df['EMA20'], line=dict(color='#F59E0B', width=1.4), name='EMA 20'), row=1, col=1)
        fig_main.add_trace(go.Scatter(x=chart_df['date'], y=chart_df['EMA50'], line=dict(color='#38BDF8', width=1.4), name='EMA 50'), row=1, col=1)
        fig_main.add_trace(go.Scatter(x=chart_df['date'], y=chart_df['SMA100'], line=dict(color='#A855F7', width=1.4), name='SMA 100'), row=1, col=1)

        bar_colors = ['#10B981' if c >= o else '#EF4444' for c, o in zip(chart_df['close'], chart_df['open'])]
        vol_col = chart_df['volume'] if 'volume' in chart_df.columns else pd.Series([0] * len(chart_df))
        fig_main.add_trace(go.Bar(x=chart_df['date'], y=vol_col, marker_color=bar_colors, name='Volume', showlegend=False), row=2, col=1)

        fig_main.add_annotation(xref="paper", yref="y1", x=1.0, y=current_price, text=f"<b>{current_price:.2f}</b>", showarrow=True,
                                 arrowhead=0, arrowwidth=1.5, arrowcolor=change_color, ax=44, ay=0, font=dict(size=12.5, color="#FFFFFF"),
                                 bgcolor=change_color, bordercolor=change_color, borderwidth=1, borderpad=3)

        fig_main.update_layout(
            height=395, margin=dict(l=10, r=52, t=8, b=8), paper_bgcolor="#0F172A", plot_bgcolor="#0F172A",
            xaxis=dict(gridcolor="#1E293B", showticklabels=False, zeroline=False),
            xaxis2=dict(gridcolor="#1E293B", tickfont=dict(size=11.5, color="#64748B"), nticks=7, zeroline=False),
            yaxis=dict(gridcolor="#1E293B", tickfont=dict(size=11.5, color="#64748B"), side='right', zeroline=False),
            yaxis2=dict(gridcolor="#1E293B", showticklabels=False, zeroline=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0.01, font=dict(size=12, color="#CBD5E1")),
            xaxis_rangeslider_visible=False
        )
        show_chart(fig_main, key="entry_timing_candles", expand_height=750)

    with r1_c3:
        r1_val = safe(stock_info.get('resistance_60d'), current_price * 1.05)
        s1_val = safe(stock_info.get('support_60d'), current_price * 0.95)
        w20 = stock_daily.tail(20)
        w120 = stock_daily.tail(120)
        r2_val = round(float(w20['high'].max()), 2) if not w20.empty else r1_val
        r3_val = round(float(w120['high'].max()), 2) if not w120.empty else r1_val * 1.02
        s2_val = round(float(w20['low'].min()), 2) if not w20.empty else s1_val
        s3_val = round(float(w120['low'].min()), 2) if not w120.empty else s1_val * 0.98

        st.markdown(f"""<div style="background-color:#0F172A; border:1px solid #1E293B; border-radius:12px; padding:14px; min-height:495px; display:flex; flex-direction:column; justify-content:space-between;">
<div style="font-size:14.5px; font-weight:bold; color:#94A3B8; letter-spacing:0.5px;">KEY LEVELS (Actual, from price history)</div>
<div><div style="font-size:12.5px; font-weight:bold; color:#EF4444; margin-bottom:6px;">RESISTANCE</div>
<div style="display:flex; justify-content:space-between; font-size:13.5px; color:#CBD5E1; padding:4px 0; border-bottom:1px dashed #1E293B;"><span style="color:#64748B;">R3 (120d high)</span><b style="color:#F8FAFC;">{r3_val:.2f}</b></div>
<div style="display:flex; justify-content:space-between; font-size:13.5px; color:#CBD5E1; padding:4px 0; border-bottom:1px dashed #1E293B;"><span style="color:#64748B;">R2 (20d high)</span><b style="color:#F8FAFC;">{r2_val:.2f}</b></div>
<div style="display:flex; justify-content:space-between; font-size:13.5px; color:#CBD5E1; padding:4px 0;"><span style="color:#64748B;">R1 (60d high)</span><b style="color:#F8FAFC;">{r1_val:.2f}</b></div></div>
<div style="border:1px dashed #3B82F6; padding:10px 0; text-align:center; background:rgba(59,130,246,0.08); border-radius:8px; margin:auto 0;">
<div style="font-size:12.5px; color:#93C5FD; font-weight:bold;">CURRENT PRICE</div><div style="font-size:18.5px; font-weight:bold; color:#38BDF8; margin-top:2px;">{current_price:.2f}</div></div>
<div><div style="font-size:12.5px; font-weight:bold; color:#10B981; margin-bottom:6px;">SUPPORT</div>
<div style="display:flex; justify-content:space-between; font-size:13.5px; color:#CBD5E1; padding:4px 0; border-bottom:1px dashed #1E293B;"><span style="color:#64748B;">S1 (60d low)</span><b style="color:#F8FAFC;">{s1_val:.2f}</b></div>
<div style="display:flex; justify-content:space-between; font-size:13.5px; color:#CBD5E1; padding:4px 0; border-bottom:1px dashed #1E293B;"><span style="color:#64748B;">S2 (20d low)</span><b style="color:#F8FAFC;">{s2_val:.2f}</b></div>
<div style="display:flex; justify-content:space-between; font-size:13.5px; color:#CBD5E1; padding:4px 0;"><span style="color:#64748B;">S3 (120d low)</span><b style="color:#F8FAFC;">{s3_val:.2f}</b></div></div>
</div>""", unsafe_allow_html=True)

    with r1_c4:
        buy_zone_lo, buy_zone_hi = s1_val, round((s1_val + current_price) / 2, 2)
        stop_loss = s3_val
        st.markdown(f"""<div style="background-color:#0F172A; border:1px solid #1E293B; border-radius:12px; padding:14px; min-height:495px; display:flex; flex-direction:column; justify-content:space-between;">
<div style="font-size:14.5px; font-weight:bold; color:#94A3B8; letter-spacing:0.5px;">RECOMMENDED ZONE</div>
<div><div style="font-size:13px; color:#94A3B8;">Ideal Buy Zone</div>
<div style="font-size:17.5px; font-weight:bold; color:#10B981; margin-top:2px;">{buy_zone_lo:.2f} - {buy_zone_hi:.2f}</div>
<div style="font-size:12.5px; color:#64748B;">อ้างอิงจากแนวรับ 60 วันย้อนหลัง</div></div>
<div style="background:rgba(239,68,68,0.08); border-left:3px solid #EF4444; padding:8px 10px; border-radius:4px; margin-top:auto;">
<div style="font-size:12.5px; color:#EF4444; font-weight:bold;">STOP LOSS</div>
<div style="font-size:16px; font-weight:bold; color:#EF4444; margin-top:2px;">&lt; {stop_loss:.2f}</div>
<div style="font-size:12px; color:#94A3B8; margin-top:2px;">ตัดขาดทุนหากหลุดแนวรับ 120 วัน</div></div>
</div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:22px;'></div>", unsafe_allow_html=True)
    r2_c1, r2_c2, r2_c3, r2_c4, r2_c5 = st.columns(5)

    last30 = stock_daily.tail(30)
    rsi_now = safe(stock_info.get('rsi'), 50)
    macd_now = safe(stock_info.get('macd'), 0)
    adx_now = safe(stock_info.get('adx'), 20)

    def sparkline_svg(series, color, height=50):
        vals = series.dropna().tolist()
        if len(vals) < 2:
            return ""
        lo, hi = min(vals), max(vals)
        rng = (hi - lo) or 1
        w = 160
        pts = []
        for i, v in enumerate(vals):
            x = 5 + (w - 10) * i / (len(vals) - 1)
            y = 48 - ((v - lo) / rng) * 40
            pts.append(f"{x:.1f} {y:.1f}")
        path = "M " + " L ".join(pts)
        return f'<svg viewBox="0 0 {w} {height}" style="width:100%; height:{height}px; display:block;"><path d="{path}" fill="none" stroke="{color}" stroke-width="2"/></svg>'

    with r2_c1:
        macd_status = "BULLISH" if macd_now > 0 else "BEARISH"
        macd_color = "#10B981" if macd_now > 0 else "#EF4444"
        st.markdown(f"""<div style="background-color:#0F172A; border:1px solid #1E293B; border-radius:12px; padding:14px; min-height:220px; display:flex; flex-direction:column; justify-content:space-between;">
<div><div style="display:flex; justify-content:space-between; align-items:center;"><span style="font-size:14.5px; font-weight:bold; color:#CBD5E1;">MACD</span><span style="color:{macd_color}; font-size:13.5px; font-weight:bold;">{macd_status}</span></div>
<div style="font-size:23px; font-weight:bold; color:#FFFFFF; margin-top:4px;">{macd_now:.3f}</div></div>
<div style="margin-top:auto;">{sparkline_svg(last30['MACD'], macd_color)}</div></div>""", unsafe_allow_html=True)

    with r2_c2:
        rsi_status = "OVERBOUGHT" if rsi_now >= 70 else ("OVERSOLD" if rsi_now <= 30 else "NEUTRAL")
        rsi_color = "#EF4444" if rsi_now >= 70 else ("#10B981" if rsi_now <= 30 else "#F59E0B")
        st.markdown(f"""<div style="background-color:#0F172A; border:1px solid #1E293B; border-radius:12px; padding:14px; min-height:220px; display:flex; flex-direction:column; justify-content:space-between;">
<div><div style="display:flex; justify-content:space-between; align-items:center;"><span style="font-size:14.5px; font-weight:bold; color:#CBD5E1;">RSI (14)</span><span style="color:{rsi_color}; font-size:13.5px; font-weight:bold;">{rsi_status}</span></div>
<div style="font-size:26px; font-weight:bold; color:#FFFFFF; margin-top:4px;">{rsi_now:.1f}</div></div>
<div style="margin-top:auto;">{sparkline_svg(last30['RSI14'], '#A855F7')}</div></div>""", unsafe_allow_html=True)

    with r2_c3:
        adx_status = "STRONG TREND" if adx_now >= 25 else "WEAK / RANGE"
        adx_color = "#10B981" if adx_now >= 25 else "#94A3B8"
        st.markdown(f"""<div style="background-color:#0F172A; border:1px solid #1E293B; border-radius:12px; padding:14px; min-height:220px; display:flex; flex-direction:column; justify-content:space-between;">
<div><div style="display:flex; justify-content:space-between; align-items:center;"><span style="font-size:14.5px; font-weight:bold; color:#CBD5E1;">ADX (14)</span><span style="color:{adx_color}; font-size:13.5px; font-weight:bold;">{adx_status}</span></div>
<div style="font-size:26px; font-weight:bold; color:#FFFFFF; margin-top:4px;">{adx_now:.1f}</div></div>
<div style="margin-top:auto;">{sparkline_svg(last30['ADX'], '#F8FAFC')}</div></div>""", unsafe_allow_html=True)

    with r2_c4:
        ema_status = "GOLDEN (Bullish)" if stock_info.get('ema20', 0) > stock_info.get('ema50', 0) else "DEATH (Bearish)"
        ema_color = "#10B981" if stock_info.get('ema20', 0) > stock_info.get('ema50', 0) else "#EF4444"
        st.markdown(f"""<div style="background-color:#0F172A; border:1px solid #1E293B; border-radius:12px; padding:14px; min-height:220px; display:flex; flex-direction:column; justify-content:space-between;">
<div><div style="display:flex; justify-content:space-between; align-items:center;"><span style="font-size:14.5px; font-weight:bold; color:#CBD5E1;">EMA20 / EMA50</span><span style="color:{ema_color}; font-size:13.5px; font-weight:bold;">{ema_status}</span></div>
<div style="font-size:16.5px; font-weight:bold; color:#F59E0B; margin-top:4px;">{safe(stock_info.get('ema20')):.2f} <span style="color:#64748B; font-size:13.5px;">/</span> <span style="color:#38BDF8;">{safe(stock_info.get('ema50')):.2f}</span></div></div>
<div style="margin-top:auto;">{sparkline_svg(last30['EMA20'], '#F59E0B')}</div></div>""", unsafe_allow_html=True)

    with r2_c5:
        vol_now = safe(last30.iloc[-1]['volume']) if not last30.empty else 0
        vol_avg = safe(last30.iloc[-1]['Volume Avg']) if not last30.empty and 'Volume Avg' in last30.columns else vol_now
        vol_diff = ((vol_now - vol_avg) / vol_avg * 100) if vol_avg else 0
        vol_status = "Increasing" if vol_diff > 0 else "Decreasing"
        vol_color = "#10B981" if vol_diff > 0 else "#EF4444"
        st.markdown(f"""<div style="background-color:#0F172A; border:1px solid #1E293B; border-radius:12px; padding:14px; min-height:220px; display:flex; flex-direction:column; justify-content:space-between;">
<div><div style="font-size:14.5px; font-weight:bold; color:#CBD5E1;">VOLUME (vs 20D avg)</div>
<div style="font-size:16.5px; font-weight:bold; color:{vol_color}; margin-top:4px;">{vol_status}</div><div style="font-size:13px; color:#94A3B8;">{vol_diff:+.1f}% vs Avg.</div></div>
<div style="margin-top:auto;">{sparkline_svg(last30['volume'], vol_color)}</div></div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:22px;'></div>", unsafe_allow_html=True)
    r3_c1, r3_c2, r3_c3, r3_c4 = st.columns([1.1, 1.25, 1.4, 1.55])

    short_bull = current_price > safe(stock_info.get('ema20'))
    med_bull = safe(stock_info.get('ema20')) > safe(stock_info.get('ema50'))
    long_ref = stock_daily.iloc[max(0, len(stock_daily) - 252)]['close'] if len(stock_daily) > 0 else current_price
    long_bull = current_price > long_ref

    def trend_row(label, sub, is_bull):
        c = "#10B981" if is_bull else "#EF4444"
        txt = "Bullish ↗" if is_bull else "Bearish ↘"
        return f"""<div style="background:#151E2F; border:1px solid #1E293B; border-radius:8px; padding:10px 12px; display:flex; justify-content:space-between; align-items:center;">
<span style="font-size:14px; color:#CBD5E1;">{label} <span style="font-size:12.5px; color:#64748B;">({sub})</span></span>
<span style="background:rgba(16,185,129,0.15); color:{c}; font-size:14px; font-weight:bold; padding:3px 10px; border-radius:12px;">{txt}</span></div>"""

    with r3_c1:
        st.markdown(f"""<div style="background-color:#0F172A; border:1px solid #1E293B; border-radius:12px; padding:16px; min-height:295px; display:flex; flex-direction:column; justify-content:space-between;">
<div style="font-size:14.5px; font-weight:bold; color:#94A3B8; letter-spacing:0.5px;">TREND ANALYSIS</div>
<div style="display:flex; flex-direction:column; gap:10px; margin:auto 0;">
{trend_row("Short Term", "Price vs EMA20", short_bull)}
{trend_row("Medium Term", "EMA20 vs EMA50", med_bull)}
{trend_row("Long Term", "vs ~1Y ago", long_bull)}
</div></div>""", unsafe_allow_html=True)

    with r3_c2:
        summary_items = []
        summary_items.append((short_bull, "ราคาปัจจุบันอยู่เหนือ EMA20" if short_bull else "ราคาปัจจุบันอยู่ต่ำกว่า EMA20"))
        summary_items.append((med_bull, "EMA20 อยู่เหนือ EMA50 (แนวโน้มขาขึ้นระยะกลาง)" if med_bull else "EMA20 อยู่ต่ำกว่า EMA50 (แนวโน้มขาลงระยะกลาง)"))
        summary_items.append((macd_now > 0, "MACD เป็นบวก ส่งสัญญาณโมเมนตัมขาขึ้น" if macd_now > 0 else "MACD เป็นลบ ส่งสัญญาณโมเมนตัมขาลง"))
        summary_items.append((adx_now >= 25, f"ADX ที่ {adx_now:.1f} ยืนยันแนวโน้มแข็งแรง" if adx_now >= 25 else f"ADX ที่ {adx_now:.1f} บ่งชี้ตลาด sideway"))
        sig_html = "".join([f'<div style="display:flex; gap:8px;"><span style="color:{"#10B981" if ok else "#EF4444"}; font-size:15px;">{"✔" if ok else "✖"}</span><span>{txt}</span></div>' for ok, txt in summary_items])
        st.markdown(f"""<div style="background-color:#0F172A; border:1px solid #1E293B; border-radius:12px; padding:16px; min-height:295px; display:flex; flex-direction:column; justify-content:space-between;">
<div style="font-size:14.5px; font-weight:bold; color:#94A3B8; letter-spacing:0.5px;">SIGNAL SUMMARY</div>
<div style="font-size:14px; color:#CBD5E1; line-height:1.6; display:flex; flex-direction:column; gap:6px; margin:auto 0;">{sig_html}</div>
</div>""", unsafe_allow_html=True)

    with r3_c3:
        # หา EMA20/EMA50 crossover จริงในช่วง 120 วันล่าสุด
        hist120 = stock_daily.tail(120).copy().reset_index(drop=True)
        hist120['diff'] = hist120['EMA20'] - hist120['EMA50']
        hist120['cross'] = np.sign(hist120['diff']).diff().fillna(0)
        events = hist120[hist120['cross'] != 0].tail(5)
        rows_html = ""
        for _, ev in events.iloc[::-1].iterrows():
            label = "Golden Cross" if ev['cross'] > 0 else "Death Cross"
            lc = "#10B981" if ev['cross'] > 0 else "#EF4444"
            rows_html += f"""<tr style="border-bottom:1px solid #1E293B;"><td style="padding:4px 0; color:#94A3B8;">{ev['date'].strftime('%d %b %Y')}</td>
<td><span style="color:{lc}; font-weight:bold;">{label}</span></td><td>{ev['close']:.2f}</td><td style="color:#64748B;">EMA Crossover</td></tr>"""
        if not rows_html:
            rows_html = '<tr><td colspan="4" style="padding:8px 0; color:#64748B; text-align:center;">ไม่พบสัญญาณ Cross ในช่วง 120 วันล่าสุด</td></tr>'
        st.markdown(f"""<div style="background-color:#0F172A; border:1px solid #1E293B; border-radius:12px; padding:16px; min-height:295px; display:flex; flex-direction:column; justify-content:space-between;">
<div style="font-size:14.5px; font-weight:bold; color:#94A3B8; letter-spacing:0.5px;">RECENT SIGNAL HISTORY (EMA Crossovers, actual)</div>
<table style="width:100%; text-align:left; font-size:13.5px; color:#CBD5E1; border-collapse:collapse; margin:auto 0;">
<tr style="border-bottom:1px solid #1E293B; color:#64748B; font-size:12.5px;"><th style="padding:4px 0;">Date</th><th>Signal</th><th>Price</th><th>Type</th></tr>
{rows_html}
</table></div>""", unsafe_allow_html=True)

    with r3_c4:
        risk_lvl = "HIGH" if adx_now < 15 else ("MEDIUM" if adx_now < 25 else "LOW")
        risk_lvl_color = {"LOW": "#10B981", "MEDIUM": "#F59E0B", "HIGH": "#EF4444"}[risk_lvl]
        st.markdown(f"""<div style="background-color:#0F172A; border:1px solid #1E293B; border-radius:12px; padding:16px; min-height:295px; display:flex; flex-direction:column; justify-content:space-between;">
<div><div style="font-size:14.5px; font-weight:bold; color:#94A3B8; letter-spacing:0.5px; margin-bottom:6px;">TECHNICAL NOTES</div>
<p style="font-size:13px; color:#CBD5E1; line-height:1.55; margin:0;">
สัญญาณรวมล่าสุดของ {selected_ticker} คือ <b>{trend_signal}</b> (Timing Score {timing_score:.0f}/100) แนวรับใกล้สุดอยู่ที่ {s1_val:.2f} บาท และแนวต้านอยู่ที่ {r1_val:.2f} บาท หากราคาหลุด {s3_val:.2f} ควรพิจารณาตัดขาดทุน
</p></div>
<div style="display:flex; justify-content:space-between; align-items:flex-end; border-top:1px solid #1E293B; padding-top:10px;">
<div><div style="font-size:12.5px; color:#94A3B8; font-weight:bold; margin-bottom:4px;">RISK LEVEL (จาก ADX)</div>
<div style="font-size:16px; font-weight:bold; color:{risk_lvl_color};">{risk_lvl}</div></div>
<div style="text-align:right;"><div style="font-size:12.5px; color:#94A3B8; font-weight:bold;">MACD SIGNAL</div>
<div style="font-size:15px; font-weight:bold; color:{macd_color}; margin-top:2px;">{macd_status}</div></div>
</div></div>""", unsafe_allow_html=True)

    render_nav_footer("m3", prev_page=" ⚖️ Fair Value", next_page=" 🔮 AI Prediction")

# ============================================================================
# PAGE 5: MODULE 4 - AI PREDICTION
# ============================================================================
elif nav_page == " 🔮 AI Prediction":
    ai_score = int(round(safe(stock_info.get('ai_score'), 50)))
    ai_status = "BULLISH" if ai_score >= 70 else ("NEUTRAL" if ai_score >= 45 else "BEARISH")
    ai_color = "#10B981" if ai_score >= 70 else ("#F59E0B" if ai_score >= 45 else "#EF4444")
    prob_up = safe(stock_info.get('prob_up'), 50)
    down_prob = round(100 - prob_up, 1)

    st.markdown(f"""<div style="display:flex; justify-content:space-between; align-items:flex-end; margin-bottom:15px;">
<div><div style="font-size:14.5px; color:#64748B; margin-bottom:2px;">Home / Module 4 / AI Prediction</div>
<div style="display:flex; align-items:baseline; gap:8px;"><h2 style="margin:0; font-size:23px; font-weight:bold; color:#F8FAFC; letter-spacing:0.5px;">AI PREDICTION</h2></div></div>
<div style="text-align:right; display:flex; align-items:center; gap:16px;">
<div><span style="font-size:13px; color:#64748B;">Data as of</span><br><b style="color:#CBD5E1; font-size:15px;">{stock_info.get('latest_date','-')}</b></div>
<div><span style="font-size:13px; color:#64748B;">Model</span><br><b style="color:#38BDF8; font-size:15px;">Random Forest (n=200, depth=4)</b></div>
<div><span style="font-size:13px; color:#64748B;">Target</span><br><b style="color:#CBD5E1; font-size:15px;">10-Day Forward Direction</b></div>
</div></div>""", unsafe_allow_html=True)

    r1_c1, r1_c2, r1_c3 = st.columns([1.1, 1.25, 1.65])

    with r1_c1:
        ai_stars = min(5, max(1, round(ai_score / 20)))
        st.markdown(f"""<div style="background-color:#0F172A; border:1px solid #1E293B; border-radius:12px; padding:16px; min-height:260px; display:flex; flex-direction:column; justify-content:space-between;">
<div style="font-size:14.5px; font-weight:bold; color:#94A3B8; letter-spacing:0.5px;">AI PREDICTION SUMMARY</div>
<div style="display:flex; align-items:center; justify-content:space-between; margin:auto 0;">
<div style="text-align:center;"><div style="background:rgba(16,185,129,0.12); border:1px solid {ai_color}; border-radius:50%; width:78px; height:78px; display:flex; align-items:center; justify-content:center; font-size:28px; margin:0 auto 6px auto;">🔮</div>
<div style="color:{ai_color}; font-size:16px; font-weight:bold;">{ai_status}</div><div style="color:#64748B; font-size:12.5px; letter-spacing:0.5px;">PREDICTION</div>
<div style="color:{ai_color}; font-size:13px; letter-spacing:1px; margin-top:2px;">{'★'*ai_stars}{'☆'*(5-ai_stars)}</div></div>
<div style="text-align:right;"><div style="font-size:13px; color:#64748B;">Prediction Score</div><div style="font-size:28px; font-weight:bold; color:{ai_color}; line-height:1.1;">{ai_score}<span style="font-size:15px; color:#64748B;">/100</span></div>
<div style="font-size:13px; color:#64748B;">Test Accuracy</div><div style="font-size:16.5px; font-weight:bold; color:#10B981;">{safe(stock_info.get('accuracy')):.1f}%</div></div></div>
<p style="font-size:13px; color:#CBD5E1; line-height:1.35; margin:0;">โมเดล Random Forest คาดการณ์ทิศทางราคาหุ้น <b>{selected_ticker}</b> ใน 10 วันทำการถัดไป จาก technical indicators จริง (Train: 2023-2024 / Test: 2025)</p>
</div>""", unsafe_allow_html=True)

    with r1_c2:
        st.markdown(f"""<div style="background-color:#0F172A; border:1px solid #1E293B; border-radius:12px; padding:16px; min-height:260px; display:flex; flex-direction:column; justify-content:space-between; text-align:center;">
<div style="font-size:14.5px; font-weight:bold; color:#94A3B8; letter-spacing:0.5px; text-align:left;">PREDICTION PROBABILITY</div>
<div style="margin:auto 0;"><svg viewBox="0 0 100 55" style="width:150px; height:95px; display:block; margin:0 auto;">
<path d="M 10 50 A 40 40 0 0 1 90 50" fill="none" stroke="#1E293B" stroke-width="9" stroke-linecap="round" />
<path d="M 10 50 A 40 40 0 0 1 {10 + 80*min(1,prob_up/100):.1f} {50 - (40*np.sin(np.pi*min(1,prob_up/100))):.1f}" fill="none" stroke="{ai_color}" stroke-width="9" stroke-linecap="round" />
<text x="50" y="38" text-anchor="middle" font-size="20" font-weight="bold" fill="#FFFFFF">{prob_up:.0f}%</text>
<text x="50" y="47" text-anchor="middle" font-size="9.5" fill="#94A3B8">Probability of</text>
<text x="50" y="54" text-anchor="middle" font-size="10.5" font-weight="bold" fill="{ai_color}">{ai_status}</text></svg></div>
<div style="display:flex; justify-content:space-between; align-items:center; border-top:1px solid #1E293B; padding-top:8px;">
<div style="text-align:left;"><div style="font-size:12.5px; color:#64748B;">UPTREND</div><div style="font-size:16.5px; font-weight:bold; color:#10B981;">{prob_up:.0f}%</div></div>
<div style="text-align:right;"><div style="font-size:12.5px; color:#64748B;">DOWNTREND</div><div style="font-size:16.5px; font-weight:bold; color:#EF4444;">{down_prob:.0f}%</div></div>
</div></div>""", unsafe_allow_html=True)

    with r1_c3:
        n_train = len(stock_daily[stock_daily['date'] < '2025-01-01'])
        n_test = len(stock_daily[stock_daily['date'] >= '2025-01-01'])
        st.markdown(f"""<div style="background-color:#0F172A; border:1px solid #1E293B; border-radius:12px; padding:16px; min-height:260px; display:flex; flex-direction:column; justify-content:space-between;">
<div style="font-size:14.5px; font-weight:bold; color:#94A3B8; letter-spacing:0.5px;">MODEL & DATA SUMMARY</div>
<div style="display:grid; grid-template-columns: repeat(3, 1fr); gap:8px; margin-top:8px;">
<div style="background:#151E2F; border:1px solid #1E293B; border-radius:8px; padding:10px 8px; text-align:center;"><div style="font-size:12.5px; color:#64748B;">Train Samples</div><div style="font-size:19px; font-weight:bold; color:#FFFFFF; margin:3px 0;">{n_train}</div><div style="font-size:12px; color:#64748B;">2023-2024</div></div>
<div style="background:#151E2F; border:1.5px solid #2563EB; border-radius:8px; padding:10px 8px; text-align:center;"><div style="font-size:12.5px; color:#64748B;">Test Samples</div><div style="font-size:19px; font-weight:bold; color:#FFFFFF; margin:3px 0;">{n_test}</div><div style="font-size:12px; color:#64748B;">2025</div></div>
<div style="background:#151E2F; border:1px solid #1E293B; border-radius:8px; padding:10px 8px; text-align:center;"><div style="font-size:12.5px; color:#64748B;">Features</div><div style="font-size:19px; font-weight:bold; color:#FFFFFF; margin:3px 0;">6</div><div style="font-size:12px; color:#64748B;">Technical</div></div>
</div>
<p style="font-size:12.5px; color:#94A3B8; line-height:1.4; margin-top:10px;">โมเดลถูกฝึกแยกเป็นรายหุ้น โดยใช้ close, EMA20, EMA50, RSI14, MACD, ADX เป็น input และ label เป้าหมายคือราคาปิดใน 10 วันถัดไปสูงกว่าปัจจุบันหรือไม่</p>
</div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:22px;'></div>", unsafe_allow_html=True)
    r2_c1, r2_c2 = st.columns([1.55, 1.25])

    with r2_c1:
        st.markdown("""<div style="background-color:#0F172A; border:1px solid #1E293B; border-radius:12px 12px 0 0; padding:12px 16px 0 16px;">
<div><span style="font-size:14.5px; font-weight:bold; color:#94A3B8; letter-spacing:0.5px;">PRICE HISTORY + MODEL-IMPLIED FORWARD RANGE</span></div></div>""", unsafe_allow_html=True)

        hist_tail = stock_daily.tail(150)
        vol_annual = safe(stock_info.get('volatility'), 25.0) / 100
        daily_vol = vol_annual / np.sqrt(252)
        horizon_days = 10
        future_dates = pd.bdate_range(start=hist_tail['date'].iloc[-1], periods=horizon_days + 1)[1:]
        drift = (prob_up - 50) / 50 * daily_vol * horizon_days  # ทิศทางอิงจาก prob_up จริงของโมเดล
        t = np.arange(1, horizon_days + 1)
        median_path = current_price * (1 + drift * (t / horizon_days))
        band = current_price * daily_vol * np.sqrt(t) * 1.28  # ~80% band จาก volatility จริง
        upper_path = median_path + band
        lower_path = median_path - band

        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(x=hist_tail['date'], y=hist_tail['close'], mode='lines', line=dict(color='#38BDF8', width=2), name='Actual Price'))
        fig_forecast.add_trace(go.Scatter(x=future_dates, y=lower_path, mode='lines', line=dict(color='#EF4444', width=1.6, dash='dash'), name='Lower Bound (80%)'))
        fig_forecast.add_trace(go.Scatter(x=future_dates, y=median_path, mode='lines', line=dict(color='#10B981', width=2.0, dash='dash'), name='Model-Implied Median'))
        fig_forecast.add_trace(go.Scatter(x=future_dates, y=upper_path, mode='lines', line=dict(color='#2DD4BF', width=1.6, dash='dash'), name='Upper Bound (80%)'))

        fig_forecast.update_layout(
            height=310, margin=dict(l=35, r=45, t=10, b=25), paper_bgcolor="#0F172A", plot_bgcolor="#0F172A",
            xaxis=dict(gridcolor="#1E293B", tickfont=dict(size=11.5, color="#64748B"), zeroline=False),
            yaxis=dict(title=dict(text="Price (THB)", font=dict(size=12, color="#64748B")), gridcolor="#1E293B", tickfont=dict(size=11.5, color="#64748B"), zeroline=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, font=dict(size=11.5, color="#CBD5E1"))
        )
        show_chart(fig_forecast, key="ai_forecast", expand_height=700)
        st.markdown(f"""<div style="font-size:12px; color:#64748B; padding:0 16px 10px 16px; background:#0F172A; border:1px solid #1E293B; border-top:none; border-radius:0 0 12px 12px;">
* ช่วงคาดการณ์คำนวณจาก Annualized Volatility จริง ({safe(stock_info.get('volatility')):.1f}%) และความน่าจะเป็นขาขึ้นจากโมเดล ({prob_up:.0f}%) ไม่ใช่การรับประกันผลตอบแทน</div>""", unsafe_allow_html=True)

    with r2_c2:
        st.markdown("""<div style="background-color:#0F172A; border:1px solid #1E293B; border-radius:12px 12px 0 0; padding:12px 16px 0 16px;">
<div><span style="font-size:14.5px; font-weight:bold; color:#94A3B8; letter-spacing:0.5px;">FEATURE IMPORTANCE (Random Forest, actual)</span></div></div>""", unsafe_allow_html=True)

        fi = feat_imp_df[feat_imp_df['ticker'] == selected_ticker].sort_values('importance')
        if not fi.empty:
            fig_shap = go.Figure(go.Bar(
                x=fi['importance'], y=fi['feature'], orientation='h', marker=dict(color='#8B5CF6'),
                text=[f"{v:.3f}" for v in fi['importance']], textposition='outside', textfont=dict(size=11.5, color='#CBD5E1')
            ))
            fig_shap.update_layout(
                height=322, margin=dict(l=10, r=35, t=10, b=25), paper_bgcolor="#0F172A", plot_bgcolor="#0F172A",
                xaxis=dict(gridcolor="#1E293B", tickfont=dict(size=11.5, color="#64748B"), zeroline=False),
                yaxis=dict(tickfont=dict(size=11.5, color="#CBD5E1"), gridcolor="#1E293B", zeroline=False), showlegend=False
            )
            show_chart(fig_shap, key="ai_feature_importance", expand_height=650)
        else:
            st.info("ไม่มีข้อมูล Feature Importance")

    st.markdown("<div style='margin-top:22px;'></div>", unsafe_allow_html=True)
    r3_c1, r3_c2, r3_c3, r3_c4 = st.columns([1.1, 1.25, 1.35, 1.1])

    with r3_c1:
        st.markdown(f"""<div style="background-color:#0F172A; border:1px solid #1E293B; border-radius:12px; padding:14px; min-height:290px; display:flex; flex-direction:column; justify-content:space-between;">
<div><div style="font-size:13.5px; font-weight:bold; color:#94A3B8; letter-spacing:0.5px;">MODEL PERFORMANCE (TEST SET 2025, actual)</div>
<div style="display:grid; grid-template-columns: repeat(2, 1fr); gap:6px; margin-top:10px; text-align:center;">
<div style="background:#151E2F; border:1px solid #1E293B; border-radius:6px; padding:6px 2px;"><div style="font-size:12px; color:#64748B;">Accuracy</div><div style="font-size:16px; font-weight:bold; color:#F8FAFC;">{safe(stock_info.get('accuracy')):.1f}%</div></div>
<div style="background:#151E2F; border:1px solid #1E293B; border-radius:6px; padding:6px 2px;"><div style="font-size:12px; color:#64748B;">Precision</div><div style="font-size:16px; font-weight:bold; color:#F8FAFC;">{safe(stock_info.get('precision')):.1f}%</div></div>
<div style="background:#151E2F; border:1px solid #1E293B; border-radius:6px; padding:6px 2px;"><div style="font-size:12px; color:#64748B;">ROC-AUC</div><div style="font-size:16px; font-weight:bold; color:#F8FAFC;">{safe(stock_info.get('roc_auc')):.2f}</div></div>
<div style="background:#151E2F; border:1px solid #1E293B; border-radius:6px; padding:6px 2px;"><div style="font-size:12px; color:#64748B;">F1-Score</div><div style="font-size:16px; font-weight:bold; color:#F8FAFC;">{safe(stock_info.get('f1_score')):.1f}%</div></div>
</div></div><div style="font-size:11.5px; color:#64748B; border-top:1px solid #1E293B; padding-top:6px;">Validation: Out-of-time (Train 2023-24 / Test 2025)</div>
</div>""", unsafe_allow_html=True)

    with r3_c2:
        st.markdown("""<div style="background-color:#0F172A; border:1px solid #1E293B; border-radius:12px 12px 0 0; padding:12px 14px 0 14px;">
<div style="font-size:13.5px; font-weight:bold; color:#94A3B8; letter-spacing:0.5px;">HISTORICAL PREDICTION PERFORMANCE (Test Set, actual)</div></div>""", unsafe_allow_html=True)
        bt = backtest_df[backtest_df['ticker'] == selected_ticker].sort_values('date') if not backtest_df.empty else pd.DataFrame()
        if not bt.empty:
            bt_q = bt.set_index('date').resample('W').mean(numeric_only=True).dropna().reset_index()
            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(x=bt_q['date'], y=bt_q['actual_close'], mode='lines', name='Actual Close', line=dict(color='#38BDF8', width=1.5), yaxis='y1'))
            fig_bt.add_trace(go.Scatter(x=bt_q['date'], y=bt_q['predicted_up_prob'] * 100, mode='lines', name='Predicted Up Prob (%)', line=dict(color='#10B981', width=1.5, dash='dash'), yaxis='y2'))
            fig_bt.update_layout(
                height=120, margin=dict(l=25, r=25, t=5, b=15), paper_bgcolor="#0F172A", plot_bgcolor="#0F172A",
                xaxis=dict(tickfont=dict(size=10.5, color="#64748B"), gridcolor="#1E293B"),
                yaxis=dict(tickfont=dict(size=10.5, color="#64748B"), gridcolor="#1E293B", zeroline=False),
                yaxis2=dict(overlaying='y', side='right', showgrid=False, tickfont=dict(size=10.5, color="#64748B")),
                showlegend=False
            )
            show_chart(fig_bt, key="ai_backtest", expand_height=550)
            hit_rate = ((bt['predicted_up_prob'] > 0.5).astype(int) == (bt['actual_close'].diff().shift(-1) > 0).astype(int)).mean() * 100
            st.markdown(f"""<div style="background:#0F172A; border:1px solid #1E293B; border-top:none; border-radius:0 0 12px 12px; padding:0 12px 10px 12px; font-size:11.5px; color:#64748B;">* Test-set Accuracy: {safe(stock_info.get('accuracy')):.1f}%</div>""", unsafe_allow_html=True)
        else:
            st.info("ไม่มีข้อมูล Backtest")

    with r3_c3:
        top_feat = fi.sort_values('importance', ascending=False).iloc[0]['feature'] if not fi.empty else "N/A"
        st.markdown(f"""<div style="background-color:#0F172A; border:1px solid #1E293B; border-radius:12px; padding:14px; min-height:290px; display:flex; flex-direction:column; justify-content:space-between;">
<div><div style="font-size:13.5px; font-weight:bold; color:#94A3B8; letter-spacing:0.5px; margin-bottom:6px;">EXPLAINABLE AI SUMMARY ({selected_ticker})</div>
<p style="font-size:13px; color:#CBD5E1; line-height:1.45; margin:0 0 8px 0;">โมเดลประเมินความน่าจะเป็นขาขึ้นสำหรับ <b>{selected_ticker}</b> อยู่ที่ <b>{prob_up:.0f}%</b> โดย feature ที่มีอิทธิพลสูงสุดคือ <b>{top_feat}</b>:</p>
<div style="font-size:12.5px; color:#CBD5E1; line-height:1.5; display:flex; flex-direction:column; gap:4px;">
<div style="display:flex; gap:6px;"><span style="color:#10B981;">✔</span><span>Test Accuracy บนข้อมูลปี 2025 อยู่ที่ {safe(stock_info.get('accuracy')):.1f}%</span></div>
<div style="display:flex; gap:6px;"><span style="color:#10B981;">✔</span><span>ROC-AUC = {safe(stock_info.get('roc_auc')):.2f} (ยิ่งใกล้ 1 ยิ่งแยกแยะได้ดี)</span></div>
<div style="display:flex; gap:6px;"><span style="color:#10B981;">✔</span><span>Signal ปัจจุบัน: {stock_info.get('ai_signal','-')}</span></div>
</div></div></div>""", unsafe_allow_html=True)

    with r3_c4:
        st.markdown(f"""<div style="background-color:#0F172A; border:1px solid #1E293B; border-radius:12px; padding:14px; min-height:290px; display:flex; flex-direction:column; justify-content:space-between;">
<div><div style="font-size:13.5px; font-weight:bold; color:#94A3B8; letter-spacing:0.5px;">AI RECOMMENDATION</div>
<div style="display:flex; align-items:center; gap:8px; margin:8px 0 4px 0;"><div>
<div style="font-size:23px; font-weight:bold; color:{ai_color}; line-height:1;">{stock_info.get('ai_signal','-')}</div>
<div style="font-size:12.5px; font-weight:bold; color:{ai_color};">Prob. Up: {prob_up:.0f}%</div></div></div>
<div style="font-size:12.5px; color:#CBD5E1; border-top:1px dashed #1E293B; padding-top:6px; margin-top:4px;">
<div style="display:flex; justify-content:space-between; margin-bottom:3px;"><span style="color:#64748B;">Target Period</span><b style="color:#F8FAFC;">10 Trading Days</b></div>
<div style="display:flex; justify-content:space-between;"><span style="color:#64748B;">Model Accuracy</span><b style="color:#F59E0B;">{safe(stock_info.get('accuracy')):.1f}%</b></div>
</div></div><div style="font-size:11.5px; color:#64748B; text-align:center;">โปรดใช้ประกอบการตัดสินใจลงทุน ไม่ใช่คำแนะนำโดยตรง</div>
</div>""", unsafe_allow_html=True)

    render_nav_footer("m4", prev_page=" ⏱️ Entry Timing", next_page=" 🛡️ Risk Analysis")

# ============================================================================
# PAGE 6: MODULE 5 - RISK ANALYSIS
# ============================================================================
elif nav_page == " 🛡️ Risk Analysis":
    # หมายเหตุ: หน้านี้แสดงผลแยกเป็นองค์ประกอบ (component-based) ตามหลักการที่ตกลงกัน
    # จะไม่รวมทุกมิติเป็น "คะแนนความเสี่ยงเดียว" เพราะทำให้เสีย information — คนละสาเหตุแต่ตัวเลขออกมาเท่ากันได้
    # risk_score (0-100) ยังคงมีอยู่ในระบบและถูกใช้ในหน้า Overview / Industry Benchmark เพื่อความสอดคล้องของภาพรวมเท่านั้น

    beta_val = safe(stock_info.get('beta'), 1.0)
    sector_beta_val = safe(stock_info.get('sector_relative_beta'), beta_val)
    downside_beta_val = safe(stock_info.get('downside_beta'), beta_val)
    vol_val = safe(stock_info.get('volatility'), 25.0)
    dd_val = safe(stock_info.get('max_drawdown'), 20.0)
    calmar_val = safe(stock_info.get('calmar_ratio'), 0.0)
    var95_val = safe(stock_info.get('var_95'), 0.0)
    cvar95_val = safe(stock_info.get('cvar_95'), var95_val)
    sharpe_val = safe(stock_info.get('sharpe_ratio'), 0.0)
    psr_val = safe(stock_info.get('psr_pct'), 50.0)
    sortino_val = safe(stock_info.get('sortino_ratio'), 0.0)
    downside_dev_val = safe(stock_info.get('downside_deviation'), 0.0)
    adv_val = safe(stock_info.get('avg_daily_value_mb'), 0.0)
    turnover_val = safe(stock_info.get('turnover_ratio_pct'), 0.0)
    worst20d_val = safe(stock_info.get('worst_20d_move_pct'), 0.0)
    crash_impact = safe(stock_info.get('beta_crash_impact_pct'), round(beta_val * -20, 1))
    recovery_status = stock_info.get('recovery_status', '-')
    peak_date = stock_info.get('peak_date', '-')
    trough_date = stock_info.get('trough_date', '-')
    recovery_date_val = stock_info.get('recovery_date')
    recovery_days = int(safe(stock_info.get('recovery_duration_days'), 0))

    st.markdown(f"""<div style="display:flex; justify-content:space-between; align-items:flex-end; margin-bottom:15px;">
<div><div style="font-size:14.5px; color:#64748B; margin-bottom:2px;">Home / Module 5 / Risk Analysis</div>
<div style="display:flex; align-items:baseline; gap:8px;"><h2 style="margin:0; font-size:23px; font-weight:bold; color:#F8FAFC; letter-spacing:0.5px;">RISK ANALYSIS</h2></div>
<div style="font-size:13.5px; color:#94A3B8; margin-top:2px;">แสดงผลแยกตามองค์ประกอบ (ไม่รวมเป็นคะแนนเดียว) เพื่อไม่ให้เสีย information ในการตัดสินใจ</div></div>
<div style="text-align:right; display:flex; align-items:center; gap:16px;">
<div><span style="font-size:13px; color:#64748B;">Analysis Date</span><br><b style="color:#CBD5E1; font-size:15px;">{stock_info.get('latest_date','-')}</b></div>
<div><span style="font-size:13px; color:#64748B;">Data Period</span><br><b style="color:#CBD5E1; font-size:15px;">2023-2025 (3Y)</b></div>
</div></div>""", unsafe_allow_html=True)

    def metric_card(label, value, unit, formula, color="#F8FAFC", sub=None):
        sub_html = f'<div style="font-size:11.5px; color:#64748B; margin-top:3px;">{sub}</div>' if sub else ""
        return f"""<div style="background-color:#0F172A; border:1px solid #1E293B; border-radius:12px; padding:14px; min-height:150px; display:flex; flex-direction:column; justify-content:space-between;">
<div style="font-size:12.5px; font-weight:bold; color:#94A3B8; letter-spacing:0.3px;">{label}</div>
<div style="font-size:24px; font-weight:bold; color:{color}; margin:6px 0 2px 0;">{value}<span style="font-size:13px; color:#64748B; font-weight:normal;">{unit}</span></div>
<div>{sub_html}<div style="font-size:11px; color:#475569; border-top:1px solid #1E293B; padding-top:5px; margin-top:5px; line-height:1.35;">📐 {formula}</div></div>
</div>"""

    # ---------------- ส่วนที่ 1: สูตรเดิม (ไม่เปลี่ยน) ----------------
    st.markdown("""<div style="font-size:14.5px; font-weight:bold; color:#F8FAFC; letter-spacing:0.5px; margin-bottom:8px;">
ส่วนที่ 1 — MARKET & PRICE RISK <span style="font-size:13px; color:#94A3B8; font-weight:normal;">(สูตรมาตรฐานสากล ไม่ปรับเปลี่ยน)</span></div>""", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(metric_card("BETA (CAPM)", f"{beta_val:.2f}", "", "Cov(R_stock, R_SET)/Var(R_SET) — ค่าจริงจาก stock_risk_metrics.csv",
                                 "#EF4444" if beta_val > 1 else "#10B981"), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card("ANNUALIZED VOLATILITY", f"{vol_val:.1f}", "%", "std(daily_returns) × √252 × 100", "#F59E0B"), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card("MAX DRAWDOWN (MDD)", f"-{dd_val:.1f}", "%", "min[(Price_t − Cummax(Price)) / Cummax(Price)] × 100", "#EF4444"), unsafe_allow_html=True)
    with c4:
        st.markdown(metric_card("CALMAR RATIO", f"{calmar_val:.2f}", "", "Average_Annual_Return / |Max_Drawdown|",
                                 "#10B981" if calmar_val > 0.5 else "#F59E0B"), unsafe_allow_html=True)

    st.markdown("<div style='margin-top:18px;'></div>", unsafe_allow_html=True)
    r2_c1, r2_c2, r2_c3 = st.columns(3)
    rh = risk_hist_df[risk_hist_df['ticker'] == selected_ticker].sort_values('date') if not risk_hist_df.empty else pd.DataFrame()

    with r2_c1:
        st.markdown("""<div style="background-color:#0F172A; border:1px solid #1E293B; border-radius:12px 12px 0 0; padding:12px 14px 0 14px;">
<div style="font-size:14px; font-weight:bold; color:#94A3B8; letter-spacing:0.5px;">BETA เทียบ Peers (8 หุ้น)</div></div>""", unsafe_allow_html=True)
        beta_cmp = scores_df[['ticker', 'beta']].sort_values('beta')
        colors_beta = ['#A855F7' if t == selected_ticker else '#38BDF8' for t in beta_cmp['ticker']]
        fig_beta = go.Figure(go.Bar(x=beta_cmp['beta'], y=beta_cmp['ticker'], orientation='h', marker=dict(color=colors_beta)))
        fig_beta.add_vline(x=1.0, line_width=1, line_dash="dash", line_color="#64748B")
        fig_beta.update_layout(height=160, margin=dict(l=40, r=10, t=10, b=20), paper_bgcolor="#0F172A", plot_bgcolor="#0F172A",
            xaxis=dict(tickfont=dict(size=11, color="#64748B"), gridcolor="#1E293B"),
            yaxis=dict(tickfont=dict(size=11, color="#CBD5E1"), gridcolor="#1E293B"), showlegend=False)
        show_chart(fig_beta, key="risk_beta", expand_height=550)

    with r2_c2:
        st.markdown("""<div style="background-color:#0F172A; border:1px solid #1E293B; border-radius:12px 12px 0 0; padding:12px 14px 0 14px;">
<div style="font-size:14px; font-weight:bold; color:#94A3B8; letter-spacing:0.5px;">Rolling 30D Volatility (actual)</div></div>""", unsafe_allow_html=True)
        if not rh.empty:
            fig_vol = go.Figure()
            fig_vol.add_trace(go.Scatter(x=rh['date'], y=rh['rolling_vol_30d'], mode='lines', line=dict(color='#38BDF8', width=1.8)))
            fig_vol.update_layout(height=160, margin=dict(l=30, r=10, t=10, b=20), paper_bgcolor="#0F172A", plot_bgcolor="#0F172A",
                xaxis=dict(tickfont=dict(size=11, color="#64748B"), gridcolor="#1E293B"),
                yaxis=dict(tickfont=dict(size=11, color="#64748B"), gridcolor="#1E293B", zeroline=False), showlegend=False)
            show_chart(fig_vol, key="risk_volatility", expand_height=550)
        else:
            st.info("ไม่มีข้อมูล")

    with r2_c3:
        st.markdown("""<div style="background-color:#0F172A; border:1px solid #1E293B; border-radius:12px 12px 0 0; padding:12px 14px 0 14px;">
<div style="font-size:14px; font-weight:bold; color:#94A3B8; letter-spacing:0.5px;">Drawdown — Actual (2023-2025)</div></div>""", unsafe_allow_html=True)
        if not rh.empty:
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(x=rh['date'], y=rh['drawdown_pct'], mode='lines', line=dict(color='#EF4444', width=1.5), fill='tozeroy', fillcolor='rgba(239,68,68,0.2)'))
            fig_dd.update_layout(height=160, margin=dict(l=30, r=10, t=10, b=20), paper_bgcolor="#0F172A", plot_bgcolor="#0F172A",
                xaxis=dict(tickfont=dict(size=11, color="#64748B"), gridcolor="#1E293B"),
                yaxis=dict(tickfont=dict(size=11, color="#64748B"), gridcolor="#1E293B", zeroline=False), showlegend=False)
            show_chart(fig_dd, key="risk_drawdown", expand_height=550)
        else:
            st.info("ไม่มีข้อมูล")

    # ---------------- ส่วนที่ 2.1 / 3.4: Downside Beta & Sector-Relative Beta (peer proxy) ----------------
    st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)
    st.markdown("""<div style="font-size:14.5px; font-weight:bold; color:#F8FAFC; letter-spacing:0.5px; margin-bottom:8px;">
ส่วนที่ 2.1 / 3.4 — DOWNSIDE BETA & SECTOR-RELATIVE BETA <span style="font-size:13px; color:#94A3B8; font-weight:normal;">(Peer-Proxy Benchmark)</span></div>""", unsafe_allow_html=True)
    pb_c1, pb_c2, pb_c3 = st.columns([1, 1, 1.4])
    with pb_c1:
        st.markdown(metric_card("SECTOR-RELATIVE BETA", f"{sector_beta_val:.2f}", "",
                                 "Cov(R_stock, R_sector)/Var(R_sector), ทุกวัน", "#38BDF8"), unsafe_allow_html=True)
    with pb_c2:
        st.markdown(metric_card("DOWNSIDE BETA (β⁻)", f"{downside_beta_val:.2f}", "",
                                 "Cov/Var เฉพาะวันที่ R_sector < 0 (Ang, Chen & Xing 2006)",
                                 "#EF4444" if downside_beta_val > sector_beta_val else "#F59E0B"), unsafe_allow_html=True)
    with pb_c3:
        st.markdown(f"""<div style="background-color:rgba(245,158,11,0.08); border-left:3px solid #F59E0B; border-radius:8px; padding:14px; min-height:150px; display:flex; flex-direction:column; justify-content:center;">
<div style="font-size:12.5px; font-weight:bold; color:#F59E0B; margin-bottom:4px;">⚠️ ข้อจำกัดของข้อมูล</div>
<div style="font-size:12.5px; color:#CBD5E1; line-height:1.5;">ระบบไม่มีราคาปิดรายวันของ SET Index ย้อนหลัง (README ตัด yfinance ออกเพื่อไม่พึ่ง internet ตอน deploy) ตัวเลขนี้จึงใช้ <b>ค่าเฉลี่ยผลตอบแทนหุ้นกลุ่ม {stock_info.get('sector','-')} (ไม่รวม {selected_ticker} เอง)</b> เป็น benchmark proxy แทน ไม่ใช่ Beta เทียบตลาดจริง</div>
</div>""", unsafe_allow_html=True)

    # ---------------- ส่วนที่ 2.2 / 3.1: VaR & CVaR ----------------
    st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)
    st.markdown("""<div style="font-size:14.5px; font-weight:bold; color:#F8FAFC; letter-spacing:0.5px; margin-bottom:8px;">
ส่วนที่ 2.2 / 3.1 — TAIL RISK: VALUE AT RISK & EXPECTED SHORTFALL</div>""", unsafe_allow_html=True)
    v_c1, v_c2 = st.columns(2)
    with v_c1:
        st.markdown(metric_card("VaR 95% (Parametric)", f"-{var95_val:.2f}", "%",
                                 "1.645 × daily_std × 100 (Jorion 2007) — สมมติ Normal Distribution",
                                 "#F59E0B", sub="ขาดทุนสูงสุดที่คาดไว้ใน 1 วัน ด้วยความเชื่อมั่น 95%"), unsafe_allow_html=True)
    with v_c2:
        st.markdown(metric_card("CVaR 95% (Expected Shortfall)", f"-{cvar95_val:.2f}", "%",
                                 "ค่าเฉลี่ยผลตอบแทนใน 5% ที่แย่ที่สุดจริง (Historical Simulation) — มาตรฐาน Basel Committee",
                                 "#EF4444", sub="ถ้าเกิดใน 5% ที่แย่ที่สุด จะเสียหายเฉลี่ยเท่านี้ (VaR ตอบคำถามนี้ไม่ได้)"), unsafe_allow_html=True)

    # ---------------- ส่วนที่ 2.3 / 2.4: Sortino + MAR, Sharpe + PSR ----------------
    st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)
    st.markdown(f"""<div style="font-size:14.5px; font-weight:bold; color:#F8FAFC; letter-spacing:0.5px; margin-bottom:8px;">
ส่วนที่ 2.3 / 2.4 — RISK-ADJUSTED RETURN <span style="font-size:13px; color:#94A3B8; font-weight:normal;">(Risk-free rate = {RISK_FREE_RATE_PCT:.2f}% ต่อปี, อ้างอิง ThaiBMA 1Y Gov Bond)</span></div>""", unsafe_allow_html=True)
    s_c1, s_c2 = st.columns(2)
    with s_c1:
        st.markdown(f"""<div style="background-color:#0F172A; border:1px solid #1E293B; border-radius:12px; padding:14px; min-height:165px; display:flex; flex-direction:column; justify-content:space-between;">
<div style="font-size:12.5px; font-weight:bold; color:#94A3B8;">SHARPE RATIO + PSR</div>
<div style="display:flex; align-items:baseline; gap:14px; margin:6px 0;">
<div><span style="font-size:24px; font-weight:bold; color:{'#10B981' if sharpe_val>0.5 else '#F59E0B'};">{sharpe_val:.2f}</span><span style="font-size:11.5px; color:#64748B;"> Sharpe</span></div>
<div><span style="font-size:20px; font-weight:bold; color:#A855F7;">{psr_val:.0f}%</span><span style="font-size:11.5px; color:#64748B;"> PSR</span></div>
</div>
<div style="font-size:11px; color:#475569; border-top:1px solid #1E293B; padding-top:5px; line-height:1.4;">📐 Sharpe = (Annual_Return − Risk_free)/Annual_Std<br>PSR = Φ((SR−SR*)×√(n−1) / √(1−γ₃SR+((γ₄−1)/4)SR²)) — ความน่าจะเป็นว่า Sharpe จริง &gt; {PSR_BENCHMARK_SR_DISPLAY} ไม่ใช่ความบังเอิญ (Bailey &amp; López de Prado, 2012)</div>
</div>""", unsafe_allow_html=True)
    with s_c2:
        st.markdown(f"""<div style="background-color:#0F172A; border:1px solid #1E293B; border-radius:12px; padding:14px; min-height:165px; display:flex; flex-direction:column; justify-content:space-between;">
<div style="font-size:12.5px; font-weight:bold; color:#94A3B8;">SORTINO RATIO + MAR</div>
<div style="display:flex; align-items:baseline; gap:14px; margin:6px 0;">
<div><span style="font-size:24px; font-weight:bold; color:{'#10B981' if sortino_val>0.5 else '#F59E0B'};">{sortino_val:.2f}</span><span style="font-size:11.5px; color:#64748B;"> Sortino</span></div>
<div><span style="font-size:16px; font-weight:bold; color:#F8FAFC;">{downside_dev_val:.1f}%</span><span style="font-size:11.5px; color:#64748B;"> Downside Dev.</span></div>
</div>
<div style="font-size:11px; color:#475569; border-top:1px solid #1E293B; padding-top:5px; line-height:1.4;">📐 MAR = Risk-free rate รายปี ; Downside_Dev = √(mean(min(0, r_i−MAR/252))²)×√252<br>Sortino = (Annual_Return − MAR)/Downside_Deviation (Sortino &amp; van der Meer)</div>
</div>""", unsafe_allow_html=True)

    # ---------------- ส่วนที่ 2.5: Recovery Duration ----------------
    st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)
    st.markdown("""<div style="font-size:14.5px; font-weight:bold; color:#F8FAFC; letter-spacing:0.5px; margin-bottom:8px;">
ส่วนที่ 2.5 — RECOVERY DURATION <span style="font-size:13px; color:#94A3B8; font-weight:normal;">(จากจุด Max Drawdown จริง)</span></div>""", unsafe_allow_html=True)
    rec_color = "#10B981" if recovery_status == "Recovered" else "#F59E0B"
    rec_badge = "✅ RECOVERED" if recovery_status == "Recovered" else "⏳ ONGOING"
    rec_detail = (f"ฟื้นตัวกลับสู่จุดสูงสุดเดิมแล้ว ใช้เวลา <b>{recovery_days} วันทำการ</b> (Recovery Date: {recovery_date_val})"
                  if recovery_status == "Recovered" else
                  f"ยังไม่ฟื้นตัวกลับสู่จุดสูงสุดเดิม เป็นเวลา <b>{recovery_days} วันทำการ</b> จนถึงข้อมูลล่าสุด")
    st.markdown(f"""<div style="background-color:#0F172A; border:1px solid #1E293B; border-radius:12px; padding:16px;">
<div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:10px;">
<div style="display:flex; align-items:center; gap:18px;">
<div style="text-align:center;"><div style="font-size:11.5px; color:#64748B;">PEAK</div><div style="font-size:14px; font-weight:bold; color:#F8FAFC;">{peak_date}</div></div>
<div style="color:#64748B;">→</div>
<div style="text-align:center;"><div style="font-size:11.5px; color:#EF4444;">TROUGH (MDD)</div><div style="font-size:14px; font-weight:bold; color:#EF4444;">{trough_date}</div></div>
<div style="color:#64748B;">→</div>
<div style="text-align:center;"><div style="font-size:11.5px; color:{rec_color};">RECOVERY</div><div style="font-size:14px; font-weight:bold; color:{rec_color};">{recovery_date_val if recovery_date_val else '—'}</div></div>
</div>
<span style="background:rgba(16,185,129,0.15); color:{rec_color}; font-size:13px; font-weight:bold; padding:5px 12px; border-radius:8px;">{rec_badge}</span>
</div>
<div style="font-size:13px; color:#CBD5E1; margin-top:10px; line-height:1.5;">{rec_detail}</div>
<div style="font-size:11px; color:#475569; border-top:1px solid #1E293B; padding-top:6px; margin-top:8px;">📐 Recovery_Duration = recovery_date − trough_date (วันทำการ) ; หากยังไม่ฟื้นตัว → "Ongoing" + จำนวนวันที่ผ่านมา</div>
</div>""", unsafe_allow_html=True)

    # ---------------- ส่วนที่ 3.3: Liquidity Risk ----------------
    st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)
    st.markdown("""<div style="font-size:14.5px; font-weight:bold; color:#F8FAFC; letter-spacing:0.5px; margin-bottom:8px;">
ส่วนที่ 3.3 — LIQUIDITY RISK (Market-based) <span style="font-size:13px; color:#94A3B8; font-weight:normal;">(สภาพคล่องซื้อขาย ไม่ใช่สภาพคล่องบัญชี — คำนวณจาก 90 วันล่าสุด)</span></div>""", unsafe_allow_html=True)
    l_c1, l_c2, l_c3 = st.columns([1, 1, 1.6])
    with l_c1:
        st.markdown(metric_card("AVG. DAILY VALUE", f"{adv_val:,.1f}", " MB",
                                 "mean(Close × Volume) ย้อนหลัง 90 วัน", "#38BDF8"), unsafe_allow_html=True)
    with l_c2:
        st.markdown(metric_card("TURNOVER RATIO", f"{turnover_val:.3f}", "%",
                                 "mean(Volume) / Shares_Outstanding × 100 (90 วัน)", "#38BDF8"), unsafe_allow_html=True)
    with l_c3:
        liq_cmp = scores_df[['ticker', 'avg_daily_value_mb']].sort_values('avg_daily_value_mb')
        colors_liq = ['#A855F7' if t == selected_ticker else '#38BDF8' for t in liq_cmp['ticker']]
        fig_liq = go.Figure(go.Bar(x=liq_cmp['avg_daily_value_mb'], y=liq_cmp['ticker'], orientation='h', marker=dict(color=colors_liq)))
        fig_liq.update_layout(height=150, margin=dict(l=45, r=15, t=8, b=20), paper_bgcolor="#0F172A", plot_bgcolor="#0F172A",
            xaxis=dict(title=dict(text="Avg Daily Value (MB)", font=dict(size=10.5, color="#64748B")), tickfont=dict(size=10.5, color="#64748B"), gridcolor="#1E293B"),
            yaxis=dict(tickfont=dict(size=10.5, color="#CBD5E1"), gridcolor="#1E293B"), showlegend=False)
        show_chart(fig_liq, key="risk_liquidity", expand_height=520)

    # ---------------- ส่วนที่ 3.2: Correlation Matrix ----------------
    st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)
    st.markdown("""<div style="font-size:14.5px; font-weight:bold; color:#F8FAFC; letter-spacing:0.5px; margin-bottom:8px;">
ส่วนที่ 3.2 — CORRELATION MATRIX <span style="font-size:13px; color:#94A3B8; font-weight:normal;">(ระหว่างหุ้นทั้ง 8 ตัว จาก daily returns จริง — ใช้ประเมิน diversification benefit)</span></div>""", unsafe_allow_html=True)
    if not corr_matrix_df.empty:
        pivot_corr = corr_matrix_df.pivot(index='ticker_a', columns='ticker_b', values='correlation').reindex(index=TARGET_STOCKS, columns=TARGET_STOCKS)
        fig_corr = go.Figure(data=go.Heatmap(
            z=pivot_corr.values, x=pivot_corr.columns.tolist(), y=pivot_corr.index.tolist(),
            zmin=-1, zmax=1, colorscale=[[0, '#EF4444'], [0.5, '#151E2F'], [1, '#10B981']],
            text=np.round(pivot_corr.values, 2), texttemplate="%{text}", textfont=dict(size=11, color='#F1F5F9'),
            colorbar=dict(tickfont=dict(color='#94A3B8', size=10.5))
        ))
        fig_corr.update_layout(height=330, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="#0F172A", plot_bgcolor="#0F172A",
            xaxis=dict(tickfont=dict(color='#CBD5E1', size=11.5)), yaxis=dict(tickfont=dict(color='#CBD5E1', size=11.5), autorange='reversed'))
        show_chart(fig_corr, key="risk_correlation", expand_height=650)
        st.caption("📐 ρ(i,j) = Cov(R_i, R_j) / (σ_i × σ_j) — ค่าใกล้ +1 = เคลื่อนไหวไปทางเดียวกันมาก (diversify ได้น้อย), ใกล้ -1 หรือ 0 = diversify ได้ดี")
    else:
        st.info("ไม่มีข้อมูล Correlation Matrix")

    # ---------------- ส่วนที่ 5 (ปรับปรุง): Stress Test ----------------
    st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)
    st.markdown("""<div style="font-size:14.5px; font-weight:bold; color:#F8FAFC; letter-spacing:0.5px; margin-bottom:8px;">
STRESS TEST SCENARIO <span style="font-size:13px; color:#94A3B8; font-weight:normal;">(ตัดสถานการณ์ CPI/M2 ออก เพราะไม่มีข้อมูลย้อนหลังในระบบ)</span></div>""", unsafe_allow_html=True)
    st.markdown(f"""<div style="background-color:#0F172A; border:1px solid #1E293B; border-radius:12px; padding:14px;">
<table style="width:100%; font-size:13.5px; color:#CBD5E1; border-collapse:collapse;">
<tr style="border-bottom:1px solid #1E293B; color:#64748B; font-size:12.5px;"><th style="text-align:left; padding:5px 0;">Scenario</th><th style="text-align:left;">วิธีคำนวณ</th><th style="text-align:right;">Est. Impact</th></tr>
<tr style="border-bottom:1px solid #1E293B;"><td style="padding:6px 0;">Market Crash (SET {MARKET_CRASH_SHOCK_PCT:.0f}%)</td><td style="color:#64748B; font-size:12.5px;">Beta × Market Shock (CAPM)</td><td style="text-align:right; color:#EF4444; font-weight:bold;">{crash_impact:+.1f}%</td></tr>
<tr><td style="padding:6px 0;">Worst Historical 20-Day Move</td><td style="color:#64748B; font-size:12.5px;">min(rolling 20-day return) จริงจากราคาย้อนหลัง</td><td style="text-align:right; color:#EF4444; font-weight:bold;">{worst20d_val:+.1f}%</td></tr>
</table>
<div style="font-size:11.5px; color:#64748B; border-top:1px solid #1E293B; padding-top:6px; margin-top:8px;">Beta ที่ใช้ = {beta_val:.2f} (จาก stock_risk_metrics.csv จริง) ; Worst 20D Move เป็นเหตุการณ์จริงที่เคยเกิดขึ้น ไม่ใช่การประมาณ</div>
</div>""", unsafe_allow_html=True)

    # ---------------- Explainable Risk Summary (แยกองค์ประกอบ ไม่รวมเป็นเลขเดียว) ----------------
    st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)
    st.markdown(f"""<div style="background-color:#0F172A; border:1px solid #1E293B; border-radius:12px; padding:16px;">
<div style="font-size:14px; font-weight:bold; color:#94A3B8; letter-spacing:0.5px; margin-bottom:8px;">EXPLAINABLE RISK SUMMARY ({selected_ticker})</div>
<p style="font-size:13px; color:#CBD5E1; line-height:1.6; margin:0 0 8px 0;">
Beta = {beta_val:.2f} (Sector-Relative {sector_beta_val:.2f}, Downside {downside_beta_val:.2f}) • Volatility รายปี = {vol_val:.1f}% • Max Drawdown = {dd_val:.1f}% ({recovery_status})
• VaR 95% = -{var95_val:.2f}% ต่อวัน, CVaR 95% = -{cvar95_val:.2f}% • Sharpe = {sharpe_val:.2f} (PSR {psr_val:.0f}%), Sortino = {sortino_val:.2f} • Avg Daily Value = {adv_val:,.1f} ล้านบาท
</p>
<div style="font-size:12px; color:#F59E0B; background:rgba(245,158,11,0.08); border-left:3px solid #F59E0B; padding:8px 10px; border-radius:4px; line-height:1.5;">
<b>หมายเหตุสำคัญ:</b> ตัวเลขทั้งหมดใน Module นี้เป็นเครื่องมือ risk screening ไม่ใช่ investment advisory signal ควรใช้ประกอบกับ fundamental analysis, valuation และดุลยพินิจของนักวิเคราะห์เสมอ — ไม่ควรใช้ตัดสินใจซื้อ-ขายจากตัวเลขกลุ่มนี้เพียงอย่างเดียว
</div></div>""", unsafe_allow_html=True)

    render_nav_footer("m5", prev_page=" 🔮 AI Prediction", next_page=" 📊 Industry Benchmark")

# ============================================================================
# PAGE 7: MODULE 6 - INDUSTRY BENCHMARK
# ============================================================================
elif "Industry Benchmark" in nav_page:
    n_sector = len(sector_peers)
    sector_rank = int(stock_info.get('sector_rank', 1))
    overall_rank = int(stock_info.get('overall_rank', 1))
    n_all = len(scores_df)
    pct_in_sector = round((1 - (sector_rank - 1) / max(n_sector, 1)) * 100)

    st.markdown(f"""
    <div style="display:flex; justify-content:space-between; align-items:flex-end; margin-bottom:15px; border-bottom:1px solid #1E293B; padding-bottom:10px;">
        <div><div style="font-size:14.5px; color:#64748B; margin-bottom:2px;">Home / Module 6 / Industry Benchmark</div>
        <div style="display:flex; align-items:baseline; gap:10px;"><h2 style="margin:0; color:#F8FAFC; font-size:23px;">INDUSTRY BENCHMARK</h2>
        <span style="font-size:18.5px; color:#A855F7; font-weight:bold;">{selected_ticker} ⭐</span>
        <span style="font-size:15px; color:#64748B;">{stock_info.get('sector','-')}</span></div></div>
        <div style="text-align:right; display:flex; gap:20px;">
        <div><span style="font-size:13.5px; color:#64748B;">Current Price</span><br><b style="color:{change_color}; font-size:16.5px;">{current_price:.2f} THB</b> <span style="color:{change_color}; font-size:13.5px;">({change_sign}{change_pct:.2f}%) {arrow_sign}</span></div>
        <div><span style="font-size:13.5px; color:#64748B;">Sector</span><br><b style="color:#CBD5E1; font-size:15px;">{stock_info.get('sector','-')}</b></div>
        <div><span style="font-size:13.5px; color:#64748B;">Universe</span><br><b style="color:#CBD5E1; font-size:15px;">{n_all} หุ้นที่ติดตาม (2023-2025)</b></div>
        </div>
    </div>""", unsafe_allow_html=True)

    r1_c1, r1_c2, r1_c3 = st.columns([1.1, 0.8, 2.1])

    with r1_c1:
        position_label = "INDUSTRY LEADER" if sector_rank == 1 else ("STRONG COMPETITOR" if sector_rank <= max(2, n_sector // 2) else "LAGGING PEER")
        pos_stars = 5 if sector_rank == 1 else (4 if sector_rank <= max(2, n_sector // 2) else 2)
        st.markdown(f"""<div style="background-color:#151E2F; border:1px solid #1E293B; border-radius:8px; padding:14px; height:185px;">
<div style="font-size:14.5px; color:#94A3B8; font-weight:bold; margin-bottom:8px;">STRATEGIC INVESTMENT POSITION</div>
<div style="display:flex; align-items:center; gap:10px; margin-bottom:6px;">
<div style="background:rgba(168,85,247,0.15); border:1px solid #A855F7; border-radius:50%; width:50px; height:50px; display:flex; align-items:center; justify-content:center; font-size:21px;">🏆</div>
<div><div style="color:#C084FC; font-size:17.5px; font-weight:bold;">{position_label}</div><div style="color:#A855F7; font-size:15px; letter-spacing:2px;">{'★'*pos_stars}{'☆'*(5-pos_stars)}</div></div>
<div style="margin-left:auto;"><span style="background-color:rgba(168,85,247,0.2); color:#C084FC; font-size:13.5px; font-weight:bold; padding:2px 6px; border-radius:4px;">Rank {sector_rank}/{n_sector}</span></div>
</div><p style="color:#94A3B8; font-size:13.5px; line-height:1.4; margin:0;">อันดับที่ {sector_rank} จาก {n_sector} บริษัทในกลุ่ม {stock_info.get('sector','-')} จาก Overall Score = {safe(stock_info.get('overall_score')):.1f}/100</p>
</div>""", unsafe_allow_html=True)

    with r1_c2:
        st.markdown(f"""<div style="background-color:#151E2F; border:1px solid #1E293B; border-radius:8px; padding:14px; height:185px; text-align:center;">
<div style="font-size:14.5px; color:#94A3B8; font-weight:bold; margin-bottom:4px;">SECTOR RANKING</div>
<div style="font-size:13.5px; color:#64748B;">{stock_info.get('sector','-')}</div>
<div style="margin:4px 0;"><div style="font-size:13.5px; color:#F59E0B;">Rank</div>
<div style="font-size:28px; color:#F8FAFC; font-weight:bold; line-height:1;">{sector_rank}</div><div style="font-size:12.5px; color:#64748B;">/ {n_sector} หุ้นในกลุ่ม</div></div>
<span style="background-color:rgba(245,158,11,0.15); color:#F59E0B; font-size:13.5px; font-weight:bold; padding:2px 8px; border-radius:8px;">Top {pct_in_sector}%</span>
</div>""", unsafe_allow_html=True)

    with r1_c3:
        def pct_rank(col, ascending=False):
            r = scores_df[col].rank(ascending=ascending, pct=True)
            v = r[scores_df['ticker'] == selected_ticker].values[0] if selected_ticker in scores_df['ticker'].values else 0.5
            return round((1 - v) * 100) if not ascending else round(v * 100)

        percentile_dims = [
            ("Profitability", int(round(100 - scores_df['health_score'].rank(pct=True)[scores_df['ticker'] == selected_ticker].values[0] * 100)), "#10B981"),
            ("Growth", int(round(100 - scores_df['revenue_growth_yoy'].rank(pct=True)[scores_df['ticker'] == selected_ticker].values[0] * 100)) if stock_info.get('revenue_growth_yoy') is not None else 50, "#3B82F6"),
            ("Valuation", int(round(100 - scores_df['valuation_score'].rank(pct=True)[scores_df['ticker'] == selected_ticker].values[0] * 100)), "#F59E0B"),
            ("Entry Timing", int(round(100 - scores_df['timing_score'].rank(pct=True)[scores_df['ticker'] == selected_ticker].values[0] * 100)), "#10B981"),
            ("Risk (safer)", int(round(100 - scores_df['risk_score'].rank(pct=True)[scores_df['ticker'] == selected_ticker].values[0] * 100)), "#F59E0B"),
            ("AI Prediction", int(round(100 - scores_df['ai_score'].rank(pct=True)[scores_df['ticker'] == selected_ticker].values[0] * 100)), "#A855F7"),
        ]

        def dim_pct_card(label, pct, color):
            tier = "Excellent" if pct <= 20 else ("Good" if pct <= 45 else ("Fair" if pct <= 70 else "Weak"))
            return f"""<div style="background:#0F172A; padding:6px 2px; border-radius:6px; border:1px solid #1E293B;">
<div style="color:#94A3B8; font-size:13px;">{label}</div><div style="color:{color}; font-size:15px; font-weight:bold; margin:2px 0;">Top {max(pct,1)}%</div>
<div style="color:{color}; font-size:12.5px;">{tier}</div></div>"""

        st.markdown(f"""<div style="background-color:#151E2F; border:1px solid #1E293B; border-radius:8px; padding:14px; height:185px;">
<div style="font-size:14.5px; color:#94A3B8; font-weight:bold; margin-bottom:8px;">DIMENSION PERCENTILE RANK (vs. {n_all} หุ้นที่ติดตาม)</div>
<div style="display:grid; grid-template-columns: repeat(6, 1fr); gap:6px; text-align:center;">
{''.join([dim_pct_card(l, p, c) for l, p, c in percentile_dims])}
</div></div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
    r2_c1, r2_c2, r2_c3 = st.columns([1.6, 1.1, 1.3])

    with r2_c1:
        peers_sorted = sector_peers.sort_values('overall_score', ascending=False)

        def badge(val, thresholds, labels, colors):
            for th, lab, col in zip(thresholds, labels, colors):
                if val >= th:
                    return f'<span style="color:{col}; font-weight:bold;">{lab}</span>'
            return f'<span style="color:{colors[-1]};">{labels[-1]}</span>'

        rows_html = ""
        for _, p in peers_sorted.iterrows():
            is_sel = p['ticker'] == selected_ticker
            row_bg = "background:rgba(168,85,247,0.08);" if is_sel else ""
            star_n = min(5, max(1, round(safe(p['overall_score']) / 20)))
            health_b = badge(p['health_score'], [70, 45, 0], ["Excellent", "Good", "Weak"], ["#10B981", "#3B82F6", "#EF4444"])
            val_b = "Undervalued" if p['margin_of_safety'] > 10 else ("Overvalued" if p['margin_of_safety'] < -10 else "Fair Value")
            val_c = "#10B981" if p['margin_of_safety'] > 10 else ("#EF4444" if p['margin_of_safety'] < -10 else "#94A3B8")
            timing_b = badge(p['timing_score'], [65, 45, 0], ["Good Entry", "Neutral", "Bad Entry"], ["#10B981", "#F59E0B", "#EF4444"])
            ai_b = badge(p['ai_score'], [65, 45, 0], ["Bullish", "Neutral", "Bearish"], ["#10B981", "#94A3B8", "#EF4444"])
            risk_b = "Low" if p['risk_score'] >= 65 else ("Medium" if p['risk_score'] >= 40 else "High")
            risk_c = "#10B981" if p['risk_score'] >= 65 else ("#F59E0B" if p['risk_score'] >= 40 else "#EF4444")
            name_disp = f"⭐ {p['ticker']}" if is_sel else p['ticker']
            name_c = "#C084FC" if is_sel else "#F8FAFC"
            rows_html += f"""<tr style="border-bottom:1px solid #1E293B; {row_bg}">
<td style="text-align:left; padding:6px 0; color:{name_c}; font-weight:bold;">{name_disp}</td>
<td>{health_b}</td><td><span style="color:{val_c};">{val_b}</span></td><td>{timing_b}</td><td>{ai_b}</td>
<td><span style="color:{risk_c};">{risk_b}</span></td><td style="color:#A855F7; letter-spacing:1px;">{'★'*star_n}{'☆'*(5-star_n)}</td></tr>"""

        st.markdown(f"""<div style="background-color:#151E2F; border:1px solid #1E293B; border-radius:8px; padding:14px; height:350px;">
<div style="font-size:14.5px; color:#94A3B8; font-weight:bold; margin-bottom:6px;">PEER COMPARISON — {stock_info.get('sector','-')} ({n_sector} หุ้น)</div>
<table style="width:100%; text-align:center; font-size:14px; color:#CBD5E1; border-collapse:collapse;">
<tr style="border-bottom:1px solid #1E293B; color:#64748B; font-size:13px;"><th style="text-align:left; padding:5px 0;">Company</th><th>Health</th><th>Fair Value</th><th>Entry Timing</th><th>AI Prediction</th><th>Risk</th><th>Overall</th></tr>
{rows_html}
</table>
<div style="font-size:12.5px; color:#64748B; margin-top:6px;">*จัดอันดับจาก Overall Score ที่คำนวณจริงจากข้อมูลใน cis_summary_scores</div>
</div>""", unsafe_allow_html=True)

    with r2_c2:
        cats = ['Health', 'Valuation', 'Timing', 'AI Pred.', 'Risk', 'Industry']
        stock_vals = [safe(stock_info.get('health_score')), safe(stock_info.get('valuation_score')), safe(stock_info.get('timing_score')),
                      safe(stock_info.get('ai_score')), safe(stock_info.get('risk_score')), safe(stock_info.get('industry_score'))]
        sector_avg_vals = [sector_peers['health_score'].mean(), sector_peers['valuation_score'].mean(), sector_peers['timing_score'].mean(),
                            sector_peers['ai_score'].mean(), sector_peers['risk_score'].mean(), sector_peers['industry_score'].mean()]
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(r=stock_vals, theta=cats, fill='toself', fillcolor='rgba(168,85,247,0.3)', line=dict(color='#A855F7', width=2), name=selected_ticker))
        fig_radar.add_trace(go.Scatterpolar(r=sector_avg_vals, theta=cats, line=dict(color='#64748B', width=1.5, dash='dash'), name='Sector Avg'))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100], showticklabels=False, linecolor="#1E293B", gridcolor="#1E293B"),
                       angularaxis=dict(linecolor="#1E293B", gridcolor="#1E293B", tickfont=dict(size=11.5, color="#94A3B8"))),
            paper_bgcolor="#151E2F", plot_bgcolor="#151E2F", height=310, margin=dict(l=25, r=25, t=30, b=15),
            title=dict(text="RADAR: STOCK vs SECTOR AVG", font=dict(size=13.5, color="#94A3B8"), x=0.05, y=0.98),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11.5, color="#CBD5E1"))
        )
        show_chart(fig_radar, key="industry_radar", expand_height=650)

    with r2_c3:
        matrix_df = scores_df[['ticker', 'health_score', 'overall_score']].copy()
        matrix_df.columns = ['Company', 'Business_Quality', 'Investment_Attract']
        color_map = {t: ('#C084FC' if t == selected_ticker else '#38BDF8') for t in matrix_df['Company']}
        fig_matrix = px.scatter(matrix_df, x='Business_Quality', y='Investment_Attract', text='Company', color='Company', color_discrete_map=color_map)
        fig_matrix.update_traces(textposition='top center', marker=dict(size=13, line=dict(width=1, color='white')))
        fig_matrix.add_hline(y=50, line_width=1, line_dash="dash", line_color="#334155")
        fig_matrix.add_vline(x=50, line_width=1, line_dash="dash", line_color="#334155")
        fig_matrix.add_annotation(x=25, y=95, text="💎 Hidden Gem", showarrow=False, font=dict(size=11.5, color="#34D399"))
        fig_matrix.add_annotation(x=80, y=95, text="🏆 Market Leader", showarrow=False, font=dict(size=11.5, color="#C084FC"))
        fig_matrix.add_annotation(x=25, y=10, text="⚠️ Value Trap", showarrow=False, font=dict(size=11.5, color="#F87171"))
        fig_matrix.add_annotation(x=80, y=10, text="⭐ Competitive", showarrow=False, font=dict(size=11, color="#FBBF24"))
        fig_matrix.update_layout(
            paper_bgcolor="#151E2F", plot_bgcolor="#0F172A", height=310, margin=dict(l=15, r=15, t=30, b=15),
            title=dict(text="STRATEGIC MATRIX (All 8 Stocks)", font=dict(size=13.5, color="#94A3B8"), x=0.05, y=0.98),
            xaxis=dict(title=dict(text="Business Quality (Health Score) →", font=dict(size=11.5, color="#64748B")), range=[0, 100], showgrid=False, showticklabels=False),
            yaxis=dict(title=dict(text="Investment Attractiveness (Overall) →", font=dict(size=11.5, color="#64748B")), range=[0, 100], showgrid=False, showticklabels=False),
            showlegend=False
        )
        show_chart(fig_matrix, key="industry_matrix", expand_height=650)

    st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
    r3_c1, r3_c2, r3_c3 = st.columns([1.3, 1.5, 1.2])

    with r3_c1:
        strengths_ib, weaknesses_ib = [], []
        if stock_info['health_score'] > sector_peers['health_score'].mean(): strengths_ib.append("Health Score สูงกว่าค่าเฉลี่ยกลุ่ม")
        if stock_info['ai_score'] > sector_peers['ai_score'].mean(): strengths_ib.append("AI Prediction Score สูงกว่าค่าเฉลี่ยกลุ่ม")
        if safe(stock_info.get('revenue_growth_yoy')) > 0: strengths_ib.append(f"รายได้เติบโต {safe(stock_info.get('revenue_growth_yoy')):.1f}% YoY")
        if not strengths_ib: strengths_ib.append("ผลประกอบการยังอยู่ระหว่างพัฒนาเทียบกลุ่ม")
        if stock_info['valuation_score'] < sector_peers['valuation_score'].mean(): weaknesses_ib.append("Valuation แพงกว่าค่าเฉลี่ยกลุ่ม")
        if stock_info['risk_score'] < sector_peers['risk_score'].mean(): weaknesses_ib.append("ความเสี่ยง (Volatility/Drawdown) สูงกว่าค่าเฉลี่ยกลุ่ม")
        if not weaknesses_ib: weaknesses_ib.append("ไม่พบจุดอ่อนเชิงเปรียบเทียบที่ชัดเจนกับกลุ่ม")

        st.markdown(f"""<div style="background-color:#151E2F; border:1px solid #1E293B; border-radius:8px; padding:14px; height:275px;">
<div style="font-size:14.5px; color:#94A3B8; font-weight:bold; margin-bottom:6px;">COMPETITIVE ADVANTAGE</div>
<div style="font-size:13px; color:#10B981; font-weight:bold; margin-bottom:2px;">STRENGTHS</div>
<ul style="color:#CBD5E1; font-size:13.5px; line-height:1.4; padding-left:14px; margin:0 0 6px 0;">{''.join([f'<li>{s}</li>' for s in strengths_ib])}</ul>
<div style="font-size:13px; color:#EF4444; font-weight:bold; margin-bottom:2px;">WEAKNESSES / RISKS</div>
<ul style="color:#CBD5E1; font-size:13.5px; line-height:1.4; padding-left:14px; margin:0;">{''.join([f'<li>{w}</li>' for w in weaknesses_ib])}</ul>
</div>""", unsafe_allow_html=True)

    with r3_c2:
        st.markdown(f"""<div style="background-color:#151E2F; border:1px solid #1E293B; border-radius:8px; padding:14px; height:275px;">
<div style="font-size:14.5px; color:#94A3B8; font-weight:bold; margin-bottom:6px;">EXPLAINABLE AI SUMMARY</div>
<p style="color:#CBD5E1; font-size:14px; line-height:1.4; margin:0 0 8px 0;">
<b>{selected_ticker}</b> อยู่อันดับที่ <b>{sector_rank}</b> จาก {n_sector} บริษัทในกลุ่ม {stock_info.get('sector','-')} (Overall Score {safe(stock_info.get('overall_score')):.1f}/100) เมื่อเทียบกับบริษัทในกลุ่มเดียวกัน:</p>
<div style="color:#CBD5E1; font-size:13.5px; line-height:1.5;">
<div><span style="color:#10B981;">✔</span> Health Score: {safe(stock_info.get('health_score')):.1f} (Sector avg {sector_peers['health_score'].mean():.1f})</div>
<div><span style="color:#10B981;">✔</span> Valuation Score: {safe(stock_info.get('valuation_score')):.1f} (Sector avg {sector_peers['valuation_score'].mean():.1f})</div>
<div><span style="color:#10B981;">✔</span> AI Prediction Score: {safe(stock_info.get('ai_score')):.1f} (Sector avg {sector_peers['ai_score'].mean():.1f})</div>
<div><span style="color:#10B981;">✔</span> Risk Score: {safe(stock_info.get('risk_score')):.1f} (Sector avg {sector_peers['risk_score'].mean():.1f})</div>
</div></div>""", unsafe_allow_html=True)

    with r3_c3:
        rec = stock_info.get('recommendation', 'ACCUMULATE')
        rec_color2 = {"STRONG BUY": "#10B981", "BUY": "#10B981", "ACCUMULATE": "#84CC16", "REDUCE / SELL": "#EF4444"}.get(rec, "#F59E0B")
        conf_lvl = "High" if abs(safe(stock_info.get('margin_of_safety'))) > 15 else "Medium"
        st.markdown(f"""<div style="background-color:#151E2F; border:1px solid #1E293B; border-radius:8px; padding:14px; height:275px; text-align:center;">
<div style="font-size:14.5px; color:#94A3B8; font-weight:bold; margin-bottom:4px; text-align:left;">FINAL RECOMMENDATION</div>
<div style="display:flex; justify-content:center; align-items:center; gap:8px; margin:4px 0;">
<div><h1 style="color:{rec_color2}; margin:0; font-size:24px; line-height:1.1;">{rec}</h1></div></div>
<div style="text-align:left; font-size:13.5px; margin-top:8px; border-top:1px dashed #334155; padding-top:6px;">
<div style="display:flex; justify-content:space-between; margin-bottom:3px;"><span style="color:#94A3B8;">Confidence Level</span><span style="color:{rec_color2}; font-weight:bold;">{conf_lvl}</span></div>
<div style="display:flex; justify-content:space-between; margin-bottom:3px;"><span style="color:#94A3B8;">Overall Score</span><span style="color:#F59E0B; font-weight:bold;">{safe(stock_info.get('overall_score')):.1f}/100</span></div>
<div style="display:flex; justify-content:space-between; margin-bottom:3px;"><span style="color:#94A3B8;">Sector Rank</span><span style="color:#F59E0B; font-weight:bold;">{sector_rank} / {n_sector}</span></div>
</div></div>""", unsafe_allow_html=True)

    import base64
    csv_text = scores_df[scores_df['ticker'] == selected_ticker].to_csv(index=False)
    b64_csv = base64.b64encode(csv_text.encode('utf-8-sig')).decode()
    download_link = f'data:file/csv;base64,{b64_csv}'

    st.markdown(f"""<div style="display:flex; justify-content:space-between; align-items:center; background-color:#151E2F; border:1px solid #1E293B; border-radius:8px; padding:10px 14px; margin-top:12px;">
<div style="font-size:14px; color:#CBD5E1;"><b>EXPORT & INDUSTRY DATA</b><br><span style="color:#94A3B8; font-size:13px;">ดาวน์โหลดคะแนนวิเคราะห์ทั้งหมดของ {selected_ticker} (ข้อมูลจริงจาก cis_summary_scores)</span></div>
<a href="{download_link}" download="{selected_ticker}_CIS_Analysis.csv" style="background:#3B82F6; color:white; border:none; padding:6px 14px; border-radius:6px; font-size:13px; text-decoration:none; display:inline-block; font-weight:bold; cursor:pointer;">📥 Export Data (CSV)</a>
</div>""", unsafe_allow_html=True)

    render_nav_footer("m6", prev_page=" 🛡️ Risk Analysis", next_page=None)
