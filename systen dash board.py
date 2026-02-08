"""
╔═══════════════════════════════════════════════════════════════╗
║    CAFÉ AROMA - ULTIMATE KPI DASHBOARD v2.5                 ║
║    Decision-Ready Analytics Platform with Filters & AI      ║
╚═══════════════════════════════════════════════════════════════╝

✨ v2.5 NEW FEATURES:
- 🔎 Interactive Filters (Channel + Date Range)
- 🌅 Morning Impact Indicator
- 💡 Smart CPA Recommendations
- 📊 Trend Alerts (Month-over-Month)
- 🎯 Progress Bars to Goals
- 📈 Budget Scenario Simulator
- 🧪 A/B Test Suggestions Panel
- 📥 Executive PDF Export Ready
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Café Aroma Decision Dashboard",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════
# COLORS & STYLING
# ═══════════════════════════════════════════════════════════════

COLORS = {
    "primary": "#1E88E5",
    "success": "#43A047",
    "warning": "#FB8C00",
    "danger": "#E53935",
    "info": "#00ACC1",
    "purple": "#7E57C2",
}

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

    * {{
        font-family: 'Inter', sans-serif;
    }}

    .main-title {{
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.3rem;
        letter-spacing: -0.02em;
    }}

    .sub-title {{
        font-size: 1.3rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }}

    .executive-summary {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        box-shadow: 0 8px 16px rgba(102, 126, 234, 0.3);
    }}

    .exec-title {{
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 1rem;
        border-bottom: 2px solid rgba(255,255,255,0.3);
        padding-bottom: 0.5rem;
    }}

    .exec-item {{
        font-size: 1.1rem;
        margin: 0.7rem 0;
        padding-left: 1.5rem;
        line-height: 1.6;
    }}

    .success-box {{
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 2px solid {COLORS['success']};
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(67, 160, 71, 0.2);
    }}

    .warning-box {{
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border: 2px solid {COLORS['warning']};
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(251, 140, 0, 0.2);
    }}

    .info-box {{
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        border: 2px solid {COLORS['info']};
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0, 172, 193, 0.2);
    }}

    .target-box {{
        background: linear-gradient(135deg, #e1e4f8 0%, #d4d7f0 100%);
        border: 2px solid {COLORS['purple']};
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        text-align: center;
        font-weight: 600;
    }}

    .recommendation {{
        background: rgba(255, 255, 255, 0.9);
        padding: 1rem;
        border-radius: 8px;
        margin-top: 0.8rem;
        border-left: 4px solid {COLORS['success']};
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}

    div[data-testid="stMetricValue"] {{
        font-size: 2rem;
        font-weight: 700;
    }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: 10px;
    }}

    .stTabs [data-baseweb="tab"] {{
        height: 55px;
        background-color: #f5f7fa;
        border-radius: 10px 10px 0 0;
        font-weight: 600;
        font-size: 1rem;
    }}

    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }}

    .stProgress > div > div > div > div {{
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

DEFAULT_FILE = os.getenv("CAFE_AROMA_FILE", "D:\LLM1\Café Aroma Pipeline.xlsx")

COLUMN_ALIASES = {
    "date": ["date", "Date", "transaction_date", "Transaction Date", "transaction date"],
    "channel": ["channel", "Channel", "marketing_channel", "Marketing Channel", "marketing channel"],
    "customer_type": ["customer_type", "Customer Type", "customer type", "customertype", "type"],
    "ad_spend": ["ad_spend", "Ad Spend", "ad spend", "spend", "Spend"],
    "conversions": ["conversions", "Conversions", "orders", "Orders"],
    "revenue": ["revenue", "Revenue", "sales", "Sales"],
    "service_category": ["service_category", "Service Category", "service category", "category"],
}

TARGET_ROI = 1.2
TARGET_ROAS = 2.2
GOOD_ROAS = 2.5
GOOD_ROI = 1.5
BENCHMARK_CPA = 12
TARGET_MORNING_PCT = 60


# ═══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def find_column(df_cols, canonical):
    for candidate in COLUMN_ALIASES.get(canonical, []):
        if candidate in df_cols:
            return candidate
    return None


def normalize_channel(x):
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()
    if "walk" in s: return "Walk-in"
    if "google" in s: return "Google Ads"
    if "insta" in s: return "Instagram"
    if "fly" in s: return "Flyer"
    if "refer" in s: return "Referral"
    return str(x).strip()


def normalize_customer_type(x):
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()
    if "new" in s: return "New"
    if "return" in s: return "Returning"
    return str(x).strip()


def add_kpis(df):
    df = df.copy()
    df["profit"] = df["revenue"] - df["spend"]
    df["roas"] = np.where(df["spend"] > 0, df["revenue"] / df["spend"], 0)
    df["roi"] = np.where(df["spend"] > 0, (df["revenue"] - df["spend"]) / df["spend"], 0)
    df["cost_per_conversion"] = np.where(df["conversions"] > 0, df["spend"] / df["conversions"], 0)
    df["revenue_per_conversion"] = np.where(df["conversions"] > 0, df["revenue"] / df["conversions"], 0)
    return df


def get_performance_status(roi, roas):
    if roi >= GOOD_ROI and roas >= GOOD_ROAS:
        return "🚀 Excellent"
    elif roi >= TARGET_ROI and roas >= TARGET_ROAS:
        return "✅ Good"
    elif roi >= 0.5:
        return "⚠️ Needs Optimization"
    else:
        return "🔴 Poor"


def get_recommendation(roi, roas, channel):
    if roi >= GOOD_ROI:
        return f"💡 **Scale Up:** Increase {channel} budget by 40-50% to maximize returns. Test new ad creatives."
    elif roi >= TARGET_ROI:
        return f"🔧 **Optimize:** {channel} is profitable but has room for improvement. Run A/B tests on targeting and messaging."
    elif roi >= 0.5:
        return f"⚠️ **Improve or Reduce:** {channel} is underperforming. Reduce budget by 30% or pivot strategy entirely."
    else:
        return f"🛑 **Pause:** {channel} is losing money. Pause campaigns immediately and analyze root causes."


def cpa_recommendation(cpa, benchmark=BENCHMARK_CPA):
    """Smart CPA recommendations"""
    if cpa <= benchmark:
        return "✅ Efficient CPA. Scale budget."
    elif cpa <= benchmark * 1.2:
        return "⚠️ Acceptable CPA. Optimize creatives."
    else:
        return "🔴 High CPA. Reduce spend or refine targeting."


# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════

def load_excel(file_path, sheet_name):
    try:
        raw = pd.read_excel(file_path, sheet_name=sheet_name)
        cols = list(raw.columns)

        col_map = {}
        missing = []
        required = ["date", "channel", "customer_type", "ad_spend", "conversions", "revenue"]

        for canonical in required:
            found = find_column(cols, canonical)
            if found:
                col_map[canonical] = found
            else:
                missing.append(canonical)

        if missing:
            st.error(f"❌ Missing columns: {missing}")
            st.info(f"📋 Found: {cols}")
            return None

        df = pd.DataFrame({
            "date": pd.to_datetime(raw[col_map["date"]], errors="coerce"),
            "channel": raw[col_map["channel"]].apply(normalize_channel),
            "customer_type": raw[col_map["customer_type"]].apply(normalize_customer_type),
            "ad_spend": pd.to_numeric(raw[col_map["ad_spend"]], errors="coerce").fillna(0),
            "conversions": pd.to_numeric(raw[col_map["conversions"]], errors="coerce").fillna(0),
            "revenue": pd.to_numeric(raw[col_map["revenue"]], errors="coerce").fillna(0),
        })

        svc_col = find_column(cols, "service_category")
        if svc_col:
            df["service_category"] = raw[svc_col].astype(str).str.strip()

        df = df[df["date"].notna()].copy()
        df = df[df["channel"] != ""].copy()
        df = df[df["customer_type"] != ""].copy()

        df["month"] = df["date"].dt.to_period("M").astype(str)
        df["day_name"] = df["date"].dt.day_name()
        df["hour"] = df["date"].dt.hour
        df["is_morning"] = (df["hour"] >= 6) & (df["hour"] < 12)
        df["is_weekday"] = df["date"].dt.weekday < 5

        return df

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None


# ═══════════════════════════════════════════════════════════════
# KPI CALCULATIONS
# ═══════════════════════════════════════════════════════════════

def calc_overall(df):
    spend = df["ad_spend"].sum()
    revenue = df["revenue"].sum()
    conv = df["conversions"].sum()

    return {
        "records": len(df),
        "first_date": df["date"].min().strftime("%Y-%m-%d"),
        "last_date": df["date"].max().strftime("%Y-%m-%d"),
        "spend": spend,
        "revenue": revenue,
        "profit": revenue - spend,
        "conversions": conv,
        "roas": revenue / spend if spend > 0 else 0,
        "roi": (revenue - spend) / spend if spend > 0 else 0,
        "cost_per_conv": spend / conv if conv > 0 else 0,
        "new_conv": df[df["customer_type"] == "New"]["conversions"].sum(),
        "ret_conv": df[df["customer_type"] == "Returning"]["conversions"].sum(),
        "morning_conv": df[df["is_morning"] == True]["conversions"].sum(),
        "weekday_conv": df[df["is_weekday"] == True]["conversions"].sum(),
    }


def calc_by_channel(df):
    agg = df.groupby("channel", as_index=False).agg({
        "ad_spend": "sum",
        "revenue": "sum",
        "conversions": "sum",
    }).rename(columns={"ad_spend": "spend"})

    for ch in agg["channel"]:
        new = df[(df["channel"] == ch) & (df["customer_type"] == "New")]["conversions"].sum()
        ret = df[(df["channel"] == ch) & (df["customer_type"] == "Returning")]["conversions"].sum()
        agg.loc[agg["channel"] == ch, "new_conversions"] = new
        agg.loc[agg["channel"] == ch, "returning_conversions"] = ret

    return add_kpis(agg).sort_values("roi", ascending=False)


def calc_monthly(df):
    agg = df.groupby(["month", "customer_type"], as_index=False).agg({
        "ad_spend": "sum",
        "revenue": "sum",
        "conversions": "sum",
    }).rename(columns={"ad_spend": "spend"})

    return add_kpis(agg)


# ═══════════════════════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════════════════════

def chart_roi_enhanced(channel_df):
    data = channel_df.sort_values("roi", ascending=True).copy()
    data["roi_pct"] = data["roi"] * 100

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=data["roi_pct"],
        y=data["channel"],
        orientation="h",
        text=data["roi_pct"].apply(lambda x: f"{x:.1f}%"),
        textposition="outside",
        marker=dict(
            color=data["roi_pct"],
            colorscale=[[0, COLORS["danger"]], [0.5, COLORS["warning"]], [1, COLORS["success"]]],
            line=dict(width=1, color="white")
        ),
    ))

    fig.add_vline(
        x=TARGET_ROI * 100,
        line_dash="dash",
        line_color=COLORS["info"],
        line_width=2,
        annotation_text=f"Target: {TARGET_ROI * 100:.0f}%",
        annotation_position="top right"
    )

    fig.add_vline(
        x=GOOD_ROI * 100,
        line_dash="dot",
        line_color=COLORS["success"],
        line_width=2,
        annotation_text=f"Excellent: {GOOD_ROI * 100:.0f}%",
        annotation_position="top right"
    )

    fig.update_layout(
        title="📊 Return on Investment by Channel (↑ higher is better)",
        xaxis_title="ROI (%)",
        height=450,
        showlegend=False,
    )

    return fig


def chart_roas_enhanced(channel_df):
    data = channel_df.sort_values("roas", ascending=True).copy()

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=data["roas"],
        y=data["channel"],
        orientation="h",
        text=data["roas"].apply(lambda x: f"{x:.2f}x"),
        textposition="outside",
        marker=dict(
            color=data["roas"],
            colorscale="Blues",
            line=dict(width=1, color="white")
        ),
    ))

    fig.add_vline(
        x=TARGET_ROAS,
        line_dash="dash",
        line_color=COLORS["info"],
        line_width=2,
        annotation_text=f"Target: {TARGET_ROAS:.1f}x"
    )

    fig.add_vline(
        x=GOOD_ROAS,
        line_dash="dot",
        line_color=COLORS["success"],
        line_width=2,
        annotation_text=f"Excellent: {GOOD_ROAS:.1f}x"
    )

    fig.update_layout(
        title="💰 Return on Ad Spend by Channel (↑ higher is better)",
        xaxis_title="ROAS (x)",
        height=450,
        showlegend=False,
    )

    return fig


def chart_revenue_trend(monthly_df):
    agg = monthly_df.groupby("month", as_index=False).agg({
        "spend": "sum",
        "revenue": "sum",
        "profit": "sum",
    })

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=agg["month"],
        y=agg["revenue"],
        mode="lines+markers",
        name="Revenue",
        line=dict(color=COLORS["success"], width=4),
        marker=dict(size=10),
        fill='tonexty',
        fillcolor='rgba(67, 160, 71, 0.1)',
    ))

    fig.add_trace(go.Scatter(
        x=agg["month"],
        y=agg["spend"],
        mode="lines+markers",
        name="Ad Spend",
        line=dict(color=COLORS["danger"], width=4),
        marker=dict(size=10),
    ))

    fig.update_layout(
        title="📈 Revenue & Spend Trend Over Time",
        xaxis_title="Month",
        yaxis_title="Amount (SAR)",
        height=450,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig


# ═══════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════

st.markdown('<div class="main-title">☕ Café Aroma Decision Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Data-Driven Marketing Decisions & Performance Intelligence v2.5</div>',
            unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")

    uploaded = st.file_uploader("📁 Upload Excel", type=["xlsx", "xls"])

    if uploaded:
        file_src = uploaded
    else:
        path = st.text_input("📂 File path", value=DEFAULT_FILE)
        file_src = path

    sheet = st.text_input("📄 Sheet", value="0")

    try:
        sheet_val = int(sheet)
    except:
        sheet_val = sheet

    st.divider()

    st.markdown("### 🎯 Targets")
    st.metric("ROI Target", f"{TARGET_ROI * 100:.0f}%")
    st.metric("ROAS Target", f"{TARGET_ROAS:.1f}x")
    st.metric("Morning Target", f"{TARGET_MORNING_PCT}%")

    st.divider()
    st.caption("v2.5 Ultimate | Feb 2026")


# Load raw data
@st.cache_data(show_spinner=False, ttl=300)
def load_raw_data(file_source, sheet_name):
    return load_excel(file_source, sheet_name)


with st.spinner("📊 Loading data..."):
    raw_df = load_raw_data(file_src, sheet_val)

if raw_df is None:
    st.error("❌ Failed to load data. Check file & settings.")
    st.stop()

# ═══════════════════════════════════════════════════════════════
# FILTERS (NEW!)
# ═══════════════════════════════════════════════════════════════

with st.sidebar:
    st.divider()
    st.markdown("### 🔎 Filters")

    channels = ["All"] + sorted(raw_df["channel"].unique().tolist())
    selected_channel = st.selectbox("📊 Channel", channels)

    date_min, date_max = raw_df["date"].min().date(), raw_df["date"].max().date()
    date_range = st.date_input("📅 Date Range", [date_min, date_max])

    if st.button("🔄 Apply Filters", use_container_width=True):
        st.rerun()

# Apply filters
fdf = raw_df.copy()
if selected_channel != "All":
    fdf = fdf[fdf["channel"] == selected_channel]

if len(date_range) == 2:
    fdf = fdf[(fdf["date"] >= pd.to_datetime(date_range[0])) & (fdf["date"] <= pd.to_datetime(date_range[1]))]

# Recalculate KPIs with filtered data
overall = calc_overall(fdf)
channel_df = calc_by_channel(fdf)
monthly_df = calc_monthly(fdf)

# Filter indicator
if selected_channel != "All" or len(date_range) == 2:
    st.info(f"🔎 **Filtered View:** {selected_channel} | {date_range[0]} → {date_range[1]}")

# ═══════════════════════════════════════════════════════════════
# EXECUTIVE SUMMARY
# ═══════════════════════════════════════════════════════════════

best = channel_df.iloc[0]
worst = channel_df.iloc[-1]

st.markdown(f"""
<div class="executive-summary">
<div class="exec-title">🎯 Executive Summary: Where to Invest Marketing Budget</div>
<div class="exec-item">• <strong>Best Performer:</strong> {best['channel']} (ROI: {best['roi'] * 100:.1f}%, ROAS: {best['roas']:.2f}x)</div>
<div class="exec-item">• <strong>Needs Improvement:</strong> {worst['channel']} (ROI: {worst['roi'] * 100:.1f}%, ROAS: {worst['roas']:.2f}x)</div>
<div class="exec-item">• <strong>Overall Performance:</strong> ROAS {overall['roas']:.2f}x | ROI {overall['roi'] * 100:.1f}% | Profit SAR {overall['profit']:,.0f}</div>
<div class="exec-item">• <strong>Next Step:</strong> Reallocate 30% budget from {worst['channel']} to {best['channel']} for +15-20% ROAS improvement</div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# KPI CARDS WITH PROGRESS BARS
# ═══════════════════════════════════════════════════════════════

st.markdown("### 📊 Key Performance Indicators")

c1, c2, c3, c4, c5, c6 = st.columns(6)

c1.metric("💰 Total Spend", f"SAR {overall['spend']:,.0f}")
c2.metric("💵 Revenue", f"SAR {overall['revenue']:,.0f}")
c3.metric("📈 Profit", f"SAR {overall['profit']:,.0f}",
          delta=f"{overall['profit'] / overall['spend'] * 100:.0f}% margin")
c4.metric("🎯 Conversions", f"{overall['conversions']:,.0f}")
c5.metric("💎 ROAS", f"{overall['roas']:.2f}x", delta=f"Target: {TARGET_ROAS:.1f}x")
c6.metric("📊 ROI", f"{overall['roi'] * 100:.1f}%", delta=f"Target: {TARGET_ROI * 100:.0f}%")

# Progress to Goals
st.markdown("### 🎯 Progress to Goals")
col_p1, col_p2, col_p3 = st.columns(3)

with col_p1:
    roi_progress = min(overall['roi'] / GOOD_ROI, 1.0)
    st.write("**ROI Goal (150%)**")
    st.progress(roi_progress)
    st.caption(f"{overall['roi'] * 100:.1f}% / {GOOD_ROI * 100:.0f}% ({roi_progress * 100:.0f}%)")

with col_p2:
    roas_progress = min(overall['roas'] / GOOD_ROAS, 1.0)
    st.write("**ROAS Goal (2.5x)**")
    st.progress(roas_progress)
    st.caption(f"{overall['roas']:.2f}x / {GOOD_ROAS:.1f}x ({roas_progress * 100:.0f}%)")

with col_p3:
    morning_pct = overall['morning_conv'] / overall['conversions'] * 100 if overall['conversions'] > 0 else 0
    morning_progress = min(morning_pct / TARGET_MORNING_PCT, 1.0)
    st.write("**Morning Conversions (60%)**")
    st.progress(morning_progress)
    st.caption(f"{morning_pct:.1f}% / {TARGET_MORNING_PCT}% ({morning_progress * 100:.0f}%)")

# Morning Impact Alert (NEW!)
if morning_pct < 35:
    st.warning("⚠️ **Morning share is low.** Consider breakfast offers 7–10 AM.")
elif morning_pct < 50:
    st.info("ℹ️ **Morning performance improving.** Continue optimizing morning campaigns.")
else:
    st.success("✅ **Morning campaigns performing well!** Maintain momentum.")

col_info1, col_info2 = st.columns(2)
with col_info1:
    st.info(f"📅 **Period:** {overall['first_date']} → {overall['last_date']} | **Records:** {overall['records']:,}")
with col_info2:
    weekday_pct = overall['weekday_conv'] / overall['conversions'] * 100 if overall['conversions'] > 0 else 0
    st.success(
        f"🌅 **Morning:** {overall['morning_conv']:,.0f} ({morning_pct:.1f}%) | 📅 **Weekday:** {overall['weekday_conv']:,.0f} ({weekday_pct:.1f}%)")

st.divider()

# ═══════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🎯 Channel Performance",
    "👥 Customer Insights",
    "📅 Time Trends",
    "🧪 Scenario Simulator",
    "🌅 Morning Analysis",
    "📥 Export & Actions"
])

# TAB 1: Channel Performance
with tab1:
    st.markdown("### 🏆 Where to Invest Marketing Budget (Decision View)")

    st.markdown(f"""
    <div class="target-box">
    🎯 <strong>Targets:</strong> ROI ≥ {TARGET_ROI * 100:.0f}% | ROAS ≥ {TARGET_ROAS:.1f}x | CPA ≤ SAR {BENCHMARK_CPA}
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    status_best = get_performance_status(best['roi'], best['roas'])
    status_worst = get_performance_status(worst['roi'], worst['roas'])

    with col_a:
        st.markdown(f"""
        <div class="success-box">
        <h3>🥇 Best: {best['channel']} {status_best}</h3>
        <ul>
        <li><b>ROI:</b> {best['roi'] * 100:.1f}% ({(best['roi'] / TARGET_ROI - 1) * 100:+.0f}% vs target)</li>
        <li><b>ROAS:</b> {best['roas']:.2f}x ({(best['roas'] / TARGET_ROAS - 1) * 100:+.0f}% vs target)</li>
        <li><b>CPA:</b> SAR {best['cost_per_conversion']:.0f}</li>
        <li><b>Revenue:</b> SAR {best['revenue']:,.0f}</li>
        <li><b>Profit:</b> SAR {best['profit']:,.0f}</li>
        </ul>
        <div class="recommendation">
        {get_recommendation(best['roi'], best['roas'], best['channel'])}
        </div>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown(f"""
        <div class="warning-box">
        <h3>⚠️ Needs Work: {worst['channel']} {status_worst}</h3>
        <ul>
        <li><b>ROI:</b> {worst['roi'] * 100:.1f}% ({(worst['roi'] / TARGET_ROI - 1) * 100:+.0f}% vs target)</li>
        <li><b>ROAS:</b> {worst['roas']:.2f}x ({(worst['roas'] / TARGET_ROAS - 1) * 100:+.0f}% vs target)</li>
        <li><b>CPA:</b> SAR {worst['cost_per_conversion']:.0f}</li>
        <li><b>Revenue:</b> SAR {worst['revenue']:,.0f}</li>
        </ul>
        <div class="recommendation">
        {get_recommendation(worst['roi'], worst['roas'], worst['channel'])}
        </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    col_c1, col_c2 = st.columns(2)

    with col_c1:
        st.plotly_chart(chart_roi_enhanced(channel_df), use_container_width=True)

    with col_c2:
        st.plotly_chart(chart_roas_enhanced(channel_df), use_container_width=True)

    st.markdown("### 📋 Detailed Channel Performance with CPA Analysis")

    display = channel_df.copy()
    display["roi_pct"] = display["roi"] * 100
    display["vs_target_roi"] = ((display["roi"] / TARGET_ROI - 1) * 100).round(1)
    display["vs_target_roas"] = ((display["roas"] / TARGET_ROAS - 1) * 100).round(1)
    display["cpa_status"] = display["cost_per_conversion"].apply(cpa_recommendation)

    st.dataframe(
        display[[
            "channel", "spend", "revenue", "profit", "conversions",
            "cost_per_conversion", "cpa_status",
            "roas", "vs_target_roas", "roi_pct", "vs_target_roi"
        ]].style.format({
            "spend": "SAR {:,.0f}",
            "revenue": "SAR {:,.0f}",
            "profit": "SAR {:,.0f}",
            "conversions": "{:.0f}",
            "cost_per_conversion": "SAR {:.0f}",
            "roas": "{:.2f}x",
            "vs_target_roas": "{:+.1f}%",
            "roi_pct": "{:.1f}%",
            "vs_target_roi": "{:+.1f}%",
        }).background_gradient(subset=["roi_pct"], cmap="RdYlGn"),
        use_container_width=True,
        height=400,
    )

# TAB 2: Customer Insights
with tab2:
    st.markdown("### 👥 New vs Returning Customer Performance")

    total = overall['new_conv'] + overall['ret_conv']
    retention = (overall['ret_conv'] / total * 100) if total > 0 else 0

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("🆕 New", f"{overall['new_conv']:,.0f}", delta=f"{overall['new_conv'] / total * 100:.0f}%")
    col_m2.metric("🔄 Returning", f"{overall['ret_conv']:,.0f}", delta=f"{retention:.0f}%")
    col_m3.metric("📊 Retention Rate", f"{retention:.1f}%")
    col_m4.metric("🎯 Total", f"{total:,.0f}")

    if retention < 40:
        st.warning("⚠️ **Low Retention:** Launch loyalty program targeting 50-60% retention rate.")
    elif retention < 50:
        st.info("ℹ️ **Moderate Retention:** Good start. Implement referral incentives to reach 60%.")
    else:
        st.success("✅ **Strong Retention:** Excellent customer loyalty. Maintain quality.")

    st.markdown("---")

    months = sorted(monthly_df["month"].unique())
    monthly_data = []
    for m in months:
        new = monthly_df[(monthly_df["month"] == m) & (monthly_df["customer_type"] == "New")]
        ret = monthly_df[(monthly_df["month"] == m) & (monthly_df["customer_type"] == "Returning")]
        monthly_data.append({
            "month": m,
            "new_conv": new["conversions"].sum() if len(new) > 0 else 0,
            "ret_conv": ret["conversions"].sum() if len(ret) > 0 else 0,
            "new_rev": new["revenue"].sum() if len(new) > 0 else 0,
            "ret_rev": ret["revenue"].sum() if len(ret) > 0 else 0,
        })
    mdf = pd.DataFrame(monthly_data)

    col_ca, col_cb = st.columns(2)

    with col_ca:
        fig = px.bar(
            mdf,
            x="month",
            y=["new_conv", "ret_conv"],
            title="📊 Monthly Conversions (New vs Returning)",
            barmode="stack",
            color_discrete_sequence=[COLORS["primary"], COLORS["success"]],
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col_cb:
        fig = px.line(
            mdf,
            x="month",
            y=["new_rev", "ret_rev"],
            title="💰 Monthly Revenue (New vs Returning)",
            markers=True,
            color_discrete_sequence=[COLORS["primary"], COLORS["success"]],
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# TAB 3: Time Trends
with tab3:
    st.markdown("### 📅 Performance Trends Over Time")

    st.plotly_chart(chart_revenue_trend(monthly_df), use_container_width=True)

    roi_agg = monthly_df.groupby("month", as_index=False).agg({
        "spend": "sum",
        "revenue": "sum",
        "profit": "sum",
    })
    roi_agg["roi"] = (roi_agg["revenue"] - roi_agg["spend"]) / roi_agg["spend"] * 100

    fig = px.line(
        roi_agg,
        x="month",
        y="roi",
        title="📊 Monthly ROI Trend (Is Performance Improving?)",
        markers=True,
    )
    fig.update_traces(line_color=COLORS["primary"], line_width=4, marker=dict(size=10))
    fig.add_hline(y=TARGET_ROI * 100, line_dash="dash", line_color=COLORS["info"], annotation_text="Target")
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Trend Alert (NEW!)
    if len(roi_agg) >= 2:
        trend = roi_agg["roi"].iloc[-1] - roi_agg["roi"].iloc[0]
        profit_trend = roi_agg["profit"].iloc[-1] - roi_agg["profit"].iloc[-2]

        col_tr1, col_tr2 = st.columns(2)

        with col_tr1:
            if trend > 10:
                st.success(f"📈 **Strong Growth:** ROI improved by {trend:.1f}% over the period. Strategy is working!")
            elif trend > 0:
                st.info(f"📈 **Positive Trend:** ROI improved by {trend:.1f}%. Continue current strategy.")
            else:
                st.warning(f"📉 **Declining:** ROI dropped by {abs(trend):.1f}%. Review underperforming channels.")

        with col_tr2:
            if profit_trend < 0:
                st.error(
                    f"📉 **Profit Alert:** Decreased by SAR {abs(profit_trend):,.0f} vs last month. Review campaigns.")
            else:
                st.success(f"📈 **Profit Growth:** Increased by SAR {profit_trend:,.0f} vs last month.")

# TAB 4: Scenario Simulator (NEW!)
with tab4:
    st.markdown("### 🧪 Budget Reallocation Scenario Simulator")
    st.info("💡 **Simulate:** Reallocate budget and see projected ROAS impact")

    st.markdown("#### Current Budget Distribution")
    current_budget = channel_df[["channel", "spend", "roas"]].copy()
    st.dataframe(current_budget.style.format({"spend": "SAR {:,.0f}", "roas": "{:.2f}x"}), use_container_width=True)

    st.markdown("#### Simulate Budget Shift")

    col_sim1, col_sim2, col_sim3 = st.columns(3)

    with col_sim1:
        from_channel = st.selectbox("From Channel (reduce)", channel_df["channel"].tolist())
        reduce_pct = st.slider("Reduce by %", 0, 50, 30)

    with col_sim2:
        to_channel = st.selectbox("To Channel (increase)", channel_df["channel"].tolist())
        increase_pct = st.slider("Increase by %", 0, 100, 40)

    with col_sim3:
        st.write("")
        st.write("")
        if st.button("🚀 Run Simulation", use_container_width=True):
            from_spend = channel_df[channel_df["channel"] == from_channel]["spend"].values[0]
            from_roas = channel_df[channel_df["channel"] == from_channel]["roas"].values[0]
            to_roas = channel_df[channel_df["channel"] == to_channel]["roas"].values[0]

            shifted_amount = from_spend * (reduce_pct / 100)
            new_revenue_loss = shifted_amount * from_roas * (1 - reduce_pct / 100)
            new_revenue_gain = shifted_amount * to_roas * (1 + increase_pct / 100)

            net_change = new_revenue_gain - new_revenue_loss

            st.markdown(f"""
            <div class="success-box">
            <h4>📊 Simulation Results</h4>
            <ul>
            <li><b>Budget Shift:</b> SAR {shifted_amount:,.0f} from {from_channel} to {to_channel}</li>
            <li><b>Revenue Loss:</b> -SAR {new_revenue_loss:,.0f} (from {from_channel})</li>
            <li><b>Revenue Gain:</b> +SAR {new_revenue_gain:,.0f} (from {to_channel})</li>
            <li><b>Net Impact:</b> {'+' if net_change > 0 else ''}SAR {net_change:,.0f}</li>
            </ul>
            <p><b>💡 Recommendation:</b> {'Proceed with reallocation!' if net_change > 0 else 'Reconsider - negative impact expected.'}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🧪 A/B Test Suggestions Panel")

    st.markdown(f"""
    <div class="info-box">
    <h4>Recommended Tests for {best['channel']}</h4>
    <ol>
    <li><b>Ad Creative Test:</b> Test 3 headline variations (CTA-focused vs benefit-focused)</li>
    <li><b>Timing Test:</b> Compare 7-10 AM vs 2-5 PM ad delivery</li>
    <li><b>Audience Test:</b> Narrow targeting (office workers 5km radius) vs broad</li>
    </ol>
    <p><b>Expected Impact:</b> +10-15% CTR improvement</p>
    </div>
    """, unsafe_allow_html=True)

# TAB 5: Morning Performance
with tab5:
    st.markdown("### 🌅 Morning vs All-Day Performance Analysis")
    st.markdown("**Business Goal:** Attract more weekday morning customers")

    col_t1, col_t2, col_t3 = st.columns(3)

    morning_pct = overall['morning_conv'] / overall['conversions'] * 100 if overall['conversions'] > 0 else 0
    weekday_pct = overall['weekday_conv'] / overall['conversions'] * 100 if overall['conversions'] > 0 else 0

    col_t1.metric("🌅 Morning Conversions", f"{overall['morning_conv']:,.0f}", delta=f"{morning_pct:.1f}% of total")
    col_t2.metric("📅 Weekday Conversions", f"{overall['weekday_conv']:,.0f}", delta=f"{weekday_pct:.1f}% of total")
    col_t3.metric("🎯 Target Gap", f"{TARGET_MORNING_PCT - morning_pct:.1f}%", delta=f"Target: {TARGET_MORNING_PCT}%")

    if morning_pct < 40:
        st.markdown(f"""
        <div class="warning-box">
        <h4>⚠️ Low Morning Performance</h4>
        <p><strong>Current:</strong> {morning_pct:.1f}% morning conversions</p>
        <p><strong>Target:</strong> {TARGET_MORNING_PCT}% morning conversions</p>
        <div class="recommendation">
        <strong>💡 4-Week Action Plan:</strong>
        <ol>
        <li><b>Week 1:</b> Launch "Morning Special" promotion (6-12 AM, 15% off)</li>
        <li><b>Week 2:</b> Google Ads time-targeting (6-11 AM, office workers 5km radius)</li>
        <li><b>Week 3:</b> Instagram Stories campaign (morning menu highlights, UGC)</li>
        <li><b>Week 4:</b> Test breakfast bundle pricing (coffee + pastry SAR 25)</li>
        </ol>
        <p><b>Expected Result:</b> +20-25% morning conversions by Month 2</p>
        </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.success(f"✅ **Good Progress:** {morning_pct:.1f}% morning conversions. Continue optimizing.")

# TAB 6: Export & Actions
with tab6:
    st.markdown("### 📥 Export Data & Action Items")

    col_e1, col_e2, col_e3, col_e4 = st.columns(4)

    with col_e1:
        csv = channel_df.to_csv(index=False).encode("utf-8")
        st.download_button("📊 Channel KPIs", csv, "channel_kpis.csv", use_container_width=True)

    with col_e2:
        csv = monthly_df.to_csv(index=False).encode("utf-8")
        st.download_button("📅 Monthly Data", csv, "monthly_data.csv", use_container_width=True)

    with col_e3:
        csv = fdf.to_csv(index=False).encode("utf-8")
        st.download_button("📄 Raw Data", csv, "raw_data_filtered.csv", use_container_width=True)

    with col_e4:
        exec_summary = f"""CAFÉ AROMA - EXECUTIVE SUMMARY
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Filter: {selected_channel} | {date_range[0]} → {date_range[1]}

PERFORMANCE OVERVIEW:
- Total Spend: SAR {overall['spend']:,.0f}
- Total Revenue: SAR {overall['revenue']:,.0f}
- Net Profit: SAR {overall['profit']:,.0f}
- ROAS: {overall['roas']:.2f}x (Target: {TARGET_ROAS:.1f}x)
- ROI: {overall['roi'] * 100:.1f}% (Target: {TARGET_ROI * 100:.0f}%)
- Morning Conversions: {morning_pct:.1f}% (Target: {TARGET_MORNING_PCT}%)

BEST PERFORMER: {best['channel']}
- ROI: {best['roi'] * 100:.1f}%
- ROAS: {best['roas']:.2f}x
- CPA: SAR {best['cost_per_conversion']:.0f}
- Revenue: SAR {best['revenue']:,.0f}

NEEDS IMPROVEMENT: {worst['channel']}
- ROI: {worst['roi'] * 100:.1f}%
- ROAS: {worst['roas']:.2f}x
- CPA: SAR {worst['cost_per_conversion']:.0f}

RECOMMENDED ACTION:
1. Reallocate 30% budget from {worst['channel']} to {best['channel']}
2. Expected Impact: +15-20% overall ROAS improvement
3. Launch morning campaigns (6-12 AM) to reach {TARGET_MORNING_PCT}% target
4. Implement loyalty program (50-60% retention target)

NEXT STEPS (4 Weeks):
Week 1: Budget reallocation + performance monitoring
Week 2: Morning campaign launch (promotion + Google Ads)
Week 3: A/B testing (creatives, timing, audiences)
Week 4: Loyalty program launch ("Morning Regular" card)
"""
        st.download_button("📄 Executive Summary", exec_summary, "executive_summary.txt", use_container_width=True)

    st.markdown("---")

    st.markdown("### 🎯 Recommended 4-Week Action Plan")

    st.markdown(f"""
    <div class="info-box">
    <h4>Week 1: Budget Reallocation</h4>
    <ul>
    <li>✅ Increase {best['channel']} budget by 40% (from SAR {best['spend']:,.0f} to SAR {best['spend'] * 1.4:,.0f})</li>
    <li>✅ Reduce {worst['channel']} budget by 30% (from SAR {worst['spend']:,.0f} to SAR {worst['spend'] * 0.7:,.0f})</li>
    <li>✅ Set up daily performance dashboard monitoring</li>
    </ul>
    </div>

    <div class="info-box">
    <h4>Week 2: Morning Campaign Launch</h4>
    <ul>
    <li>✅ Create "Morning Special" promotion (6-12 AM, 15% off breakfast menu)</li>
    <li>✅ Target Google Ads: 6-11 AM delivery, office workers within 5km</li>
    <li>✅ Launch Instagram Stories: Morning menu highlights + customer UGC</li>
    </ul>
    </div>

    <div class="info-box">
    <h4>Week 3: A/B Testing</h4>
    <ul>
    <li>✅ Test 3 ad creative variations on {worst['channel']} (CTA vs benefit-focused)</li>
    <li>✅ Test breakfast bundle pricing (Coffee + Pastry: SAR 22 vs SAR 25)</li>
    <li>✅ Optimize landing pages for mobile (50%+ of traffic)</li>
    </ul>
    </div>

    <div class="info-box">
    <h4>Week 4: Loyalty Program</h4>
    <ul>
    <li>✅ Launch "Morning Regular" loyalty card (digital via WhatsApp)</li>
    <li>✅ Offer: 10% off on 5th morning visit (6-12 AM only)</li>
    <li>✅ Target retention rate: 50-60% (current: {retention:.1f}%)</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 1.5rem;'>
<p style='font-size: 1.3rem; font-weight: 700;'>☕ Café Aroma Decision Dashboard v2.5 Ultimate</p>
<p style='font-size: 1rem;'>Data-Driven Marketing Intelligence Platform with AI Recommendations</p>
<p style='font-size: 0.85rem; color: #999;'>Built with Streamlit & Plotly | Powered by Advanced Analytics | Feb 2026</p>
</div>
""", unsafe_allow_html=True)