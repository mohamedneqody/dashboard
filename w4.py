"""
╔═══════════════════════════════════════════════════════════════╗
║    CAFÉ AROMA - KPI DASHBOARD (FINAL VERSION)               ║
║    Complete, Tested & Ready to Run                          ║
╚═══════════════════════════════════════════════════════════════╝

✅ FEATURES:
- Auto-detects column names (spaces/underscores)
- Premium UI with colors & gradients
- Interactive charts (Plotly)
- Export to CSV
- Insights & Recommendations
- 100% Working - Tested!

SAVE AS: cafe_aroma_final.py

USAGE:
    streamlit run cafe_aroma_final.py
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG (Must be first Streamlit command)
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Café Aroma KPI Dashboard",
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
}

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}
    
    .main-title {{
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }}
    
    .sub-title {{
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }}
    
    .success-box {{
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 2px solid {COLORS['success']};
        border-radius: 10px;
        padding: 1.2rem;
        margin: 1rem 0;
    }}
    
    .warning-box {{
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border: 2px solid {COLORS['warning']};
        border-radius: 10px;
        padding: 1.2rem;
        margin: 1rem 0;
    }}
    
    .info-box {{
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        border: 2px solid {COLORS['info']};
        border-radius: 10px;
        padding: 1.2rem;
        margin: 1rem 0;
    }}
    
    div[data-testid="stMetricValue"] {{
        font-size: 1.8rem;
        font-weight: 700;
        color: {COLORS['primary']};
    }}
    
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        height: 50px;
        background-color: #f0f2f6;
        border-radius: 8px 8px 0 0;
        font-weight: 600;
    }}
    
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

DEFAULT_FILE = "Café Aroma Pipeline.xlsx"

COLUMN_ALIASES = {
    "date": ["date", "Date", "transaction_date", "Transaction Date", "transaction date"],
    "channel": ["channel", "Channel", "marketing_channel", "Marketing Channel", "marketing channel"],
    "customer_type": ["customer_type", "Customer Type", "customer type", "customertype", "type"],
    "ad_spend": ["ad_spend", "Ad Spend", "ad spend", "spend", "Spend"],
    "conversions": ["conversions", "Conversions", "orders", "Orders"],
    "revenue": ["revenue", "Revenue", "sales", "Sales"],
    "service_category": ["service_category", "Service Category", "service category", "category"],
}

# ═══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def find_column(df_cols, canonical):
    """Find column from aliases"""
    for candidate in COLUMN_ALIASES.get(canonical, []):
        if candidate in df_cols:
            return candidate
    return None

def normalize_channel(x):
    """Normalize channel names"""
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
    """Normalize customer type"""
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()
    if "new" in s: return "New"
    if "return" in s: return "Returning"
    return str(x).strip()

def add_kpis(df):
    """Add KPI calculations"""
    df = df.copy()
    df["profit"] = df["revenue"] - df["spend"]
    df["roas"] = np.where(df["spend"] > 0, df["revenue"] / df["spend"], 0)
    df["roi"] = np.where(df["spend"] > 0, (df["revenue"] - df["spend"]) / df["spend"], 0)
    df["cost_per_conversion"] = np.where(df["conversions"] > 0, df["spend"] / df["conversions"], 0)
    df["revenue_per_conversion"] = np.where(df["conversions"] > 0, df["revenue"] / df["conversions"], 0)
    return df

# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════

def load_excel(file_path, sheet_name):
    """Load and process Excel file"""
    try:
        # Read Excel
        raw = pd.read_excel(file_path, sheet_name=sheet_name)
        cols = list(raw.columns)

        # Map columns
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
            st.info(f"📋 Found columns: {cols}")
            return None

        # Build DataFrame
        df = pd.DataFrame({
            "date": pd.to_datetime(raw[col_map["date"]], errors="coerce"),
            "channel": raw[col_map["channel"]].apply(normalize_channel),
            "customer_type": raw[col_map["customer_type"]].apply(normalize_customer_type),
            "ad_spend": pd.to_numeric(raw[col_map["ad_spend"]], errors="coerce").fillna(0),
            "conversions": pd.to_numeric(raw[col_map["conversions"]], errors="coerce").fillna(0),
            "revenue": pd.to_numeric(raw[col_map["revenue"]], errors="coerce").fillna(0),
        })

        # Add service category if exists
        svc_col = find_column(cols, "service_category")
        if svc_col:
            df["service_category"] = raw[svc_col].astype(str).str.strip()

        # Clean
        df = df[df["date"].notna()].copy()
        df = df[df["channel"] != ""].copy()
        df = df[df["customer_type"] != ""].copy()

        df["month"] = df["date"].dt.to_period("M").astype(str)

        return df

    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
        return None

# ═══════════════════════════════════════════════════════════════
# KPI CALCULATIONS
# ═══════════════════════════════════════════════════════════════

def calc_overall(df):
    """Overall KPIs"""
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
    }

def calc_by_channel(df):
    """KPIs by channel"""
    agg = df.groupby("channel", as_index=False).agg({
        "ad_spend": "sum",
        "revenue": "sum",
        "conversions": "sum",
    }).rename(columns={"ad_spend": "spend"})

    # New/Returning split
    for ch in agg["channel"]:
        new = df[(df["channel"] == ch) & (df["customer_type"] == "New")]["conversions"].sum()
        ret = df[(df["channel"] == ch) & (df["customer_type"] == "Returning")]["conversions"].sum()
        agg.loc[agg["channel"] == ch, "new_conversions"] = new
        agg.loc[agg["channel"] == ch, "returning_conversions"] = ret

    return add_kpis(agg).sort_values("roi", ascending=False)

def calc_monthly(df):
    """Monthly KPIs by customer type"""
    agg = df.groupby(["month", "customer_type"], as_index=False).agg({
        "ad_spend": "sum",
        "revenue": "sum",
        "conversions": "sum",
    }).rename(columns={"ad_spend": "spend"})

    return add_kpis(agg)

def calc_service(df):
    """Service category KPIs"""
    if "service_category" not in df.columns:
        return None

    agg = df.groupby("service_category", as_index=False).agg({
        "revenue": "sum",
        "conversions": "sum",
    })

    agg["avg_revenue"] = agg["revenue"] / agg["conversions"]

    return agg.sort_values("revenue", ascending=False)

# ═══════════════════════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════════════════════

def chart_roi(channel_df):
    """ROI chart"""
    data = channel_df.sort_values("roi", ascending=True).copy()
    data["roi_pct"] = data["roi"] * 100

    fig = go.Figure(data=[
        go.Bar(
            x=data["roi_pct"],
            y=data["channel"],
            orientation="h",
            text=data["roi_pct"].apply(lambda x: f"{x:.1f}%"),
            textposition="outside",
            marker=dict(
                color=data["roi_pct"],
                colorscale=[[0, COLORS["danger"]], [0.5, COLORS["warning"]], [1, COLORS["success"]]],
            ),
        )
    ])

    fig.update_layout(
        title="📊 ROI by Channel",
        xaxis_title="ROI (%)",
        height=400,
        showlegend=False,
    )

    fig.add_vline(x=100, line_dash="dash", line_color=COLORS["info"])

    return fig

def chart_roas(channel_df):
    """ROAS chart"""
    data = channel_df.sort_values("roas", ascending=True).copy()

    fig = px.bar(
        data,
        x="roas",
        y="channel",
        orientation="h",
        text=data["roas"].apply(lambda x: f"{x:.2f}x"),
        title="💰 ROAS by Channel",
        color="roas",
        color_continuous_scale="Blues",
    )

    fig.update_traces(textposition="outside")
    fig.update_layout(height=400, showlegend=False)

    return fig

def chart_monthly_trend(monthly_df):
    """Monthly revenue trend"""
    agg = monthly_df.groupby("month", as_index=False).agg({
        "spend": "sum",
        "revenue": "sum",
    })

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=agg["month"],
        y=agg["revenue"],
        mode="lines+markers",
        name="Revenue",
        line=dict(color=COLORS["success"], width=3),
        marker=dict(size=8),
    ))
    fig.add_trace(go.Scatter(
        x=agg["month"],
        y=agg["spend"],
        mode="lines+markers",
        name="Spend",
        line=dict(color=COLORS["danger"], width=3),
        marker=dict(size=8),
    ))

    fig.update_layout(
        title="📈 Monthly Revenue vs Spend",
        xaxis_title="Month",
        yaxis_title="Amount (SAR)",
        height=400,
        hovermode="x unified",
    )

    return fig

# ═══════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════

# Header
st.markdown('<div class="main-title">☕ Café Aroma - KPI Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Real-time Marketing Analytics & Performance Tracking</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")

    uploaded = st.file_uploader("📁 Upload Excel", type=["xlsx", "xls"])

    if uploaded:
        file_src = uploaded
    else:
        path = st.text_input("📂 File path", value=DEFAULT_FILE)
        file_src = path

    sheet = st.text_input("📄 Sheet name/index", value="0")

    try:
        sheet_val = int(sheet)
    except:
        sheet_val = sheet

    st.divider()

    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.caption("v1.0 Final | Feb 2026")

# Load data
@st.cache_data(show_spinner=False)
def load_all(file_source, sheet_name):
    df = load_excel(file_source, sheet_name)
    if df is None:
        return None

    return {
        "df": df,
        "overall": calc_overall(df),
        "channel": calc_by_channel(df),
        "monthly": calc_monthly(df),
        "service": calc_service(df),
    }

with st.spinner("📊 Loading data..."):
    data = load_all(file_src, sheet_val)

if data is None:
    st.error("❌ Failed to load data. Check file path and settings.")
    st.stop()

overall = data["overall"]
channel_df = data["channel"]
monthly_df = data["monthly"]
service_df = data["service"]
raw_df = data["df"]

# KPI Cards
st.markdown("### 📊 Key Performance Indicators")

c1, c2, c3, c4, c5, c6 = st.columns(6)

c1.metric("💰 Total Spend", f"SAR {overall['spend']:,.0f}")
c2.metric("💵 Revenue", f"SAR {overall['revenue']:,.0f}")
c3.metric("📈 Profit", f"SAR {overall['profit']:,.0f}")
c4.metric("🎯 Conversions", f"{overall['conversions']:,.0f}")
c5.metric("💎 ROAS", f"{overall['roas']:.2f}x")
c6.metric("📊 ROI", f"{overall['roi']*100:.1f}%")

st.caption(
    f"📅 Period: **{overall['first_date']}** → **{overall['last_date']}** | "
    f"📊 Records: **{overall['records']:,}**"
)

st.divider()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Channel Performance",
    "👥 Customer Analysis",
    "📅 Monthly Trends",
    "📥 Export"
])

# TAB 1: Channel Performance
with tab1:
    st.markdown("### 🏆 Channel Performance Ranking")

    best = channel_df.iloc[0]
    worst = channel_df.iloc[-1]

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(f"""
        <div class="success-box">
        <h4>🥇 Best: {best['channel']}</h4>
        <ul>
        <li><b>ROI:</b> {best['roi']*100:.1f}%</li>
        <li><b>ROAS:</b> {best['roas']:.2f}x</li>
        <li><b>Revenue:</b> SAR {best['revenue']:,.0f}</li>
        <li><b>Profit:</b> SAR {best['profit']:,.0f}</li>
        </ul>
        <p><b>💡 Recommendation:</b> Increase budget by 40-50%</p>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown(f"""
        <div class="warning-box">
        <h4>⚠️ Needs Work: {worst['channel']}</h4>
        <ul>
        <li><b>ROI:</b> {worst['roi']*100:.1f}%</li>
        <li><b>ROAS:</b> {worst['roas']:.2f}x</li>
        <li><b>Cost/Conv:</b> SAR {worst['cost_per_conversion']:.0f}</li>
        </ul>
        <p><b>💡 Recommendation:</b> Optimize or reduce budget 30%</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    col_c1, col_c2 = st.columns(2)

    with col_c1:
        st.plotly_chart(chart_roi(channel_df), use_container_width=True)

    with col_c2:
        st.plotly_chart(chart_roas(channel_df), use_container_width=True)

    st.markdown("### 📋 Channel Details")

    display = channel_df.copy()
    display["roi_pct"] = display["roi"] * 100

    st.dataframe(
        display[[
            "channel", "spend", "revenue", "profit", "conversions",
            "new_conversions", "returning_conversions", "roas", "roi_pct",
            "cost_per_conversion"
        ]].style.format({
            "spend": "SAR {:,.0f}",
            "revenue": "SAR {:,.0f}",
            "profit": "SAR {:,.0f}",
            "conversions": "{:.0f}",
            "new_conversions": "{:.0f}",
            "returning_conversions": "{:.0f}",
            "roas": "{:.2f}x",
            "roi_pct": "{:.1f}%",
            "cost_per_conversion": "SAR {:.0f}",
        }),
        use_container_width=True,
        height=350,
    )

# TAB 2: Customer Analysis
with tab2:
    st.markdown("### 👥 New vs Returning Customers")

    total = overall['new_conv'] + overall['ret_conv']
    retention = (overall['ret_conv'] / total * 100) if total > 0 else 0

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("🆕 New", f"{overall['new_conv']:,.0f}")
    col_m2.metric("🔄 Returning", f"{overall['ret_conv']:,.0f}")
    col_m3.metric("📊 Retention", f"{retention:.1f}%")

    st.markdown("---")

    # Build monthly customer data
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
            title="📊 Monthly Conversions",
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
            title="💰 Monthly Revenue",
            markers=True,
            color_discrete_sequence=[COLORS["primary"], COLORS["success"]],
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# TAB 3: Monthly Trends
with tab3:
    st.markdown("### 📅 Monthly Performance")

    st.plotly_chart(chart_monthly_trend(monthly_df), use_container_width=True)

    # ROI trend
    roi_agg = monthly_df.groupby("month", as_index=False).agg({
        "spend": "sum",
        "revenue": "sum",
    })
    roi_agg["roi"] = (roi_agg["revenue"] - roi_agg["spend"]) / roi_agg["spend"] * 100

    fig = px.line(
        roi_agg,
        x="month",
        y="roi",
        title="📊 Monthly ROI Trend",
        markers=True,
    )
    fig.update_traces(line_color=COLORS["primary"], line_width=3)
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# TAB 4: Export
with tab4:
    st.markdown("### 📥 Download Data")

    col_e1, col_e2, col_e3 = st.columns(3)

    with col_e1:
        csv = channel_df.to_csv(index=False).encode("utf-8")
        st.download_button("📊 Channel KPIs", csv, "channel_kpis.csv", use_container_width=True)

    with col_e2:
        csv = monthly_df.to_csv(index=False).encode("utf-8")
        st.download_button("📅 Monthly Data", csv, "monthly_data.csv", use_container_width=True)

    with col_e3:
        csv = raw_df.to_csv(index=False).encode("utf-8")
        st.download_button("📄 Raw Data", csv, "raw_data.csv", use_container_width=True)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
<p><b>☕ Café Aroma KPI Dashboard v1.0</b></p>
<p>Made with Streamlit & Plotly | Feb 2026</p>
</div>
""", unsafe_allow_html=True)
