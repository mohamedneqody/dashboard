"""
╔═══════════════════════════════════════════════════════════════╗
║    CAFÉ AROMA - MORNING CUSTOMER ANALYTICS DASHBOARD         ║
║    Targeted Weekday Morning Customer Acquisition             ║
╚═══════════════════════════════════════════════════════════════╝

✨ NEW FEATURES:
- Morning Weekday Customer Analysis
- Flyer vs Email Performance Tracking
- Morning Conversion Rate Analysis
- Customer Acquisition Cost by Time
- Targeted Campaign Recommendations
- Weekly Morning Trend Analysis
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import plotly.subplots as sp

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Café Aroma Morning Analytics",
    page_icon="🌅",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════
# COLORS & STYLING
# ═══════════════════════════════════════════════════════════════

COLORS = {
    "morning_primary": "#FF8C00",  # Orange for morning theme
    "morning_secondary": "#FFD700",  # Gold
    "morning_success": "#32CD32",  # Lime green
    "morning_warning": "#FF6B6B",  # Coral red
    "morning_info": "#1E90FF",  # Dodger blue
    "morning_dark": "#2F4F4F",  # Dark slate gray
}

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    
    * {{
        font-family: 'Inter', sans-serif;
    }}
    
    .morning-title {{
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, {COLORS['morning_primary']} 0%, {COLORS['morning_secondary']} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.3rem;
        letter-spacing: -0.02em;
    }}
    
    .morning-subtitle {{
        font-size: 1.3rem;
        color: {COLORS['morning_dark']};
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }}
    
    .morning-banner {{
        background: linear-gradient(135deg, {COLORS['morning_primary']} 0%, {COLORS['morning_secondary']} 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        box-shadow: 0 8px 16px rgba(255, 140, 0, 0.3);
    }}
    
    .morning-card {{
        background: linear-gradient(135deg, #FFF8E1 0%, #FFECB3 100%);
        border: 2px solid {COLORS['morning_primary']};
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(255, 140, 0, 0.2);
    }}
    
    .morning-target {{
        background: linear-gradient(135deg, #E1F5FE 0%, #B3E5FC 100%);
        border: 2px solid {COLORS['morning_info']};
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        text-align: center;
        font-weight: 600;
    }}
    
    .morning-recommendation {{
        background: rgba(255, 255, 255, 0.95);
        padding: 1rem;
        border-radius: 8px;
        margin-top: 0.8rem;
        border-left: 4px solid {COLORS['morning_success']};
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    
    .time-slot-card {{
        background: white;
        border: 2px solid {COLORS['morning_secondary']};
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem;
        text-align: center;
    }}
    
    div[data-testid="stMetricValue"] {{
        font-size: 2rem;
        font-weight: 700;
    }}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION - MORNING CUSTOMER TARGETS
# ═══════════════════════════════════════════════════════════════

DEFAULT_FILE = "Café Aroma Pipeline.xlsx"

# Morning & Weekday Targets
TARGET_MORNING_CONVERSION_RATE = 0.25  # 25% of conversions should be in morning
TARGET_WEEKDAY_MORNING_CONVERSION = 100  # Target conversions per month
MORNING_HOURS = [6, 7, 8, 9, 10, 11]  # 6 AM - 12 PM
WEEKDAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

# Campaign Channels (Note: Add "Email" if available in data)
MORNING_CAMPAIGN_CHANNELS = ['Flyer', 'Email', 'Google Ads', 'Instagram', 'Referral']

# ═══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS - MORNING ANALYTICS
# ═══════════════════════════════════════════════════════════════

def analyze_morning_performance(df):
    """Analyze morning and weekday performance"""

    # Create time-based flags
    df['hour'] = df['date'].dt.hour
    df['is_morning'] = df['hour'].isin(MORNING_HOURS)
    df['day_name'] = df['date'].dt.day_name()
    df['is_weekday'] = df['day_name'].isin(WEEKDAYS)
    df['is_weekday_morning'] = df['is_morning'] & df['is_weekday']

    # Morning analysis
    morning_df = df[df['is_morning']]
    weekday_morning_df = df[df['is_weekday_morning']]

    # Overall metrics
    total_conversions = df['conversions'].sum()
    total_revenue = df['revenue'].sum()

    # Morning metrics
    morning_conversions = morning_df['conversions'].sum()
    morning_revenue = morning_df['revenue'].sum()
    morning_spend = morning_df['ad_spend'].sum()

    # Weekday morning metrics
    weekday_morning_conversions = weekday_morning_df['conversions'].sum()
    weekday_morning_revenue = weekday_morning_df['revenue'].sum()
    weekday_morning_spend = weekday_morning_df['ad_spend'].sum()

    # Calculate rates
    morning_conversion_rate = morning_conversions / total_conversions if total_conversions > 0 else 0
    weekday_morning_conversion_rate = weekday_morning_conversions / total_conversions if total_conversions > 0 else 0

    # Calculate efficiency metrics
    morning_roi = (morning_revenue - morning_spend) / morning_spend if morning_spend > 0 else 0
    morning_roas = morning_revenue / morning_spend if morning_spend > 0 else 0
    morning_cpa = morning_spend / morning_conversions if morning_conversions > 0 else 0

    weekday_morning_roi = (weekday_morning_revenue - weekday_morning_spend) / weekday_morning_spend if weekday_morning_spend > 0 else 0
    weekday_morning_roas = weekday_morning_revenue / weekday_morning_spend if weekday_morning_spend > 0 else 0
    weekday_morning_cpa = weekday_morning_spend / weekday_morning_conversions if weekday_morning_conversions > 0 else 0

    return {
        # Overall
        'total_conversions': total_conversions,
        'total_revenue': total_revenue,

        # Morning metrics
        'morning_conversions': morning_conversions,
        'morning_revenue': morning_revenue,
        'morning_spend': morning_spend,
        'morning_conversion_rate': morning_conversion_rate,
        'morning_roi': morning_roi,
        'morning_roas': morning_roas,
        'morning_cpa': morning_cpa,

        # Weekday morning metrics
        'weekday_morning_conversions': weekday_morning_conversions,
        'weekday_morning_revenue': weekday_morning_revenue,
        'weekday_morning_spend': weekday_morning_spend,
        'weekday_morning_conversion_rate': weekday_morning_conversion_rate,
        'weekday_morning_roi': weekday_morning_roi,
        'weekday_morning_roas': weekday_morning_roas,
        'weekday_morning_cpa': weekday_morning_cpa,

        # Dataframes for detailed analysis
        'morning_df': morning_df,
        'weekday_morning_df': weekday_morning_df
    }

def analyze_morning_by_channel(df):
    """Analyze morning performance by channel"""

    morning_df = df[df['date'].dt.hour.isin(MORNING_HOURS)]
    weekday_morning_df = df[(df['date'].dt.hour.isin(MORNING_HOURS)) &
                           (df['date'].dt.day_name().isin(WEEKDAYS))]

    # Morning by channel
    morning_channel = morning_df.groupby('channel').agg({
        'ad_spend': 'sum',
        'revenue': 'sum',
        'conversions': 'sum'
    }).rename(columns={'ad_spend': 'spend'}).reset_index()

    # Weekday morning by channel
    weekday_morning_channel = weekday_morning_df.groupby('channel').agg({
        'ad_spend': 'sum',
        'revenue': 'sum',
        'conversions': 'sum'
    }).rename(columns={'ad_spend': 'spend'}).reset_index()

    # Add KPIs
    for df_channel in [morning_channel, weekday_morning_channel]:
        df_channel['roi'] = np.where(df_channel['spend'] > 0,
                                     (df_channel['revenue'] - df_channel['spend']) / df_channel['spend'], 0)
        df_channel['roas'] = np.where(df_channel['spend'] > 0,
                                      df_channel['revenue'] / df_channel['spend'], 0)
        df_channel['cpa'] = np.where(df_channel['conversions'] > 0,
                                     df_channel['spend'] / df_channel['conversions'], 0)

    return {
        'morning_by_channel': morning_channel,
        'weekday_morning_by_channel': weekday_morning_channel
    }

def analyze_morning_by_hour(df):
    """Analyze performance by hour of the day"""

    df['hour'] = df['date'].dt.hour

    hourly_analysis = df.groupby('hour').agg({
        'ad_spend': 'sum',
        'revenue': 'sum',
        'conversions': 'sum',
        'customer_type': lambda x: (x == 'New').sum()  # Count new customers
    }).rename(columns={
        'ad_spend': 'spend',
        'customer_type': 'new_customers'
    }).reset_index()

    # Calculate hourly KPIs
    hourly_analysis['conversion_rate'] = hourly_analysis['conversions'] / hourly_analysis['conversions'].sum()
    hourly_analysis['cpa'] = np.where(hourly_analysis['conversions'] > 0,
                                     hourly_analysis['spend'] / hourly_analysis['conversions'], 0)
    hourly_analysis['roi'] = np.where(hourly_analysis['spend'] > 0,
                                     (hourly_analysis['revenue'] - hourly_analysis['spend']) / hourly_analysis['spend'], 0)

    return hourly_analysis

def analyze_morning_by_weekday(df):
    """Analyze performance by weekday"""

    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour

    weekday_analysis = df.groupby(['day_name', 'hour']).agg({
        'conversions': 'sum',
        'revenue': 'sum',
        'ad_spend': 'sum'
    }).reset_index()

    # Pivot for heatmap
    pivot_data = weekday_analysis.pivot_table(
        index='day_name',
        columns='hour',
        values='conversions',
        aggfunc='sum',
        fill_value=0
    )

    # Ensure proper order
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pivot_data = pivot_data.reindex(day_order)

    return pivot_data

def get_morning_recommendations(analysis_results):
    """Generate targeted recommendations for morning customer acquisition"""

    recs = []

    morning_rate = analysis_results['morning_conversion_rate']
    weekday_morning_rate = analysis_results['weekday_morning_conversion_rate']

    if morning_rate < TARGET_MORNING_CONVERSION_RATE:
        recs.append({
            'priority': 'high',
            'title': '📈 Increase Morning Conversion Rate',
            'action': f'Current morning conversion rate is {morning_rate:.1%} vs target of {TARGET_MORNING_CONVERSION_RATE:.0%}',
            'steps': [
                'Launch "Early Bird" promotions for 6-9 AM',
                'Optimize Flyer campaigns for morning commuters',
                'Implement email drip campaign for morning offers'
            ]
        })

    if analysis_results['weekday_morning_conversions'] < TARGET_WEEKDAY_MORNING_CONVERSION:
        recs.append({
            'priority': 'high',
            'title': '🎯 Boost Weekday Morning Traffic',
            'action': f'Only {analysis_results["weekday_morning_conversions"]} weekday morning conversions vs target of {TARGET_WEEKDAY_MORNING_CONVERSION}',
            'steps': [
                'Target office areas with morning flyer distribution',
                'Create weekday-only morning specials',
                'Partner with nearby businesses for cross-promotions'
            ]
        })

    # Check channel performance
    morning_cpa = analysis_results['morning_cpa']
    if morning_cpa > 50:  # If CPA is high
        recs.append({
            'priority': 'medium',
            'title': '💰 Reduce Morning Acquisition Cost',
            'action': f'Morning CPA is SAR {morning_cpa:.0f}',
            'steps': [
                'Focus on higher-performing morning channels',
                'Test different morning ad creatives',
                'Implement retargeting for morning visitors'
            ]
        })

    return recs

# ═══════════════════════════════════════════════════════════════
# VISUALIZATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def create_morning_heatmap(pivot_data):
    """Create heatmap of conversions by weekday and hour"""

    fig = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        x=[f"{h}:00" for h in pivot_data.columns],
        y=pivot_data.index,
        colorscale='Oranges',
        text=pivot_data.values,
        texttemplate='%{text:.0f}',
        textfont={"size": 10},
        hoverinfo='text',
        hovertemplate='<b>%{y} %{x}</b><br>Conversions: %{z:.0f}<extra></extra>'
    ))

    # Highlight morning hours
    morning_hours = [f"{h}:00" for h in MORNING_HOURS]
    for i, hour in enumerate(pivot_data.columns):
        if f"{hour}:00" in morning_hours:
            fig.add_vrect(
                x0=i-0.5, x1=i+0.5,
                fillcolor="rgba(255, 215, 0, 0.1)",
                line_width=0,
                layer="below"
            )

    fig.update_layout(
        title='🔥 Heatmap: Conversions by Day & Hour',
        xaxis_title='Hour of Day',
        yaxis_title='Day of Week',
        height=500,
        font=dict(size=12)
    )

    return fig

def create_morning_channel_performance(channel_data):
    """Create bar chart of channel performance in morning hours"""

    fig = go.Figure()

    # Add bars for morning conversions
    fig.add_trace(go.Bar(
        x=channel_data['channel'],
        y=channel_data['conversions'],
        name='Morning Conversions',
        marker_color=COLORS['morning_primary'],
        text=channel_data['conversions'].astype(int),
        textposition='outside'
    ))

    # Add line for CPA
    fig.add_trace(go.Scatter(
        x=channel_data['channel'],
        y=channel_data['cpa'],
        name='Cost per Acquisition (CPA)',
        yaxis='y2',
        line=dict(color=COLORS['morning_warning'], width=3),
        mode='lines+markers'
    ))

    fig.update_layout(
        title='📊 Morning Performance by Channel',
        xaxis_title='Channel',
        yaxis_title='Conversions',
        yaxis2=dict(
            title='CPA (SAR)',
            overlaying='y',
            side='right',
            range=[0, channel_data['cpa'].max() * 1.2]
        ),
        height=450,
        barmode='group',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )

    return fig

def create_morning_trend_chart(df):
    """Create trend chart of morning conversions over time"""

    # Add morning flag
    df['is_morning'] = df['date'].dt.hour.isin(MORNING_HOURS)

    # Group by date and morning flag
    daily_analysis = df.groupby([df['date'].dt.date, 'is_morning']).agg({
        'conversions': 'sum',
        'revenue': 'sum'
    }).reset_index()

    # Separate morning and non-morning
    morning_daily = daily_analysis[daily_analysis['is_morning']]
    other_daily = daily_analysis[~daily_analysis['is_morning']]

    fig = go.Figure()

    # Morning trend
    fig.add_trace(go.Scatter(
        x=morning_daily['date'],
        y=morning_daily['conversions'],
        name='Morning Conversions',
        mode='lines+markers',
        line=dict(color=COLORS['morning_primary'], width=3),
        fill='tozeroy',
        fillcolor='rgba(255, 140, 0, 0.1)'
    ))

    # Non-morning trend
    fig.add_trace(go.Scatter(
        x=other_daily['date'],
        y=other_daily['conversions'],
        name='Other Hours',
        mode='lines',
        line=dict(color='gray', width=2, dash='dash')
    ))

    fig.update_layout(
        title='📈 Morning vs Non-Morning Conversion Trend',
        xaxis_title='Date',
        yaxis_title='Conversions',
        height=450,
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )

    return fig

def create_morning_roi_comparison(morning_analysis, overall_analysis):
    """Create comparison of morning vs overall ROI by channel"""

    # Get channel data
    morning_channels = morning_analysis['morning_by_channel']

    # Calculate overall ROI by channel
    overall_channels = overall_analysis['channel']

    # Merge for comparison
    comparison = pd.merge(
        morning_channels[['channel', 'roi']].rename(columns={'roi': 'morning_roi'}),
        overall_channels[['channel', 'roi']].rename(columns={'roi': 'overall_roi'}),
        on='channel',
        how='inner'
    )

    fig = go.Figure()

    # Add bars for comparison
    for channel in comparison['channel']:
        row = comparison[comparison['channel'] == channel].iloc[0]
        fig.add_trace(go.Bar(
            x=['Morning', 'Overall'],
            y=[row['morning_roi'] * 100, row['overall_roi'] * 100],
            name=channel,
            text=[f"{row['morning_roi']*100:.1f}%", f"{row['overall_roi']*100:.1f}%"],
            textposition='outside'
        ))

    fig.update_layout(
        title='📊 Morning vs Overall ROI by Channel',
        xaxis_title='Time Period',
        yaxis_title='ROI (%)',
        height=450,
        barmode='group',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )

    return fig

# ═══════════════════════════════════════════════════════════════
# DATA LOADING FUNCTIONS (From original code, adapted)
# ═══════════════════════════════════════════════════════════════

COLUMN_ALIASES = {
    "date": ["date", "Date", "transaction_date", "Transaction Date", "transaction date"],
    "channel": ["channel", "Channel", "marketing_channel", "Marketing Channel", "marketing channel"],
    "customer_type": ["customer_type", "Customer Type", "customer type", "customertype", "type"],
    "ad_spend": ["ad_spend", "Ad Spend", "ad spend", "spend", "Spend"],
    "conversions": ["conversions", "Conversions", "orders", "Orders"],
    "revenue": ["revenue", "Revenue", "sales", "Sales"],
    "service_category": ["service_category", "Service Category", "service category", "category"],
}

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
    if "email" in s: return "Email"  # Add email if present
    return str(x).strip()

def normalize_customer_type(x):
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()
    if "new" in s: return "New"
    if "return" in s: return "Returning"
    return str(x).strip()

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

        return df

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None

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

    agg["roi"] = np.where(agg["spend"] > 0, (agg["revenue"] - agg["spend"]) / agg["spend"], 0)
    agg["roas"] = np.where(agg["spend"] > 0, agg["revenue"] / agg["spend"], 0)
    agg["cpa"] = np.where(agg["conversions"] > 0, agg["spend"] / agg["conversions"], 0)

    return agg.sort_values("roi", ascending=False)

# ═══════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════

st.markdown('<div class="morning-title">🌅 Café Aroma Morning Customer Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="morning-subtitle">Targeting Weekday Morning Customer Acquisition via Flyer & Email Campaigns</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Morning Analytics Settings")

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

    # Morning campaign settings
    st.markdown("### 🎯 Morning Campaign Targets")
    target_rate = st.slider("Target Morning Conversion Rate (%)", 10, 50, 25)
    TARGET_MORNING_CONVERSION_RATE = target_rate / 100

    target_conversions = st.number_input("Target Weekday Morning Conversions", 50, 500, 100)
    TARGET_WEEKDAY_MORNING_CONVERSION = target_conversions

    st.divider()

    if st.button("🔄 Refresh Analysis", use_container_width=True, type="primary"):
        st.rerun()

    st.divider()
    st.caption("🌅 Morning Analytics v1.0 | Designed for Café Aroma")

# Load data
@st.cache_data(show_spinner=False)
def load_all_data(file_source, sheet_name):
    df = load_excel(file_source, sheet_name)
    if df is None:
        return None

    # Perform morning analysis
    morning_analysis = analyze_morning_performance(df)
    channel_analysis = analyze_morning_by_channel(df)
    hourly_analysis = analyze_morning_by_hour(df)
    weekday_analysis = analyze_morning_by_weekday(df)
    overall_channel = calc_by_channel(df)

    return {
        'df': df,
        'morning_analysis': morning_analysis,
        'channel_analysis': channel_analysis,
        'hourly_analysis': hourly_analysis,
        'weekday_analysis': weekday_analysis,
        'overall_channel': overall_channel,
        'recommendations': get_morning_recommendations(morning_analysis)
    }

with st.spinner("🌅 Analyzing morning customer data..."):
    data = load_all_data(file_src, sheet_val)

if data is None:
    st.error("❌ Failed to load data. Check file & settings.")
    st.stop()

# Extract data
df = data['df']
morning_analysis = data['morning_analysis']
channel_analysis = data['channel_analysis']
hourly_analysis = data['hourly_analysis']
weekday_analysis = data['weekday_analysis']
recommendations = data['recommendations']

# Morning Banner
st.markdown(f'''
<div class="morning-banner">
<div style="font-size: 1.8rem; font-weight: 700; margin-bottom: 1rem;">🎯 Morning Customer Acquisition Strategy</div>
<div style="font-size: 1.1rem; margin: 0.7rem 0;">
• <strong>Target:</strong> Increase weekday morning (6 AM - 12 PM) conversions by 30%
</div>
<div style="font-size: 1.1rem; margin: 0.7rem 0;">
• <strong>Primary Channels:</strong> Flyer distribution & Email marketing
</div>
<div style="font-size: 1.1rem; margin: 0.7rem 0;">
• <strong>Current Morning Conversion Rate:</strong> {morning_analysis["morning_conversion_rate"]:.1%} vs Target: {TARGET_MORNING_CONVERSION_RATE:.0%}
</div>
</div>
''', unsafe_allow_html=True)

# Morning KPIs
st.markdown("### 📊 Morning Performance Dashboard")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "🌅 Morning Conversions",
        f"{morning_analysis['morning_conversions']:,.0f}",
        f"{morning_analysis['morning_conversion_rate']:.1%} of total"
    )

with col2:
    st.metric(
        "📈 Weekday Morning",
        f"{morning_analysis['weekday_morning_conversions']:,.0f}",
        f"{morning_analysis['weekday_morning_conversion_rate']:.1%} of total"
    )

with col3:
    st.metric(
        "💰 Morning ROI",
        f"{morning_analysis['morning_roi']*100:.1f}%",
        "vs target 120%"
    )

with col4:
    st.metric(
        "📊 Morning CPA",
        f"SAR {morning_analysis['morning_cpa']:.0f}",
        "Cost per acquisition"
    )

with col5:
    st.metric(
        "🎯 Morning ROAS",
        f"{morning_analysis['morning_roas']:.2f}x",
        "Return on ad spend"
    )

st.divider()

# Tabs for detailed analysis
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Channel Performance",
    "🕒 Hourly Analysis",
    "📅 Weekday Patterns",
    "🎯 Recommendations"
])

# TAB 1: Channel Performance
with tab1:
    st.markdown("### 📊 Morning Performance by Channel")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="morning-target">🎯 Focus on Flyer & Email for Morning Campaigns</div>', unsafe_allow_html=True)

        # Morning channel performance
        morning_channels = channel_analysis['morning_by_channel']
        st.dataframe(
            morning_channels[[
                'channel', 'conversions', 'revenue', 'spend', 'roi', 'cpa'
            ]].style.format({
                'conversions': '{:.0f}',
                'revenue': 'SAR {:,.0f}',
                'spend': 'SAR {:,.0f}',
                'roi': '{:.1%}',
                'cpa': 'SAR {:.0f}'
            }).highlight_max(subset=['roi'], color='lightgreen'),
            use_container_width=True,
            height=400
        )

    with col_b:
        # Create morning channel performance chart
        chart = create_morning_channel_performance(morning_channels)
        st.plotly_chart(chart, use_container_width=True)

        # Flyer-specific analysis
        if 'Flyer' in morning_channels['channel'].values:
            flyer_data = morning_channels[morning_channels['channel'] == 'Flyer'].iloc[0]
            st.markdown(f'''
            <div class="morning-card">
            <h3>📢 Flyer Campaign Analysis</h3>
            <ul>
            <li><b>Morning Conversions:</b> {flyer_data['conversions']:.0f}</li>
            <li><b>Morning CPA:</b> SAR {flyer_data['cpa']:.0f}</li>
            <li><b>Morning ROI:</b> {flyer_data['roi']:.1%}</li>
            </ul>
            <div class="morning-recommendation">
            💡 <b>Recommendation:</b> Distribute flyers near offices 7-9 AM with "early bird" coffee discounts
            </div>
            </div>
            ''', unsafe_allow_html=True)

# TAB 2: Hourly Analysis
with tab2:
    st.markdown("### 🕒 Performance by Hour of Day")

    col_a, col_b = st.columns(2)

    with col_a:
        # Morning hours performance
        morning_hours = hourly_analysis[hourly_analysis['hour'].isin(MORNING_HOURS)]

        fig_hourly = go.Figure()

        fig_hourly.add_trace(go.Bar(
            x=morning_hours['hour'],
            y=morning_hours['conversions'],
            name='Conversions',
            marker_color=COLORS['morning_primary'],
            text=morning_hours['conversions'],
            textposition='outside'
        ))

        fig_hourly.add_trace(go.Scatter(
            x=morning_hours['hour'],
            y=morning_hours['new_customers'],
            name='New Customers',
            yaxis='y2',
            mode='lines+markers',
            line=dict(color=COLORS['morning_success'], width=3)
        ))

        fig_hourly.update_layout(
            title='📈 Morning Hours Performance (6 AM - 12 PM)',
            xaxis_title='Hour',
            yaxis_title='Conversions',
            yaxis2=dict(
                title='New Customers',
                overlaying='y',
                side='right'
            ),
            height=450,
            xaxis=dict(tickmode='array', tickvals=MORNING_HOURS)
        )

        st.plotly_chart(fig_hourly, use_container_width=True)

    with col_b:
        # Best performing hours
        top_hours = hourly_analysis.sort_values('conversions', ascending=False).head(5)

        st.markdown("### 🥇 Top Performing Hours")
        for idx, row in top_hours.iterrows():
            hour_label = f"{int(row['hour'])}:00"
            if row['hour'] in MORNING_HOURS:
                hour_label += " 🌅"

            st.markdown(f'''
            <div class="time-slot-card">
            <div style="font-size: 1.2rem; font-weight: 700; color: {COLORS['morning_primary']};">{hour_label}</div>
            <div style="font-size: 0.9rem; color: #666;">{row['conversions']:.0f} conversions | SAR {row['cpa']:.0f} CPA</div>
            </div>
            ''', unsafe_allow_html=True)

        # CPA by hour
        st.markdown("### 💰 Cost Analysis by Hour")

        hourly_cpa = hourly_analysis[['hour', 'cpa', 'conversions']]
        hourly_cpa = hourly_cpa.sort_values('hour')

        fig_cpa = go.Figure()
        fig_cpa.add_trace(go.Scatter(
            x=hourly_cpa['hour'],
            y=hourly_cpa['cpa'],
            mode='lines+markers',
            name='CPA',
            line=dict(color=COLORS['morning_warning'], width=3),
            fill='tozeroy',
            fillcolor='rgba(255, 107, 107, 0.1)'
        ))

        fig_cpa.update_layout(
            title='💵 Cost per Acquisition by Hour',
            xaxis_title='Hour',
            yaxis_title='CPA (SAR)',
            height=300,
            showlegend=False
        )

        st.plotly_chart(fig_cpa, use_container_width=True)

# TAB 3: Weekday Patterns
with tab3:
    st.markdown("### 📅 Weekday Morning Patterns")

    # Heatmap
    heatmap_chart = create_morning_heatmap(weekday_analysis)
    st.plotly_chart(heatmap_chart, use_container_width=True)

    # Weekday vs Weekend comparison
    col_a, col_b = st.columns(2)

    with col_a:
        # Calculate weekday vs weekend performance
        df['is_weekday'] = df['date'].dt.day_name().isin(WEEKDAYS)
        weekday_performance = df.groupby('is_weekday').agg({
            'conversions': 'sum',
            'revenue': 'sum',
            'ad_spend': 'sum'
        }).reset_index()

        weekday_performance['cpa'] = weekday_performance['ad_spend'] / weekday_performance['conversions']
        weekday_performance['roi'] = (weekday_performance['revenue'] - weekday_performance['ad_spend']) / weekday_performance['ad_spend']

        fig_weekday = go.Figure()

        fig_weekday.add_trace(go.Bar(
            x=['Weekdays', 'Weekends'],
            y=weekday_performance['conversions'],
            name='Conversions',
            marker_color=[COLORS['morning_primary'], COLORS['morning_info']],
            text=weekday_performance['conversions'],
            textposition='outside'
        ))

        fig_weekday.update_layout(
            title='📊 Weekday vs Weekend Performance',
            xaxis_title='Day Type',
            yaxis_title='Conversions',
            height=400,
            showlegend=False
        )

        st.plotly_chart(fig_weekday, use_container_width=True)

    with col_b:
        # Trend analysis
        trend_chart = create_morning_trend_chart(df)
        st.plotly_chart(trend_chart, use_container_width=True)

# TAB 4: Recommendations
with tab4:
    st.markdown("### 🎯 Action Plan for Morning Customer Growth")

    # Priority recommendations
    high_priority = [r for r in recommendations if r['priority'] == 'high']
    medium_priority = [r for r in recommendations if r['priority'] == 'medium']

    if high_priority:
        st.markdown("### 🔴 High Priority Actions")
        for rec in high_priority:
            st.markdown(f'''
            <div class="morning-card">
            <h3>{rec['title']}</h3>
            <p>{rec['action']}</p>
            <ul>
            {"".join([f"<li>{step}</li>" for step in rec['steps']])}
            </ul>
            </div>
            ''', unsafe_allow_html=True)

    if medium_priority:
        st.markdown("### 🟡 Medium Priority Actions")
        for rec in medium_priority:
            st.markdown(f'''
            <div class="morning-card">
            <h3>{rec['title']}</h3>
            <p>{rec['action']}</p>
            <ul>
            {"".join([f"<li>{step}</li>" for step in rec['steps']])}
            </ul>
            </div>
            ''', unsafe_allow_html=True)

    # 4-Week Action Plan
    st.markdown("### 📅 4-Week Morning Campaign Plan")

    weeks = [
        {"Week": 1, "Focus": "Flyer Distribution", "Actions": ["Design morning-specific flyers", "Target office areas 7-9 AM", "Track response rate"]},
        {"Week": 2, "Focus": "Email Campaign", "Actions": ["Build morning subscriber list", "Send 'Early Bird' offers", "A/B test subject lines"]},
        {"Week": 3, "Focus": "Promotion Testing", "Actions": ["Test different discounts", "Measure conversion lift", "Optimize timing"]},
        {"Week": 4, "Focus": "Scale & Analyze", "Actions": ["Double down on winners", "Calculate ROI impact", "Plan next month"]}
    ]

    for week in weeks:
        with st.expander(f"Week {week['Week']}: {week['Focus']}", expanded=(week['Week']==1)):
            st.markdown(f"**Primary Focus:** {week['Focus']}")
            st.markdown("**Key Actions:**")
            for action in week['Actions']:
                st.markdown(f"- {action}")

    # Expected outcomes
    st.markdown("### 📈 Expected Outcomes")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown(f'''
        <div class="morning-target">
        <div style="font-size: 1.5rem; color: {COLORS['morning_success']};">+30%</div>
        <div>Weekday Morning Conversions</div>
        </div>
        ''', unsafe_allow_html=True)

    with col_b:
        st.markdown(f'''
        <div class="morning-target">
        <div style="font-size: 1.5rem; color: {COLORS['morning_success']};">-20%</div>
        <div>Morning CPA Reduction</div>
        </div>
        ''', unsafe_allow_html=True)

    with col_c:
        st.markdown(f'''
        <div class="morning-target">
        <div style="font-size: 1.5rem; color: {COLORS['morning_success']};">+25%</div>
        <div>New Morning Customers</div>
        </div>
        ''', unsafe_allow_html=True)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 1.5rem;'>
<p style='font-size: 1.2rem; font-weight: 700;'>🌅 Café Aroma Morning Customer Analytics v1.0</p>
<p style='font-size: 0.9rem;'>Designed to increase weekday morning customers using targeted flyer & email campaigns</p>
<p style='font-size: 0.8rem; color: #888;'>Target: +30% morning conversions in 30 days</p>
</div>
""", unsafe_allow_html=True)