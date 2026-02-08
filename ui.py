"""
PREMIUM INTERACTIVE CUSTOMER LOYALTY DASHBOARD - FULL VERSION
===============================================================================
Built with Plotly Dash 2.x - Modern, Interactive, Web-Based Analytics Platform
Complete with all features and visualizations
===============================================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime
import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# ============================================================================
# PREMIUM DESIGN SYSTEM
# ============================================================================

class DesignTheme:
    """Premium Design Theme for Dash Dashboard"""

    # Modern Color Palette
    COLORS = {
        'primary': '#2563EB',
        'primary_dark': '#1E40AF',
        'success': '#10B981',
        'success_dark': '#059669',
        'warning': '#F59E0B',
        'warning_dark': '#D97706',
        'danger': '#EF4444',
        'danger_dark': '#DC2626',
        'purple': '#8B5CF6',
        'purple_dark': '#7C3AED',
        'cyan': '#06B6D4',
        'cyan_dark': '#0891B2',
        'pink': '#EC4899',
        'pink_dark': '#DB2777',
        'indigo': '#6366F1',
        'indigo_dark': '#4F46E5',

        'background': '#F9FAFB',
        'card': '#FFFFFF',
        'text': '#1F2937',
        'text_light': '#6B7280',
        'border': '#E5E7EB',
    }

    CARD_STYLE = {
        'backgroundColor': '#FFFFFF',
        'borderRadius': '16px',
        'padding': '28px',
        'boxShadow': '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
        'marginBottom': '24px',
        'border': '1px solid #E5E7EB'
    }

    KPI_CARD_STYLE = {
        'borderRadius': '16px',
        'padding': '32px 24px',
        'boxShadow': '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
        'textAlign': 'center',
        'border': 'none',
        'height': '100%',
        'transition': 'transform 0.2s',
        'cursor': 'pointer'
    }


THEME = DesignTheme()


# ============================================================================
# DATA GENERATION
# ============================================================================

def create_sample_data(n_samples=250):
    """Create comprehensive sample loyalty data"""
    np.random.seed(42)

    # Generate dates
    dates = []
    for month in range(1, 7):
        for _ in range(n_samples // 6):
            day = np.random.randint(1, 28)
            dates.append(datetime(2024, month, day))

    while len(dates) < n_samples:
        dates.append(datetime(2024, np.random.randint(1, 7), np.random.randint(1, 28)))

    dates = sorted(dates[:n_samples])

    # Generate revenue with seasonality
    base_revenue = np.random.exponential(400, n_samples) + 100
    seasonality = np.sin(np.linspace(0, 2 * np.pi, n_samples)) * 100 + 100
    revenues = base_revenue + seasonality
    revenues = np.clip(revenues, 50, 3000)

    # Generate costs and other metrics
    cost_ratios = np.random.beta(2, 2, n_samples) * 0.3 + 0.5
    costs = revenues * cost_ratios
    ad_spend = costs * np.random.uniform(0.3, 0.6, n_samples)

    # Satisfaction based on profit margins
    profit_margins = (revenues - costs) / revenues
    satisfaction_base = (profit_margins * 5 + 5).clip(1, 10)
    satisfaction = satisfaction_base + np.random.normal(0, 1, n_samples)
    satisfaction = satisfaction.clip(1, 10).round().astype(int)

    # Conversions
    conversions = (ad_spend / 50 + np.random.poisson(3, n_samples)).clip(0, 25)

    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Revenue': np.round(revenues, 2),
        'Cost': np.round(costs, 2),
        'Ad_Spend': np.round(ad_spend, 2),
        'Conversions': conversions.astype(int),
        'Customer_Satisfaction': satisfaction,
        'Campaign_Type': np.random.choice(
            ['Premium Service', 'Standard Service', 'Special Offer', 'Loyalty Program', 'Welcome Package'],
            n_samples,
            p=[0.25, 0.35, 0.2, 0.1, 0.1]
        ),
        'Customer_Status': np.random.choice(['New', 'Returning'], n_samples, p=[0.35, 0.65]),
        'Channel': np.random.choice(
            ['Call Center', 'Mobile App', 'Website', 'Store Visit', 'Social Media', 'Email'],
            n_samples,
            p=[0.2, 0.25, 0.2, 0.15, 0.15, 0.05]
        ),
        'Region': np.random.choice(
            ['North', 'South', 'East', 'West', 'Central'],
            n_samples,
            p=[0.25, 0.2, 0.2, 0.2, 0.15]
        ),
        'Service_Type': np.random.choice(['Premium', 'Standard', 'Basic'], n_samples, p=[0.3, 0.5, 0.2]),
    })

    # Calculate derived metrics
    df['Profit'] = df['Revenue'] - df['Cost']
    df['Profit_Margin'] = np.where(df['Revenue'] > 0, (df['Profit'] / df['Revenue']) * 100, 0)
    df['CLV_Score'] = df.apply(
        lambda x: x['Revenue'] * 2.5 if x['Customer_Status'] == 'Returning' else x['Revenue'] * 1.2,
        axis=1
    )
    df['Ad_ROI'] = np.where(
        df['Ad_Spend'] > 0,
        ((df['Revenue'] - df['Ad_Spend']) / df['Ad_Spend']) * 100,
        0
    )
    df['Month_Name'] = df['Date'].dt.strftime('%b %Y')
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.isocalendar().week

    return df


# Generate data
print("📊 Generating sample data...")
df = create_sample_data(250)
print(f"✅ Generated {len(df)} customer records")

# Calculate key metrics
total_revenue = df['Revenue'].sum()
total_profit = df['Profit'].sum()
total_cost = df['Cost'].sum()
total_ad_spend = df['Ad_Spend'].sum()
avg_satisfaction = df['Customer_Satisfaction'].mean()
total_conversions = df['Conversions'].sum()
retention_rate = (df['Customer_Status'] == 'Returning').sum() / len(df) * 100
avg_clv = df['CLV_Score'].mean()
avg_roi = df['Ad_ROI'].mean()
profit_margin = (total_profit / total_revenue * 100) if total_revenue > 0 else 0

# Growth calculation
monthly_rev = df.groupby('Month_Name')['Revenue'].sum()
if len(monthly_rev) >= 2:
    revenue_growth = ((monthly_rev.iloc[-1] - monthly_rev.iloc[0]) / monthly_rev.iloc[0] * 100)
else:
    revenue_growth = 0

# ============================================================================
# DASH APP INITIALIZATION
# ============================================================================

app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ]
)

app.title = "Customer Loyalty Dashboard"


# ============================================================================
# HELPER FUNCTIONS FOR CHARTS
# ============================================================================

def create_revenue_chart():
    """Create revenue and profit trend chart"""
    monthly_data = df.groupby('Month_Name').agg({
        'Revenue': 'sum',
        'Profit': 'sum',
        'Cost': 'sum'
    }).reset_index()

    fig = go.Figure()

    # Revenue bars
    fig.add_trace(go.Bar(
        x=monthly_data['Month_Name'],
        y=monthly_data['Revenue'],
        name='Revenue',
        marker_color=THEME.COLORS['primary'],
        hovertemplate='<b>%{x}</b><br>Revenue: $%{y:,.0f}<extra></extra>',
        text=monthly_data['Revenue'].apply(lambda x: f'${x / 1000:.1f}K'),
        textposition='outside',
        textfont=dict(size=11, color=THEME.COLORS['text'])
    ))

    # Profit bars
    fig.add_trace(go.Bar(
        x=monthly_data['Month_Name'],
        y=monthly_data['Profit'],
        name='Profit',
        marker_color=THEME.COLORS['success'],
        hovertemplate='<b>%{x}</b><br>Profit: $%{y:,.0f}<extra></extra>',
        text=monthly_data['Profit'].apply(lambda x: f'${x / 1000:.1f}K'),
        textposition='outside',
        textfont=dict(size=11, color=THEME.COLORS['text'])
    ))

    # Trend line
    x_numeric = list(range(len(monthly_data)))
    z = np.polyfit(x_numeric, monthly_data['Revenue'], 1)
    p = np.poly1d(z)

    fig.add_trace(go.Scatter(
        x=monthly_data['Month_Name'],
        y=p(x_numeric),
        name=f'Trend {"↗" if z[0] > 0 else "↘"}',
        mode='lines',
        line=dict(color=THEME.COLORS['danger'], width=3, dash='dash'),
        hovertemplate='<b>Trend</b><br>$%{y:,.0f}<extra></extra>'
    ))

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='system-ui, -apple-system, sans-serif', size=13, color=THEME.COLORS['text']),
        # السطر 253-254
        yaxis=dict(showgrid=True, gridcolor='#F3F4F6', title='Amount ($)',
                   title_font=dict(size=14, color=THEME.COLORS['text_light'])),  # ← لاحظ الفاصلة
        #                                                                      ^^ قوس + فاصلة

        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, bgcolor='rgba(255,255,255,0.9)'),
        hovermode='x unified',
        margin=dict(l=50, r=20, t=40, b=50),
        height=380
    )

    return fig


def create_segmentation_chart():
    """Create customer segmentation chart"""
    segment_data = df.groupby('Customer_Status').agg({
        'Revenue': ['mean', 'count'],
        'CLV_Score': 'mean',
        'Customer_Satisfaction': 'mean'
    }).reset_index()

    segment_data.columns = ['Status', 'Avg_Revenue', 'Count', 'Avg_CLV', 'Avg_Satisfaction']

    fig = go.Figure()

    # Revenue bars
    fig.add_trace(go.Bar(
        x=segment_data['Status'],
        y=segment_data['Avg_Revenue'],
        name='Avg Revenue',
        marker_color=THEME.COLORS['primary'],
        text=segment_data['Avg_Revenue'].apply(lambda x: f'${x:.0f}'),
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Avg Revenue: $%{y:,.0f}<extra></extra>'
    ))

    # CLV bars (scaled)
    fig.add_trace(go.Bar(
        x=segment_data['Status'],
        y=segment_data['Avg_CLV'] / 10,
        name='Avg CLV (÷10)',
        marker_color=THEME.COLORS['cyan'],
        text=(segment_data['Avg_CLV'] / 10).apply(lambda x: f'{x:.0f}'),
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Avg CLV: $%{customdata:,.0f}<extra></extra>',
        customdata=segment_data['Avg_CLV']
    ))

    # Satisfaction bars (scaled)
    fig.add_trace(go.Bar(
        x=segment_data['Status'],
        y=segment_data['Avg_Satisfaction'] * 100,
        name='Satisfaction (×100)',
        marker_color=THEME.COLORS['warning'],
        text=segment_data['Avg_Satisfaction'].apply(lambda x: f'{x:.1f}'),
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Avg Satisfaction: %{customdata:.1f}/10<extra></extra>',
        customdata=segment_data['Avg_Satisfaction']
    ))

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='system-ui, -apple-system, sans-serif', size=13, color=THEME.COLORS['text']),
        xaxis=dict(showgrid=False, title=''),
        yaxis=dict(showgrid=True, gridcolor='#F3F4F6', title='Value',
                   title_font=dict(size=14, color=THEME.COLORS['text_light'])),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, bgcolor='rgba(255,255,255,0.9)'),
        hovermode='x unified',
        margin=dict(l=50, r=20, t=40, b=50),
        height=380,
        barmode='group',
        bargap=0.15
    )

    # Add count annotations
    for i, row in segment_data.iterrows():
        fig.add_annotation(
            x=row['Status'],
            y=segment_data[['Avg_Revenue', 'Avg_CLV', 'Avg_Satisfaction']].max().max() / 10 * 1.15,
            text=f"n={int(row['Count'])}",
            showarrow=False,
            font=dict(size=12, color='white', family='system-ui'),
            bgcolor=THEME.COLORS['warning'],
            bordercolor='white',
            borderwidth=2,
            borderpad=6,
            opacity=0.95
        )

    return fig


def create_service_chart():
    """Create service quality performance chart"""
    service_data = df.groupby('Service_Type').agg({
        'Revenue': 'mean',
        'Customer_Satisfaction': 'mean',
        'Conversions': 'mean',
        'CLV_Score': 'mean'
    }).sort_values('Revenue', ascending=True).reset_index()

    fig = go.Figure()

    # Normalize metrics for stacking
    metrics_norm = {}
    for col in ['Revenue', 'Customer_Satisfaction', 'Conversions', 'CLV_Score']:
        min_val, max_val = service_data[col].min(), service_data[col].max()
        if max_val > min_val:
            metrics_norm[col] = (service_data[col] - min_val) / (max_val - min_val)
        else:
            metrics_norm[col] = [0.5] * len(service_data)

    colors = [THEME.COLORS['primary'], THEME.COLORS['warning'], THEME.COLORS['success'], THEME.COLORS['purple']]
    labels = ['Revenue', 'Satisfaction', 'Conversions', 'CLV Score']

    for i, (col, label) in enumerate(zip(metrics_norm.keys(), labels)):
        fig.add_trace(go.Bar(
            y=service_data['Service_Type'],
            x=metrics_norm[col],
            name=label,
            orientation='h',
            marker_color=colors[i],
            hovertemplate=f'<b>%{{y}}</b><br>{label}: %{{customdata:.0f}}<extra></extra>',
            customdata=service_data[col]
        ))

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='system-ui, -apple-system, sans-serif', size=13, color=THEME.COLORS['text']),
        xaxis=dict(showgrid=True, gridcolor='#F3F4F6', title='Normalized Score',
                   title_font=dict(size=14, color=THEME.COLORS['text_light'])),
        yaxis=dict(showgrid=False, title=''),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, bgcolor='rgba(255,255,255,0.9)'),
        margin=dict(l=80, r=20, t=40, b=50),
        height=380,
        barmode='stack'
    )

    return fig


def create_channel_chart():
    """Create channel effectiveness bubble chart"""
    channel_data = df.groupby('Channel').agg({
        'Revenue': 'mean',
        'Customer_Satisfaction': 'mean',
        'Conversions': 'sum',
        'Ad_ROI': 'mean'
    }).reset_index()

    fig = go.Figure()

    # Size scaling
    sizes = channel_data['Revenue']
    size_min, size_max = sizes.min(), sizes.max()
    if size_max > size_min:
        sizes_norm = ((sizes - size_min) / (size_max - size_min) * 50) + 15
    else:
        sizes_norm = [30] * len(sizes)

    fig.add_trace(go.Scatter(
        x=channel_data['Customer_Satisfaction'],
        y=channel_data['Ad_ROI'],
        mode='markers+text',
        marker=dict(
            size=sizes_norm,
            color=channel_data['Conversions'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Conversions', title_font=dict(size=12)),
            line=dict(width=2, color='white'),
            opacity=0.8
        ),
        text=channel_data['Channel'],
        textposition='top center',
        textfont=dict(size=10, color=THEME.COLORS['text'], family='system-ui'),
        hovertemplate='<b>%{text}</b><br>Satisfaction: %{x:.1f}<br>ROI: %{y:.0f}%<br>Revenue: $%{customdata:,.0f}<extra></extra>',
        customdata=channel_data['Revenue']
    ))

    # Add quadrant lines
    x_mean = channel_data['Customer_Satisfaction'].mean()
    y_mean = channel_data['Ad_ROI'].mean()

    fig.add_hline(y=y_mean, line_dash='dash', line_color=THEME.COLORS['danger'], opacity=0.4, line_width=2)
    fig.add_vline(x=x_mean, line_dash='dash', line_color=THEME.COLORS['danger'], opacity=0.4, line_width=2)

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='system-ui, -apple-system, sans-serif', size=13, color=THEME.COLORS['text']),
        xaxis=dict(showgrid=True, gridcolor='#F3F4F6', title='Avg Satisfaction',
                   title_font=dict(size=14, color=THEME.COLORS['text_light'])),
        yaxis=dict(showgrid=True, gridcolor='#F3F4F6', title='ROI (%)',
                   title_font=dict(size=14, color=THEME.COLORS['text_light'])),
        margin=dict(l=50, r=20, t=40, b=50),
        height=380
    )

    return fig


def create_regional_chart():
    """Create regional distribution chart"""
    regional_data = df.groupby('Region')['Revenue'].sum().sort_values(ascending=True).reset_index()

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=regional_data['Region'],
        x=regional_data['Revenue'],
        orientation='h',
        marker=dict(
            color=regional_data['Revenue'],
            colorscale='RdYlGn',
            showscale=False,
            line=dict(width=2, color='white')
        ),
        text=regional_data['Revenue'].apply(lambda x: f'${x:,.0f}'),
        textposition='outside',
        textfont=dict(size=11, color=THEME.COLORS['text']),
        hovertemplate='<b>%{y}</b><br>Total Revenue: $%{x:,.0f}<extra></extra>'
    ))

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='system-ui, -apple-system, sans-serif', size=13, color=THEME.COLORS['text']),
        xaxis=dict(showgrid=True, gridcolor='#F3F4F6', title='Total Revenue ($)',
                   title_font=dict(size=14, color=THEME.COLORS['text_light'])),
        yaxis=dict(showgrid=False, title=''),
        margin=dict(l=80, r=20, t=20, b=50),
        height=380
    )

    return fig


def create_satisfaction_chart():
    """Create satisfaction distribution chart"""
    fig = go.Figure()

    statuses = df['Customer_Status'].unique()
    colors = [THEME.COLORS['primary'], THEME.COLORS['success']]

    for i, status in enumerate(statuses):
        status_data = df[df['Customer_Status'] == status]['Customer_Satisfaction']

        fig.add_trace(go.Violin(
            y=status_data,
            name=status,
            box_visible=True,
            meanline_visible=True,
            fillcolor=colors[i],
            opacity=0.7,
            line_color=colors[i],
            hovertemplate='<b>%{fullData.name}</b><br>Score: %{y}<extra></extra>'
        ))

    # Add overall mean line
    fig.add_hline(
        y=avg_satisfaction,
        line_dash='dash',
        line_color=THEME.COLORS['danger'],
        line_width=3,
        opacity=0.7,
        annotation_text=f'Overall Mean: {avg_satisfaction:.1f}',
        annotation_position='right'
    )

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='system-ui, -apple-system, sans-serif', size=13, color=THEME.COLORS['text']),
        xaxis=dict(showgrid=False, title=''),
        yaxis=dict(showgrid=True, gridcolor='#F3F4F6', title='Satisfaction Score',
                   title_font=dict(size=14, color=THEME.COLORS['text_light']), range=[0, 11]),
        margin=dict(l=50, r=20, t=40, b=50),
        height=380,
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, bgcolor='rgba(255,255,255,0.9)')
    )

    return fig


def create_clv_chart():
    """Create CLV analysis chart"""
    clv_data = df.groupby('Customer_Status').agg({
        'CLV_Score': ['mean', 'median'],
        'Revenue': 'mean'
    }).reset_index()

    clv_data.columns = ['Status', 'Mean_CLV', 'Median_CLV', 'Avg_Revenue']

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=clv_data['Status'],
        y=clv_data['Mean_CLV'],
        name='Avg CLV',
        marker_color=THEME.COLORS['purple'],
        text=clv_data['Mean_CLV'].apply(lambda x: f'${x:.0f}'),
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Avg CLV: $%{y:,.0f}<extra></extra>'
    ))

    fig.add_trace(go.Bar(
        x=clv_data['Status'],
        y=clv_data['Avg_Revenue'],
        name='Avg Revenue',
        marker_color=THEME.COLORS['primary'],
        text=clv_data['Avg_Revenue'].apply(lambda x: f'${x:.0f}'),
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Avg Revenue: $%{y:,.0f}<extra></extra>'
    ))

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='system-ui, -apple-system, sans-serif', size=13, color=THEME.COLORS['text']),
        xaxis=dict(showgrid=False, title=''),
        yaxis=dict(showgrid=True, gridcolor='#F3F4F6', title='Value ($)',
                   title_font=dict(size=14, color=THEME.COLORS['text_light'])),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, bgcolor='rgba(255,255,255,0.9)'),
        hovermode='x unified',
        margin=dict(l=50, r=20, t=40, b=50),
        height=380,
        barmode='group'
    )

    return fig


def create_trend_chart():
    """Create revenue & satisfaction trend chart"""
    monthly_data = df.groupby('Month_Name').agg({
        'Revenue': 'sum',
        'Customer_Satisfaction': 'mean',
        'Conversions': 'sum'
    }).reset_index()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Revenue line
    fig.add_trace(
        go.Scatter(
            x=monthly_data['Month_Name'],
            y=monthly_data['Revenue'],
            name='Revenue',
            mode='lines+markers',
            line=dict(color=THEME.COLORS['primary'], width=3),
            marker=dict(size=10, color=THEME.COLORS['primary'], line=dict(width=2, color='white')),
            hovertemplate='<b>%{x}</b><br>Revenue: $%{y:,.0f}<extra></extra>',
            fill='tozeroy',
            fillcolor='rgba(37, 99, 235, 0.1)'
        ),
        secondary_y=False
    )

    # Satisfaction line
    fig.add_trace(
        go.Scatter(
            x=monthly_data['Month_Name'],
            y=monthly_data['Customer_Satisfaction'],
            name='Satisfaction',
            mode='lines+markers',
            line=dict(color=THEME.COLORS['warning'], width=3),
            marker=dict(size=10, color=THEME.COLORS['warning'], line=dict(width=2, color='white'), symbol='square'),
            hovertemplate='<b>%{x}</b><br>Satisfaction: %{y:.1f}/10<extra></extra>'
        ),
        secondary_y=True
    )

    fig.update_xaxes(showgrid=False, title='', tickangle=-30)
    fig.update_yaxes(
        title_text='Revenue ($)',
        title_font=dict(size=14, color=THEME.COLORS['primary']),
        showgrid=True,
        gridcolor='#F3F4F6',
        secondary_y=False
    )
    fig.update_yaxes(
        title_text='Satisfaction Score',
        title_font=dict(size=14, color=THEME.COLORS['warning']),
        showgrid=False,
        secondary_y=True
    )

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='system-ui, -apple-system, sans-serif', size=13, color=THEME.COLORS['text']),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, bgcolor='rgba(255,255,255,0.9)'),
        hovermode='x unified',
        margin=dict(l=50, r=50, t=40, b=50),
        height=380
    )

    return fig


def create_conversion_funnel():
    """Create conversion funnel by service type"""
    conv_data = df.groupby('Service_Type')['Conversions'].sum().sort_values(ascending=False).reset_index()

    fig = go.Figure()

    colors = [THEME.COLORS['primary'], THEME.COLORS['cyan'], THEME.COLORS['warning']]

    fig.add_trace(go.Funnel(
        y=conv_data['Service_Type'],
        x=conv_data['Conversions'],
        textinfo='value+percent initial',
        textfont=dict(size=13, color='white', family='system-ui'),
        marker=dict(color=colors[:len(conv_data)]),
        hovertemplate='<b>%{y}</b><br>Conversions: %{x:,.0f}<extra></extra>'
    ))

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='system-ui, -apple-system, sans-serif', size=13, color=THEME.COLORS['text']),
        margin=dict(l=20, r=20, t=20, b=20),
        height=380
    )

    return fig


# ============================================================================
# LAYOUT
# ============================================================================

app.layout = html.Div([

    # Header Section
    html.Div([
        html.Div([
            html.H1('Customer Loyalty & Service Excellence',
                    style={
                        'fontSize': '42px',
                        'fontWeight': '800',
                        'color': THEME.COLORS['text'],
                        'marginBottom': '12px',
                        'letterSpacing': '-1px',
                        'background': f'linear-gradient(135deg, {THEME.COLORS["primary"]} 0%, {THEME.COLORS["purple"]} 100%)',
                        'WebkitBackgroundClip': 'text',
                        'WebkitTextFillColor': 'transparent',
                        'backgroundClip': 'text'
                    }),
            html.P('Comprehensive Analytics Dashboard for Customer Relationship Management',
                   style={
                       'fontSize': '17px',
                       'color': THEME.COLORS['text_light'],
                       'marginBottom': '0',
                       'fontStyle': 'italic',
                       'fontWeight': '500'
                   }),
            html.P(
                f'📊 {len(df)} Customer Records • 📅 {df["Date"].min().strftime("%b %Y")} - {df["Date"].max().strftime("%b %Y")}',
                style={
                    'fontSize': '14px',
                    'color': THEME.COLORS['text_light'],
                    'marginTop': '8px',
                    'marginBottom': '0'
                })
        ], style={'textAlign': 'center', 'padding': '48px 20px 32px 20px'})
    ], style={'backgroundColor': '#FFFFFF', 'borderBottom': f'2px solid {THEME.COLORS["border"]}',
              'boxShadow': '0 1px 3px rgba(0,0,0,0.05)'}),

    # Main Content
    html.Div([

        # KPI Cards Row
        html.Div([
            # Revenue Card
            html.Div([
                html.Div([
                    html.Div('💰', style={'fontSize': '52px', 'marginBottom': '20px',
                                         'filter': 'drop-shadow(0 2px 4px rgba(0,0,0,0.1))'}),
                    html.H2(f'${total_revenue / 1000:.1f}K',
                            style={'fontSize': '38px', 'fontWeight': '800', 'color': 'white', 'margin': '0 0 10px 0',
                                   'textShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
                    html.P('TOTAL REVENUE',
                           style={'fontSize': '13px', 'fontWeight': '700', 'color': 'rgba(255,255,255,0.95)',
                                  'margin': '0 0 10px 0', 'textTransform': 'uppercase', 'letterSpacing': '1px'}),
                    html.Div([
                        html.Span('↗' if revenue_growth >= 0 else '↘',
                                  style={'fontSize': '18px', 'marginRight': '4px'}),
                        html.Span(f'{revenue_growth:+.1f}% growth', style={'fontSize': '13px', 'fontWeight': '500'})
                    ], style={'color': 'rgba(255,255,255,0.9)'})
                ], style={**THEME.KPI_CARD_STYLE,
                          'background': f'linear-gradient(135deg, {THEME.COLORS["primary"]} 0%, {THEME.COLORS["primary_dark"]} 100%)',
                          'color': 'white'})
            ], style={'width': '16.66%', 'display': 'inline-block', 'padding': '0 12px', 'verticalAlign': 'top'}),

            # Profit Card
            html.Div([
                html.Div([
                    html.Div('📊', style={'fontSize': '52px', 'marginBottom': '20px',
                                         'filter': 'drop-shadow(0 2px 4px rgba(0,0,0,0.1))'}),
                    html.H2(f'${total_profit / 1000:.1f}K',
                            style={'fontSize': '38px', 'fontWeight': '800', 'color': 'white', 'margin': '0 0 10px 0',
                                   'textShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
                    html.P('NET PROFIT',
                           style={'fontSize': '13px', 'fontWeight': '700', 'color': 'rgba(255,255,255,0.95)',
                                  'margin': '0 0 10px 0', 'textTransform': 'uppercase', 'letterSpacing': '1px'}),
                    html.Span(f'{profit_margin:.1f}% margin',
                              style={'fontSize': '13px', 'color': 'rgba(255,255,255,0.9)', 'fontWeight': '500'})
                ], style={**THEME.KPI_CARD_STYLE,
                          'background': f'linear-gradient(135deg, {THEME.COLORS["success"]} 0%, {THEME.COLORS["success_dark"]} 100%)',
                          'color': 'white'})
            ], style={'width': '16.66%', 'display': 'inline-block', 'padding': '0 12px', 'verticalAlign': 'top'}),

            # Retention Card
            html.Div([
                html.Div([
                    html.Div('🔄', style={'fontSize': '52px', 'marginBottom': '20px',
                                         'filter': 'drop-shadow(0 2px 4px rgba(0,0,0,0.1))'}),
                    html.H2(f'{retention_rate:.1f}%',
                            style={'fontSize': '38px', 'fontWeight': '800', 'color': 'white', 'margin': '0 0 10px 0',
                                   'textShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
                    html.P('RETENTION RATE',
                           style={'fontSize': '13px', 'fontWeight': '700', 'color': 'rgba(255,255,255,0.95)',
                                  'margin': '0 0 10px 0', 'textTransform': 'uppercase', 'letterSpacing': '1px'}),
                    html.Span(f'{int((df["Customer_Status"] == "Returning").sum())} loyal customers',
                              style={'fontSize': '13px', 'color': 'rgba(255,255,255,0.9)', 'fontWeight': '500'})
                ], style={**THEME.KPI_CARD_STYLE,
                          'background': f'linear-gradient(135deg, {THEME.COLORS["purple"]} 0%, {THEME.COLORS["purple_dark"]} 100%)',
                          'color': 'white'})
            ], style={'width': '16.66%', 'display': 'inline-block', 'padding': '0 12px', 'verticalAlign': 'top'}),

            # Satisfaction Card
            html.Div([
                html.Div([
                    html.Div('⭐', style={'fontSize': '52px', 'marginBottom': '20px',
                                         'filter': 'drop-shadow(0 2px 4px rgba(0,0,0,0.1))'}),
                    html.H2(f'{avg_satisfaction:.1f}/10',
                            style={'fontSize': '38px', 'fontWeight': '800', 'color': 'white', 'margin': '0 0 10px 0',
                                   'textShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
                    html.P('AVG SATISFACTION',
                           style={'fontSize': '13px', 'fontWeight': '700', 'color': 'rgba(255,255,255,0.95)',
                                  'margin': '0 0 10px 0', 'textTransform': 'uppercase', 'letterSpacing': '1px'}),
                    html.Span(f'{len(df)} total customers',
                              style={'fontSize': '13px', 'color': 'rgba(255,255,255,0.9)', 'fontWeight': '500'})
                ], style={**THEME.KPI_CARD_STYLE,
                          'background': f'linear-gradient(135deg, {THEME.COLORS["warning"]} 0%, {THEME.COLORS["warning_dark"]} 100%)',
                          'color': 'white'})
            ], style={'width': '16.66%', 'display': 'inline-block', 'padding': '0 12px', 'verticalAlign': 'top'}),

            # CLV Card
            html.Div([
                html.Div([
                    html.Div('💎', style={'fontSize': '52px', 'marginBottom': '20px',
                                         'filter': 'drop-shadow(0 2px 4px rgba(0,0,0,0.1))'}),
                    html.H2(f'${avg_clv:.0f}',
                            style={'fontSize': '38px', 'fontWeight': '800', 'color': 'white', 'margin': '0 0 10px 0',
                                   'textShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
                    html.P('CUSTOMER CLV',
                           style={'fontSize': '13px', 'fontWeight': '700', 'color': 'rgba(255,255,255,0.95)',
                                  'margin': '0 0 10px 0', 'textTransform': 'uppercase', 'letterSpacing': '1px'}),
                    html.Span('Lifetime value',
                              style={'fontSize': '13px', 'color': 'rgba(255,255,255,0.9)', 'fontWeight': '500'})
                ], style={**THEME.KPI_CARD_STYLE,
                          'background': f'linear-gradient(135deg, {THEME.COLORS["cyan"]} 0%, {THEME.COLORS["cyan_dark"]} 100%)',
                          'color': 'white'})
            ], style={'width': '16.66%', 'display': 'inline-block', 'padding': '0 12px', 'verticalAlign': 'top'}),

            # ROI Card
            html.Div([
                html.Div([
                    html.Div('📈', style={'fontSize': '52px', 'marginBottom': '20px',
                                         'filter': 'drop-shadow(0 2px 4px rgba(0,0,0,0.1))'}),
                    html.H2(f'{avg_roi:.0f}%',
                            style={'fontSize': '38px', 'fontWeight': '800', 'color': 'white', 'margin': '0 0 10px 0',
                                   'textShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
                    html.P('ROI',
                           style={'fontSize': '13px', 'fontWeight': '700', 'color': 'rgba(255,255,255,0.95)',
                                  'margin': '0 0 10px 0', 'textTransform': 'uppercase', 'letterSpacing': '1px'}),
                    html.Span(f'${total_ad_spend / 1000:.1f}K invested',
                              style={'fontSize': '13px', 'color': 'rgba(255,255,255,0.9)', 'fontWeight': '500'})
                ], style={**THEME.KPI_CARD_STYLE,
                          'background': f'linear-gradient(135deg, {THEME.COLORS["pink"]} 0%, {THEME.COLORS["pink_dark"]} 100%)',
                          'color': 'white'})
            ], style={'width': '16.66%', 'display': 'inline-block', 'padding': '0 12px', 'verticalAlign': 'top'}),

        ], style={'marginBottom': '36px', 'marginTop': '32px'}),

        # Charts Row 1
        html.Div([
            html.Div([
                html.Div([
                    html.H3('Revenue & Profit Performance',
                            style={'fontSize': '22px', 'fontWeight': '700', 'color': THEME.COLORS['text'],
                                   'marginBottom': '24px', 'letterSpacing': '-0.3px'}),
                    dcc.Graph(figure=create_revenue_chart(), config={'displayModeBar': False})
                ], style=THEME.CARD_STYLE)
            ], style={'width': '50%', 'display': 'inline-block', 'padding': '0 12px', 'verticalAlign': 'top'}),

            html.Div([
                html.Div([
                    html.H3('Customer Segmentation Analysis',
                            style={'fontSize': '22px', 'fontWeight': '700', 'color': THEME.COLORS['text'],
                                   'marginBottom': '24px', 'letterSpacing': '-0.3px'}),
                    dcc.Graph(figure=create_segmentation_chart(), config={'displayModeBar': False})
                ], style=THEME.CARD_STYLE)
            ], style={'width': '50%', 'display': 'inline-block', 'padding': '0 12px', 'verticalAlign': 'top'}),
        ]),

        # Charts Row 2
        html.Div([
            html.Div([
                html.Div([
                    html.H3('Service Quality Performance',
                            style={'fontSize': '22px', 'fontWeight': '700', 'color': THEME.COLORS['text'],
                                   'marginBottom': '24px', 'letterSpacing': '-0.3px'}),
                    dcc.Graph(figure=create_service_chart(), config={'displayModeBar': False})
                ], style=THEME.CARD_STYLE)
            ], style={'width': '33.33%', 'display': 'inline-block', 'padding': '0 12px', 'verticalAlign': 'top'}),

            html.Div([
                html.Div([
                    html.H3('Channel Effectiveness',
                            style={'fontSize': '22px', 'fontWeight': '700', 'color': THEME.COLORS['text'],
                                   'marginBottom': '24px', 'letterSpacing': '-0.3px'}),
                    dcc.Graph(figure=create_channel_chart(), config={'displayModeBar': False})
                ], style=THEME.CARD_STYLE)
            ], style={'width': '33.33%', 'display': 'inline-block', 'padding': '0 12px', 'verticalAlign': 'top'}),

            html.Div([
                html.Div([
                    html.H3('Regional Distribution',
                            style={'fontSize': '22px', 'fontWeight': '700', 'color': THEME.COLORS['text'],
                                   'marginBottom': '24px', 'letterSpacing': '-0.3px'}),
                    dcc.Graph(figure=create_regional_chart(), config={'displayModeBar': False})
                ], style=THEME.CARD_STYLE)
            ], style={'width': '33.33%', 'display': 'inline-block', 'padding': '0 12px', 'verticalAlign': 'top'}),
        ]),

        # Charts Row 3
        html.Div([
            html.Div([
                html.Div([
                    html.H3('Satisfaction Distribution',
                            style={'fontSize': '22px', 'fontWeight': '700', 'color': THEME.COLORS['text'],
                                   'marginBottom': '24px', 'letterSpacing': '-0.3px'}),
                    dcc.Graph(figure=create_satisfaction_chart(), config={'displayModeBar': False})
                ], style=THEME.CARD_STYLE)
            ], style={'width': '33.33%', 'display': 'inline-block', 'padding': '0 12px', 'verticalAlign': 'top'}),

            html.Div([
                html.Div([
                    html.H3('Customer Lifetime Value (CLV)',
                            style={'fontSize': '22px', 'fontWeight': '700', 'color': THEME.COLORS['text'],
                                   'marginBottom': '24px', 'letterSpacing': '-0.3px'}),
                    dcc.Graph(figure=create_clv_chart(), config={'displayModeBar': False})
                ], style=THEME.CARD_STYLE)
            ], style={'width': '33.33%', 'display': 'inline-block', 'padding': '0 12px', 'verticalAlign': 'top'}),

            html.Div([
                html.Div([
                    html.H3('Conversion Funnel',
                            style={'fontSize': '22px', 'fontWeight': '700', 'color': THEME.COLORS['text'],
                                   'marginBottom': '24px', 'letterSpacing': '-0.3px'}),
                    dcc.Graph(figure=create_conversion_funnel(), config={'displayModeBar': False})
                ], style=THEME.CARD_STYLE)
            ], style={'width': '33.33%', 'display': 'inline-block', 'padding': '0 12px', 'verticalAlign': 'top'}),
        ]),

        # Charts Row 4 - Trend Analysis
        html.Div([
            html.Div([
                html.Div([
                    html.H3('Revenue & Satisfaction Trend Analysis',
                            style={'fontSize': '22px', 'fontWeight': '700', 'color': THEME.COLORS['text'],
                                   'marginBottom': '24px', 'letterSpacing': '-0.3px'}),
                    dcc.Graph(figure=create_trend_chart(), config={'displayModeBar': False})
                ], style=THEME.CARD_STYLE)
            ], style={'width': '100%', 'padding': '0 12px'}),
        ]),

        # Insights Section
        html.Div([
            html.Div([
                html.H3('Strategic Insights & Recommendations',
                        style={'fontSize': '24px', 'fontWeight': '700', 'color': THEME.COLORS['text'],
                               'marginBottom': '28px', 'letterSpacing': '-0.3px'}),

                html.Div([
                    # Insight 1
                    html.Div([
                        html.Div('✓',
                                 style={'color': THEME.COLORS['success'], 'fontSize': '24px', 'marginRight': '16px',
                                        'fontWeight': 'bold'}),
                        html.Div([
                            html.Span('Retention Rate: ',
                                      style={'fontWeight': '700', 'fontSize': '16px', 'color': THEME.COLORS['text']}),
                            html.Span(f'{retention_rate:.1f}% - ',
                                      style={'fontSize': '16px', 'color': THEME.COLORS['text']}),
                            html.Span('Excellent performance' if retention_rate > 60 else 'Needs improvement',
                                      style={'fontSize': '16px',
                                             'color': THEME.COLORS['success'] if retention_rate > 60 else THEME.COLORS[
                                                 'warning'], 'fontWeight': '600'})
                        ])
                    ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '20px', 'padding': '16px 20px',
                              'backgroundColor': '#F0FDF4', 'borderRadius': '12px',
                              'border': f'2px solid {THEME.COLORS["success"]}', 'transition': 'all 0.2s'}),

                    # Insight 2
                    html.Div([
                        html.Div('✓',
                                 style={'color': THEME.COLORS['primary'], 'fontSize': '24px', 'marginRight': '16px',
                                        'fontWeight': 'bold'}),
                        html.Div([
                            html.Span('Customer Lifetime Value: ',
                                      style={'fontWeight': '700', 'fontSize': '16px', 'color': THEME.COLORS['text']}),
                            html.Span(
                                f'Returning customers have {(df[df["Customer_Status"] == "Returning"]["CLV_Score"].mean() / df[df["Customer_Status"] == "New"]["CLV_Score"].mean()):.1f}x higher CLV - Focus on retention strategies',
                                style={'fontSize': '16px', 'color': THEME.COLORS['text']})
                        ])
                    ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '20px', 'padding': '16px 20px',
                              'backgroundColor': '#EFF6FF', 'borderRadius': '12px',
                              'border': f'2px solid {THEME.COLORS["primary"]}'}),

                    # Insight 3
                    html.Div([
                        html.Div('✓',
                                 style={'color': THEME.COLORS['warning'], 'fontSize': '24px', 'marginRight': '16px',
                                        'fontWeight': 'bold'}),
                        html.Div([
                            html.Span('Customer Satisfaction: ',
                                      style={'fontWeight': '700', 'fontSize': '16px', 'color': THEME.COLORS['text']}),
                            html.Span(f'{avg_satisfaction:.1f}/10 - ',
                                      style={'fontSize': '16px', 'color': THEME.COLORS['text']}),
                            html.Span('Good performance' if avg_satisfaction >= 7 else 'Requires immediate attention',
                                      style={'fontSize': '16px',
                                             'color': THEME.COLORS['success'] if avg_satisfaction >= 7 else
                                             THEME.COLORS['danger'], 'fontWeight': '600'})
                        ])
                    ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '20px', 'padding': '16px 20px',
                              'backgroundColor': '#FFFBEB', 'borderRadius': '12px',
                              'border': f'2px solid {THEME.COLORS["warning"]}'}),

                    # Insight 4
                    html.Div([
                        html.Div('→', style={'color': THEME.COLORS['purple'], 'fontSize': '24px', 'marginRight': '16px',
                                             'fontWeight': 'bold'}),
                        html.Div([
                            html.Span('Best Service Type: ',
                                      style={'fontWeight': '700', 'fontSize': '16px', 'color': THEME.COLORS['text']}),
                            html.Span(
                                f'{df.groupby("Service_Type")["Customer_Satisfaction"].mean().idxmax()} service has highest satisfaction - Scale this offering',
                                style={'fontSize': '16px', 'color': THEME.COLORS['text']})
                        ])
                    ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '20px', 'padding': '16px 20px',
                              'backgroundColor': '#FAF5FF', 'borderRadius': '12px',
                              'border': f'2px solid {THEME.COLORS["purple"]}'}),

                    # Insight 5
                    html.Div([
                        html.Div('→', style={'color': THEME.COLORS['cyan'], 'fontSize': '24px', 'marginRight': '16px',
                                             'fontWeight': 'bold'}),
                        html.Div([
                            html.Span('Top Channel: ',
                                      style={'fontWeight': '700', 'fontSize': '16px', 'color': THEME.COLORS['text']}),
                            html.Span(
                                f'{df.groupby("Channel")["Revenue"].mean().idxmax()} generates highest average revenue - Increase investment here',
                                style={'fontSize': '16px', 'color': THEME.COLORS['text']})
                        ])
                    ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '20px', 'padding': '16px 20px',
                              'backgroundColor': '#ECFEFF', 'borderRadius': '12px',
                              'border': f'2px solid {THEME.COLORS["cyan"]}'}),

                    # Insight 6
                    html.Div([
                        html.Div('→', style={'color': THEME.COLORS['pink'], 'fontSize': '24px', 'marginRight': '16px',
                                             'fontWeight': 'bold'}),
                        html.Div([
                            html.Span('Regional Opportunity: ',
                                      style={'fontWeight': '700', 'fontSize': '16px', 'color': THEME.COLORS['text']}),
                            html.Span(
                                f'{df.groupby("Region")["Revenue"].sum().idxmax()} region leads in revenue - Replicate success factors to other regions',
                                style={'fontSize': '16px', 'color': THEME.COLORS['text']})
                        ])
                    ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '20px', 'padding': '16px 20px',
                              'backgroundColor': '#FDF2F8', 'borderRadius': '12px',
                              'border': f'2px solid {THEME.COLORS["pink"]}'}),

                    # Action Items Header
                    html.Div([
                        html.H4('💡 Recommended Action Items:',
                                style={'fontSize': '20px', 'fontWeight': '700', 'color': THEME.COLORS['text'],
                                       'marginTop': '32px', 'marginBottom': '20px'})
                    ]),

                    # Actions
                    html.Div([
                        html.Li('Implement tiered loyalty program with exclusive benefits for returning customers',
                                style={'fontSize': '15px', 'color': THEME.COLORS['text'], 'marginBottom': '12px',
                                       'lineHeight': '1.6'}),
                        html.Li('Launch satisfaction improvement initiative targeting service quality gaps',
                                style={'fontSize': '15px', 'color': THEME.COLORS['text'], 'marginBottom': '12px',
                                       'lineHeight': '1.6'}),
                        html.Li('Develop special offers and promotions to convert new customers to loyal status',
                                style={'fontSize': '15px', 'color': THEME.COLORS['text'], 'marginBottom': '12px',
                                       'lineHeight': '1.6'}),
                        html.Li('Optimize marketing budget allocation based on channel ROI performance',
                                style={'fontSize': '15px', 'color': THEME.COLORS['text'], 'marginBottom': '12px',
                                       'lineHeight': '1.6'}),
                        html.Li('Establish customer referral program to leverage satisfied customer base',
                                style={'fontSize': '15px', 'color': THEME.COLORS['text'], 'marginBottom': '12px',
                                       'lineHeight': '1.6'}),
                        html.Li('Invest in staff training and service quality improvement programs',
                                style={'fontSize': '15px', 'color': THEME.COLORS['text'], 'lineHeight': '1.6'}),
                    ], style={'paddingLeft': '24px'})
                ])
            ], style={**THEME.CARD_STYLE, 'padding': '36px'})
        ], style={'padding': '0 12px', 'marginBottom': '32px'}),

    ], style={'maxWidth': '1800px', 'margin': '0 auto', 'padding': '0 24px 48px 24px'}),

    # Footer
    html.Div([
        html.Div([
            html.P(
                'Customer Loyalty & Service Excellence Dashboard • Powered by Plotly Dash • Real-time Interactive Analytics Platform',
                style={'textAlign': 'center', 'color': THEME.COLORS['text_light'], 'fontSize': '14px', 'margin': '0',
                       'padding': '28px', 'fontWeight': '500'}),
            html.P(f'Last Updated: {datetime.now().strftime("%B %d, %Y at %I:%M %p")} • Dashboard Version 2.0',
                   style={'textAlign': 'center', 'color': THEME.COLORS['text_light'], 'fontSize': '12px', 'margin': '0',
                          'paddingBottom': '28px'})
        ])
    ], style={'backgroundColor': '#FFFFFF', 'borderTop': f'2px solid {THEME.COLORS["border"]}', 'marginTop': '40px'}),

], style={
    'backgroundColor': THEME.COLORS['background'],
    'minHeight': '100vh',
    'fontFamily': 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'
})

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("🚀 PREMIUM INTERACTIVE DASHBOARD - FULL VERSION")
    print("=" * 80)
    print(f"\n✨ Starting Dash server...")
    print(f"📊 Dashboard will open at: http://127.0.0.1:8050")
    print(f"\n💡 Features:")
    print(f"   • 6 Interactive KPI Cards")
    print(f"   • 10 Advanced Visualizations")
    print(f"   • Real-time Data Analysis")
    print(f"   • Strategic Insights & Recommendations")
    print(f"   • Responsive Design")
    print(f"\n⚡ Press Ctrl+C to stop the server")
    print("=" * 80 + "\n")

    # Updated for Dash 2.x
    app.run(debug=True, host='127.0.0.1', port=8050)


    # أضف في آخر الملف بعد كل الدوال

    def save_dashboard_images():
        """Save all charts as images"""
        import plotly.io as pio

        # Create output folder
        import os
        output_dir = 'dashboard_images'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print("\n📸 Saving dashboard charts as images...")

        # Save each chart
        charts = {
            'revenue_chart.png': create_revenue_chart(),
            'segmentation_chart.png': create_segmentation_chart(),
            'service_chart.png': create_service_chart(),
            'channel_chart.png': create_channel_chart(),
            'regional_chart.png': create_regional_chart(),
            'satisfaction_chart.png': create_satisfaction_chart(),
            'clv_chart.png': create_clv_chart(),
            'trend_chart.png': create_trend_chart(),
            'conversion_funnel.png': create_conversion_funnel(),
        }

        for filename, fig in charts.items():
            filepath = os.path.join(output_dir, filename)
            pio.write_image(fig, filepath, width=1920, height=1080, scale=2)
            print(f"✅ Saved: {filepath}")

        print(f"\n✅ All charts saved to: {output_dir}/")

# في آخر الملف، غيّر السطر ده:

if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("🚀 PREMIUM INTERACTIVE DASHBOARD - FULL VERSION")
    print("=" * 80)
    print(f"\n✨ Starting Dash server...")
    print(f"📊 Dashboard will open at: http://127.0.0.1:8050")
    print(f"\n💡 Features:")
    print(f"   • 6 Interactive KPI Cards")
    print(f"   • 10 Advanced Visualizations")
    print(f"   • Real-time Data Analysis")
    print(f"   • Strategic Insights & Recommendations")
    print(f"   • Responsive Design")
    print(f"\n⚡ Press Ctrl+C to stop the server")
    print("=" * 80 + "\n")

    # أضف السطر ده لفتح المتصفح تلقائياً:
    import webbrowser

    webbrowser.open('http://127.0.0.1:8050')

    # Updated for Dash 2.x
    app.run(debug=True, host='127.0.0.1', port=8050)