# visualization.py
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Optional, Dict, List

# Assuming config.py is in the same directory
from config import COLORS, LOG_LEVEL_STYLES

# Helper to get color from config safely
def _get_level_color(level: str) -> str:
    level_upper = str(level).upper()
    return LOG_LEVEL_STYLES.get(level_upper, LOG_LEVEL_STYLES["DEFAULT"])["color"]

# --- Dashboard Charts ---

def create_log_level_pie_chart(log_df: pd.DataFrame, theme: Optional[str]) -> Optional[go.Figure]:
    """Generates the log level distribution pie chart."""
    if "level" not in log_df.columns or log_df["level"].empty:
        return None

    level_counts = log_df["level"].astype(str).str.upper().value_counts().reset_index()
    level_counts.columns = ["Level", "Count"]

    # Ensure all levels in the data have a color
    level_color_map = {lvl: _get_level_color(lvl) for lvl in level_counts["Level"].unique()}

    fig = px.pie(level_counts, values='Count', names='Level',
                 color='Level', color_discrete_map=level_color_map,
                 hole=0.4, template=theme)
    fig.update_traces(textposition='inside', textinfo='percent+label', pull=0.02,
                      marker_line=dict(color='#FFFFFF', width=1))
    fig.update_layout(margin=dict(t=30, b=0, l=0, r=0), height=350, showlegend=True,
                      legend=dict(orientation="h", yanchor="bottom", y=-0.1))
    return fig

def create_top_modules_error_bar_chart(log_df: pd.DataFrame, theme: Optional[str]) -> Optional[go.Figure]:
    """Generates the bar chart for top modules by error count."""
    if "level" not in log_df.columns or "module" not in log_df.columns:
        return None

    error_df = log_df[log_df['level'].astype(str).str.upper() == 'ERROR']
    if error_df.empty:
        return None

    module_errors = error_df['module'].value_counts().head(10).reset_index()
    module_errors.columns = ['Module', 'Error Count']

    fig = px.bar(module_errors, x='Error Count', y='Module', orientation='h',
                 color='Error Count', color_continuous_scale=px.colors.sequential.Reds,
                 template=theme, text_auto=True)
    fig.update_layout(margin=dict(t=30, b=20, l=10, r=10), height=350,
                      yaxis={'categoryorder':'total ascending'}, xaxis_title="Number of Errors", yaxis_title=None)
    fig.update_traces(marker_line_width=0)
    return fig

def create_log_timeline_chart(log_df: pd.DataFrame, theme: Optional[str]) -> Optional[go.Figure]:
    """Generates the stacked bar chart showing log counts over time by level."""
    if 'datetime' not in log_df.columns or not log_df['datetime'].notna().any():
        return None

    time_df_dash = log_df.dropna(subset=['datetime']).copy()
    if time_df_dash.empty: return None

    if 'level' not in time_df_dash.columns: time_df_dash['level'] = 'INFO'
    time_df_dash['level'] = time_df_dash['level'].astype(str).fillna('INFO').str.upper()

    # Handle timezone if present - convert to UTC naive for aggregation
    if time_df_dash['datetime'].dt.tz is not None:
        time_df_dash['hour'] = time_df_dash['datetime'].dt.tz_convert('UTC').dt.tz_localize(None).dt.floor('h')
    else:
        time_df_dash['hour'] = time_df_dash['datetime'].dt.floor('h')

    try:
        hourly_counts = pd.pivot_table(time_df_dash, index='hour', columns='level', aggfunc='size', fill_value=0)

        fig_timeline = go.Figure()
        # Define order and colors consistently
        level_order = sorted(hourly_counts.columns.tolist(), key=lambda x: (
             0 if x == "INFO" else 1 if x == "DEBUG" else 2 if x == "WARNING" else 3 if x == "ERROR" else 4 if x == "PARSE_ERROR" else 5
        )) # Prioritize common levels
        level_color_map = {lvl: _get_level_color(lvl) for lvl in hourly_counts.columns}

        for level in level_order:
             fig_timeline.add_trace(go.Bar(
                 x=hourly_counts.index, y=hourly_counts[level],
                 name=level, marker_color=level_color_map.get(level)
             ))
        fig_timeline.update_layout(
            barmode='stack', template=theme,
            margin=dict(t=30, b=20, l=20, r=20), height=350,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis_title="Time", yaxis_title='Log Count per Hour',
            xaxis_range=[hourly_counts.index.min(), hourly_counts.index.max()] if not hourly_counts.empty else None
        )
        return fig_timeline
    except Exception as pivot_err:
         print(f"Could not generate timeline chart due to pivot error: {pivot_err}")
         return None

# --- Advanced Visualization Charts ---

def create_latency_histogram(latency_df: pd.DataFrame, theme: Optional[str]) -> Optional[go.Figure]:
    """Generates latency histogram with log scale Y."""
    if latency_df.empty: return None
    fig = px.histogram(latency_df, x="latency", nbins=50, title="Latency Distribution (Log Scale Y)",
                       template=theme, log_y=True, marginal="box")
    fig.update_layout(height=350, margin=dict(t=40, b=0, l=0, r=0), xaxis_title="Latency (ms)")
    return fig

def create_latency_boxplot_by_module(latency_df: pd.DataFrame, theme: Optional[str]) -> Optional[go.Figure]:
    """Generates latency boxplot by module with log scale X."""
    if latency_df.empty or 'module' not in latency_df.columns or latency_df['module'].nunique() == 0:
        return None

    if 'level' not in latency_df.columns: latency_df['level'] = 'UNKNOWN'
    latency_df['level'] = latency_df['level'].astype(str).fillna('UNKNOWN').str.upper()

    top_modules_lat = latency_df['module'].value_counts().head(15).index.tolist()
    level_color_map = {lvl: _get_level_color(lvl) for lvl in latency_df['level'].unique()}

    fig = px.box(latency_df[latency_df['module'].isin(top_modules_lat)],
                 x="latency", y="module", color="level", points=False,
                 title="Latency by Module (Top 15, Log Scale X)", template=theme, log_x=True,
                 category_orders={"module": top_modules_lat[::-1]},
                 color_discrete_map=level_color_map)
    fig.update_layout(height=400, margin=dict(t=40, b=0, l=0, r=0), xaxis_title="Latency (ms, Log Scale)", yaxis_title=None)
    return fig

def create_status_code_bar_chart(status_df: pd.DataFrame, theme: Optional[str]) -> Optional[go.Figure]:
    """Generates bar chart for top status codes."""
    if status_df.empty: return None
    status_counts = status_df["status_code_str"].value_counts().reset_index().head(20)
    status_counts.columns = ["Status Code", "Count"]
    fig = px.bar(status_counts, x="Status Code", y="Count", title="Top 20 Status Codes", template=theme, text_auto=True)
    fig.update_layout(xaxis={'type': 'category'}, height=350, margin=dict(t=40, b=0, l=0, r=0))
    return fig

def create_status_code_pie_chart(status_df: pd.DataFrame, theme: Optional[str]) -> Optional[go.Figure]:
    """Generates pie chart for status code categories."""
    if status_df.empty: return None
    category_counts = status_df["status_category"].value_counts().reset_index()
    category_counts.columns = ["Category", "Count"]
    category_color_map = {"Success (2xx)": COLORS["success"], "Redirect (3xx)": COLORS["info"], "Client Error (4xx)": COLORS["warning"], "Server Error (5xx)": COLORS["error"], "Other": COLORS["secondary"]}
    fig = px.pie(category_counts, values="Count", names="Category", title="Status Code Categories",
                 color="Category", color_discrete_map=category_color_map, template=theme, hole=0.3)
    fig.update_layout(height=350, margin=dict(t=40, b=0, l=0, r=0), legend=dict(orientation="h", yanchor="bottom", y=-0.1))
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

def create_status_module_heatmap(status_df: pd.DataFrame, theme: Optional[str]) -> Optional[go.Figure]:
    """Generates heatmap of status categories vs. modules."""
    if status_df.empty or 'module' not in status_df.columns or status_df['module'].nunique() == 0:
        return None

    top_modules_stat = status_df['module'].value_counts().head(15).index.tolist()
    status_by_module = pd.crosstab(
        status_df[status_df['module'].isin(top_modules_stat)]["module"],
        status_df["status_category"]
    )
    cat_order = ['Success (2xx)', 'Redirect (3xx)', 'Client Error (4xx)', 'Server Error (5xx)', 'Other']
    status_by_module = status_by_module.reindex(columns=cat_order, fill_value=0)

    if status_by_module.empty: return None

    fig = px.imshow(status_by_module.T, title="Status Categories by Module (Top 15)",
                    color_continuous_scale=px.colors.sequential.Blues,
                    template=theme, text_auto=True, aspect="auto")
    fig.update_layout(height=max(400, len(cat_order)*50), xaxis_title="Module", yaxis_title="Status Category")
    return fig


def create_module_volume_bar_chart(log_df: pd.DataFrame, theme: Optional[str]) -> Optional[go.Figure]:
    """Generates bar chart for top modules by log volume."""
    if log_df.empty or 'module' not in log_df.columns: return None
    module_counts_viz = log_df['module'].value_counts().reset_index().head(15)
    module_counts_viz.columns = ['Module', 'Total Logs']
    fig = px.bar(module_counts_viz, x='Total Logs', y='Module', orientation='h',
                 title="Top 15 Modules by Log Volume", template=theme, text_auto=True)
    fig.update_layout(yaxis={'categoryorder':'total ascending'}, height=400, margin=dict(t=40, b=0, l=0, r=0), yaxis_title=None)
    return fig


def create_module_level_treemap(log_df: pd.DataFrame, theme: Optional[str]) -> Optional[go.Figure]:
    """Generates treemap of module activity by log level."""
    if log_df.empty or 'module' not in log_df.columns or 'level' not in log_df.columns:
        return None

    df_for_treemap = log_df.dropna(subset=['module', 'level']).copy()
    if df_for_treemap.empty: return None

    df_for_treemap['level'] = df_for_treemap['level'].astype(str).str.upper()
    treemap_data = df_for_treemap.groupby(['module', 'level']).size().reset_index(name='count')

    level_color_map = {lvl: _get_level_color(lvl) for lvl in treemap_data['level'].unique()}

    fig = px.treemap(treemap_data, path=[px.Constant("All Modules"), 'module', 'level'], values='count',
                     title='Module Activity by Log Level', template=theme, height=400,
                     color='level', color_discrete_map=level_color_map)
    fig.update_layout(margin=dict(t=50, b=0, l=0, r=0))
    fig.update_traces(textinfo='label+value+percent root')
    return fig

def create_detailed_timeseries_chart(time_agg: pd.DataFrame, agg_freq: str, theme: Optional[str]) -> Optional[go.Figure]:
    """Generates combined bar (volume) and line (error rate) chart."""
    if time_agg.empty: return None

    fig = go.Figure()
    fig.add_trace(go.Bar(x=time_agg['datetime'], y=time_agg['total_logs'], name='Total Logs', marker_color=COLORS['primary'], opacity=0.7))
    fig.add_trace(go.Scatter(x=time_agg['datetime'], y=time_agg['error_rate'], name='Error Rate (%)', yaxis='y2', mode='lines+markers', line=dict(color=COLORS['error'])))

    fig.update_layout(
        title=f'Log Volume & Error Rate (Aggregated by {agg_freq})', template=theme, height=400,
        yaxis=dict(title='Total Logs'),
        yaxis2=dict(title='Error Rate (%)', overlaying='y', side='right', range=[0, 100], showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=50, b=0, l=0, r=0),
        hovermode="x unified"
    )
    return fig

def create_day_hour_heatmap(log_df: pd.DataFrame, theme: Optional[str]) -> Optional[go.Figure]:
    """Generates heatmap of log volume by Day of Week vs Hour of Day."""
    if log_df.empty or 'datetime' not in log_df.columns: return None

    temp_df = log_df.copy()
    # Ensure datetime operations work regardless of timezone
    if temp_df['datetime'].dt.tz is not None:
        temp_df['day_of_week'] = temp_df['datetime'].dt.tz_convert('UTC').dt.day_name()
        temp_df['hour_of_day'] = temp_df['datetime'].dt.tz_convert('UTC').dt.hour
    else:
        temp_df['day_of_week'] = temp_df['datetime'].dt.day_name()
        temp_df['hour_of_day'] = temp_df['datetime'].dt.hour

    day_hour_counts = temp_df.groupby(['day_of_week', 'hour_of_day']).size().unstack(fill_value=0)
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_hour_counts = day_hour_counts.reindex(day_order, fill_value=0)

    fig = px.imshow(day_hour_counts, title="Log Volume: Day of Week vs Hour of Day",
                    template=theme, color_continuous_scale='Blues', aspect='auto', text_auto=True)
    fig.update_layout(height=350, margin=dict(t=40, b=0, l=0, r=0), xaxis_title="Hour of Day", yaxis_title=None)
    return fig

def create_errors_by_day_chart(log_df: pd.DataFrame, theme: Optional[str]) -> Optional[go.Figure]:
    """Generates bar chart of total errors by day of the week."""
    if log_df.empty or 'datetime' not in log_df.columns or 'level' not in log_df.columns:
        return None

    temp_df = log_df.copy()
    temp_df['level'] = temp_df['level'].astype(str).str.upper()
    if temp_df['datetime'].dt.tz is not None:
        temp_df['day_of_week'] = temp_df['datetime'].dt.tz_convert('UTC').dt.day_name()
    else:
        temp_df['day_of_week'] = temp_df['datetime'].dt.day_name()

    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    error_by_day = temp_df[temp_df['level']=='ERROR'].groupby('day_of_week').size().reindex(day_order, fill_value=0).reset_index()
    error_by_day.columns = ['Day', 'Error Count']

    fig = px.bar(error_by_day, x='Day', y='Error Count', title='Total Errors by Day of Week', template=theme, text_auto=True)
    fig.update_layout(height=350, margin=dict(t=40, b=0, l=0, r=0))
    return fig

# --- AI Analysis Tab Charts ---
def create_cluster_distribution_chart(clusters_summary: List[Dict], theme: Optional[str]) -> Optional[go.Figure]:
    """Bar chart showing log count per cluster."""
    if not clusters_summary: return None
    cluster_dist_df = pd.DataFrame([{'Cluster': str(cs['cluster_id']), 'Count': cs['total_logs']} for cs in clusters_summary])
    fig = px.bar(cluster_dist_df, x="Cluster", y="Count", title="Log Count by Cluster",
                 template=theme, color="Cluster", color_discrete_sequence=px.colors.qualitative.Pastel, text_auto=True)
    fig.update_layout(xaxis={'type': 'category'}, height=300, margin=dict(t=30, b=0, l=0, r=0), showlegend=False)
    return fig

def create_cluster_error_rate_chart(clusters_summary: List[Dict], theme: Optional[str]) -> Optional[go.Figure]:
    """Bar chart showing error rate per cluster."""
    if not clusters_summary: return None
    error_df = pd.DataFrame([{'Cluster': str(cs['cluster_id']), 'Error Rate': cs['error_rate']} for cs in clusters_summary])
    fig = px.bar(error_df, x="Cluster", y="Error Rate", title="Error Rate (%) by Cluster",
                 template=theme, color="Error Rate", color_continuous_scale=px.colors.sequential.Reds, range_y=[0,100], text_auto=".1f")
    fig.update_layout(xaxis={'type': 'category'}, height=300, margin=dict(t=30, b=0, l=0, r=0))
    fig.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
    return fig

def create_comparative_error_count_chart(error_profiles: List[Dict], theme: Optional[str]) -> Optional[go.Figure]:
    """Bar chart comparing error counts across clusters that have errors."""
    if not error_profiles: return None
    comp_err_df = pd.DataFrame([{'Cluster': str(ep['cluster_id']), 'Error Count': ep['error_count']} for ep in error_profiles])
    fig = px.bar(comp_err_df, x="Cluster", y="Error Count", title="Error Count per Cluster (Error Clusters Only)",
                 template=theme, color="Error Count", color_continuous_scale=px.colors.sequential.OrRd, text_auto=True)
    fig.update_layout(xaxis={'type': 'category'}, height=300, margin=dict(t=30, b=0, l=0, r=0))
    return fig
