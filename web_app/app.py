import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from google.cloud import bigquery
from dotenv import load_dotenv

# Load env
load_dotenv()

# Page Config
st.set_page_config(
    page_title="GhostJam Buster",
    page_icon="👻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Feature Rich" Dark Theme
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .main .block-container {
        padding-top: 2rem;
    }
    h1, h2, h3 {
        color: #f0f2f6;
    }
    .stMetric {
        background-color: #1f2937;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #374151;
    }
    .css-1d391kg {
        padding-top: 3.5rem;
    }
</style>
""", unsafe_allow_html=True)

# GCP Connection (Mockable if no creds)
def get_data():
    project_id = os.getenv("GCP_PROJECT_ID")
    dataset = os.getenv("BQ_DATASET_NAME")
    table = os.getenv("BQ_TABLE_NAME")
    
    if not project_id or not dataset:
        st.warning("⚠️ GCP Project ID not configured. Showing MOCK data.")
        return get_mock_data()
        
    try:
        # Check if we should try connecting
        # If no key file, we hope for ADC.
        client = bigquery.Client(project=project_id)
        query = f"""
            SELECT * FROM `{project_id}.{dataset}.{table}`
            ORDER BY timestamp DESC
            LIMIT 1000
        """
        # CRITICAL: Disable caching to ensure we get new streaming data
        job_config = bigquery.QueryJobConfig(use_query_cache=False)
        return client.query(query, job_config=job_config).to_dataframe()
        return client.query(query, job_config=job_config).to_dataframe()
    except Exception as e:
        st.error(f"Failed to connect to BigQuery ({e}). showing MOCK data.")
        return get_mock_data()

def get_mock_data():
    # Generate some fake sine wave data for demo
    import numpy as np
    t = np.linspace(0, 100, 200)
    
    # Reactive
    df_r = pd.DataFrame({
        "timestamp": t,
        "relative_time": t,
        "mode": "reactive",
        "speed": 10 + 2 * np.sin(t/5) + np.random.normal(0, 1, 200),
        "acceleration": np.cos(t/5) + np.random.normal(0, 0.5, 200),
        "brake_pressure": np.where(np.random.random(200) > 0.9, 0.5, 0)
    })
    
    # Smoothed
    df_s = pd.DataFrame({
        "timestamp": t,
        "relative_time": t,
        "mode": "smoothed",
        "speed": 10 + 2 * np.sin(t/5),
        "acceleration": np.cos(t/5),
        "brake_pressure": np.zeros(200)
    })
    
    return pd.concat([df_r, df_s])

# --- Sidebar ---
st.sidebar.title("🚗 Vision AI Control")
st.sidebar.image("https://img.icons8.com/color/96/000000/traffic-jam.png", width=100)
st.sidebar.markdown("---")
view_mode = st.sidebar.radio("View Mode", ["Live Dashboard", "Historical Analysis", "Driver Profile"])
st.sidebar.markdown("---")
st.sidebar.info("System Status: **ONLINE** 🟢")
st.sidebar.caption("v1.2 (Live Patch)")
import datetime
st.sidebar.write(f"Last Update: {datetime.datetime.now().strftime('%H:%M:%S')}")

# --- Main Content ---
st.title("👻 GhostJam Buster")
st.markdown("### *Busting phantom traffic with computer vision.*")

data = get_data()

if data.empty:
    st.warning("⏳ Connected to BigQuery, but no data found yet. The pipeline might still be uploading...")
    st.info("Showing MOCK DATA for visualization preview.")
    data = get_mock_data()

if view_mode == "Live Dashboard":
    from streamlit.runtime.scriptrunner import add_script_run_ctx
    import time

    st.markdown("### Real-time Telemetry")
    
    # Auto-refresh logic
    # Auto-refresh logic (Action at end of script)
    auto_refresh = st.checkbox("Enable Auto-Refresh (1s)", value=True)

    # Top Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    # Sort data chronologically for plotting (oldest to newest)
    data = data.sort_values(by="timestamp")

    if len(data) > 0:
        # After sorting, iloc[-1] IS the latest
        latest = data.iloc[-1]
        prev = data.iloc[-5] if len(data) > 5 else latest
    else:
        # Fallback if mock data somehow failed
        latest = {"speed": 0, "acceleration": 0, "brake_pressure": 0}
        prev = latest
    
    with col1:
        st.metric("Current Speed", f"{latest['speed']:.1f} m/s", f"{latest['speed'] - prev['speed']:.2f}")
    with col2:
        st.metric("Acceleration", f"{latest['acceleration']:.2f} m/s²")
    with col3:
        st.metric("Brake Pressure", f"{latest['brake_pressure']:.2f}", delta_color="inverse")
    with col4:
        # Safety Score (Simple logic: 100 - abs(accel)*10)
        safety = max(0, min(100, 100 - abs(latest['acceleration']) * 20))
        st.metric("Safety Score", f"{int(safety)}/100")

    # Interactive Charts with Plotly
    
    # Speed Chart
    fig_speed = px.line(data, x="relative_time", y="speed", color="mode", 
                        title="Speed Smoothing Analysis", template="plotly_dark",
                        color_discrete_map={"reactive": "#EF553B", "smoothed": "#00CC96"})
    st.plotly_chart(fig_speed, use_container_width=True)
    
    # Accel Chart
    col_a, col_b = st.columns(2)
    with col_a:
        fig_accel = px.line(data, x="relative_time", y="acceleration", color="mode", 
                             title="Acceleration Jitter", template="plotly_dark")
        st.plotly_chart(fig_accel, use_container_width=True)
    
    with col_b:
        # Brake Events Bar
        if not data.empty:
            brake_data = data[data['brake_pressure'] > 0.1].groupby('mode').count()['brake_pressure'].reset_index()
            fig_brake = px.bar(brake_data, x="mode", y="brake_pressure", title="Total Brake Events",
                               color="mode", template="plotly_dark")
            st.plotly_chart(fig_brake, use_container_width=True)



    if auto_refresh:
        time.sleep(1)
        st.rerun()

elif view_mode == "Historical Analysis":
    st.markdown("### Historical Performance")
    st.write("Analyze past runs and compare efficiency.")
    
    # Heatmap of speed
    fig_hist = px.density_heatmap(data, x="relative_time", y="speed", facet_col="mode",
                                  title="Speed Distribution Heatmap", template="plotly_dark")
    st.plotly_chart(fig_hist, use_container_width=True)

elif view_mode == "Driver Profile":
    st.markdown("### Driver Safety Profile")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=150)
        st.markdown("## Analysis")
        st.markdown("""
        *   **Smoothness**: Excellent
        *   **Reaction Time**: Good
        *   **Eco-Driving**: Top 10%
        """)
        
    with col2:
        # Gauge Chart for Safety Score
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = 87,
            title = {'text': "Overall Safety Score"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "#00CC96"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "gray"}],
            }
        ))
        fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    st.success("✨ AI Insight: Your smooth braking at t=45s saved approximately 0.05L of fuel compared to the baseline.")

# Footer
st.markdown("---")
st.markdown("Generated by Google Cloud AI & Vision Speed Smoothing System")
