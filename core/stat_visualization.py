import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Data Insight Tool", layout="wide")
st.title("🔬 Data Insights Dashboard")

# Sidebar setup
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = px.data.tips()
    st.warning("Using demo data. Upload your own CSV on the sidebar.")

# Metrics
m1, m2, m3 = st.columns(3)
m1.metric("Records", len(df))
m2.metric("Mean Value", round(df.iloc[:,0].mean(), 2))
m3.metric("Unique Items", df.iloc[:,1].nunique())

# Plotting
st.subheader("Data Visualization")
target_x = st.selectbox("Select X Axis", df.columns)
target_y = st.selectbox("Select Y Axis", df.columns)

fig = px.histogram(df, x=target_x, y=target_y, color=None, template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)