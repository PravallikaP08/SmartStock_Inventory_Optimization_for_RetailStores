import streamlit as st

st.set_page_config(page_title="SmartStock Inventory", layout="wide")

st.title("SmartStock Inventory Optimization for Retail Stores")

st.markdown("### Loading Dashboard...")

# import your milestone4 dashboard
from milestone4.dashboard import *
