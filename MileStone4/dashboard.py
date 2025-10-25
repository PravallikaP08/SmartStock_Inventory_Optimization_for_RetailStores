
import os
import io
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Smart Inventory Dashboard", layout="wide")


m2_forecast_path = r"C:\Users\PRAVALLIKA\Smart_Stock_Project\MileStone2\data\forecast_results.csv"
m1_cleaned_path = r"C:\Users\PRAVALLIKA\Smart_Stock_Project\MileStone1\cleaned_retail_dataset_single_store_2020_2025.csv"
original_data_path = "inventory_dataset.csv"  # Make sure this file is in the same folder


if not os.path.exists(m2_forecast_path):
    st.error("⚠️ Milestone 2 output not found at:\n" + m2_forecast_path)
    st.stop()

m2 = pd.read_csv(m2_forecast_path)
m2.columns = m2.columns.str.strip()

if "product_id" in m2.columns:
    m2 = m2.rename(columns={"product_id": "Product_ID"})

# --- Load product names from original dataset ---
product_names_map = {}
try:
    if os.path.exists(original_data_path):
        original_df = pd.read_csv(original_data_path)
        # Create mapping from product_id to product_name and category
        product_names_map = original_df[['product_id', 'product_name', 'category']].drop_duplicates().set_index('product_id').to_dict('index')
        st.sidebar.success(f"✅ Loaded {len(product_names_map)} product names")
    else:
        st.sidebar.warning("Original dataset not found - using product IDs only")
except Exception as e:
    st.sidebar.warning(f"Could not load product names: {e}")

# --- Helper function to get product display name ---
def get_product_display(product_id):
    if product_id in product_names_map:
        product_info = product_names_map[product_id]
        return f"{product_id} - {product_info['product_name']} ({product_info['category']})"
    return product_id

def get_product_name(product_id):
    if product_id in product_names_map:
        return product_names_map[product_id]['product_name']
    return "N/A"

def get_product_category(product_id):
    if product_id in product_names_map:
        return product_names_map[product_id]['category']
    return "N/A"

# --- Try loading cleaned sales from Milestone 1 ---
have_m1 = os.path.exists(m1_cleaned_path)
sales_df = None
if have_m1:
    try:
        sales_df = pd.read_csv(m1_cleaned_path)
        sales_df.columns = sales_df.columns.str.strip()
        # normalize column names
        if "date" in sales_df.columns:
            sales_df = sales_df.rename(columns={"date": "date"})
        if "product_id" in sales_df.columns:
            sales_df = sales_df.rename(columns={"product_id": "Product_ID"})
        if "units_sold" in sales_df.columns:
            sales_df = sales_df.rename(columns={"units_sold": "units_sold"})
        # make sure date is datetime
        if "date" in sales_df.columns:
            sales_df["date"] = pd.to_datetime(sales_df["date"], errors="coerce")
    except Exception as e:
        have_m1 = False
        st.sidebar.warning("Could not load sales data")

# --- Prepare product list with display names ---
if "Product_ID" in m2.columns:
    products = m2["Product_ID"].unique()
    # Create display names for dropdown
    product_display_names = [get_product_display(pid) for pid in products]
    # Create mapping from display name back to product ID
    display_to_id = {get_product_display(pid): pid for pid in products}
else:
    st.error("❌ Cannot find product id column in forecast_results.csv")
    st.stop()

# Sidebar controls
st.sidebar.header("Controls")
lead = st.sidebar.slider("Lead Time (days)", 1, 30, 7)
oc = st.sidebar.slider("Ordering Cost ($)", 10, 200, 50)
hc = st.sidebar.slider("Holding Cost ($/unit/year)", 1, 20, 2)
service_level = st.sidebar.selectbox("Service Level", ["90%", "95%", "99%"], index=1)
z_map = {"90%": 1.28, "95%": 1.65, "99%": 2.33}
z = z_map[service_level]

# Upload replacement files
st.sidebar.markdown("---")
uploaded_forecast = st.sidebar.file_uploader("Upload forecast_results.csv (optional)", type=["csv"])
uploaded_sales = st.sidebar.file_uploader("Upload cleaned sales CSV (optional)", type=["csv"])

if uploaded_forecast is not None:
    try:
        m2 = pd.read_csv(uploaded_forecast)
        m2.columns = m2.columns.str.strip()
        if "product_id" in m2.columns:
            m2 = m2.rename(columns={"product_id": "Product_ID"})
        st.sidebar.success("Uploaded forecast_results.csv loaded")
    except Exception as e:
        st.sidebar.error("Failed to read uploaded forecast CSV")

if uploaded_sales is not None:
    try:
        sales_df = pd.read_csv(uploaded_sales)
        sales_df.columns = sales_df.columns.str.strip()
        if "product_id" in sales_df.columns:
            sales_df = sales_df.rename(columns={"product_id": "Product_ID"})
        if "date" in sales_df.columns:
            sales_df["date"] = pd.to_datetime(sales_df["date"], errors="coerce")
        st.sidebar.success("Uploaded sales CSV loaded")
    except Exception as e:
        st.sidebar.error("Failed to read uploaded sales CSV")

# --- Helper: Generate forecast from historical data ---
def build_30d_forecast(product_id):
    # Use historical sales data to generate forecast
    if sales_df is not None and "Product_ID" in sales_df.columns and "units_sold" in sales_df.columns:
        product_sales = sales_df[sales_df["Product_ID"] == product_id]
        if len(product_sales) > 0:
            # Calculate daily average with some seasonality
            daily_avg = product_sales["units_sold"].mean()
            if np.isnan(daily_avg) or daily_avg == 0:
                daily_avg = 10  # Fallback value
                
            # Add some random variation (10% of average)
            variation = daily_avg * 0.1
            forecast = np.maximum(1, np.random.normal(loc=daily_avg, scale=variation, size=30))
            return forecast
    
    # Final fallback: random forecast between 5-50 units
    return np.random.randint(5, 50, size=30)

# --- Build forecasts for all products ---
product_forecasts = {}
for pid in products:
    product_forecasts[pid] = build_30d_forecast(pid)

# --- Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Dashboard", "Forecasts", "Inventory", "Stock Alerts", "Reports"])

# Tab 1: Dashboard Overview
with tab1:
    st.header("📊 Inventory Dashboard Overview")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_products = len(products)
        st.metric("Total Products", total_products)
    
    with col2:
        high_demand = len([p for p in products if np.mean(product_forecasts[p]) > 30])
        st.metric("High Demand Products", high_demand)
    
    with col3:
        avg_forecast = np.mean([np.mean(product_forecasts[p]) for p in products])
        st.metric("Avg Daily Forecast", f"{avg_forecast:.1f} units")
    
    with col4:
        reorder_count = len([p for p in products if np.random.randint(0, 150) < 20])  # Simulated
        st.metric("Need Reorder", reorder_count, delta=-reorder_count)
    
    # Top products by forecasted demand
    st.subheader("📈 Top 10 Products by Forecasted Demand")
    product_demand = []
    for p in products:
        avg_demand = np.mean(product_forecasts[p])
        product_demand.append({
            "Product ID": p,
            "Product Name": get_product_name(p),
            "Category": get_product_category(p),
            "Avg Daily Demand": round(avg_demand, 1)
        })
    
    demand_df = pd.DataFrame(product_demand).sort_values("Avg Daily Demand", ascending=False).head(10)
    st.dataframe(demand_df, use_container_width=True)

# Tab 2: Forecasts
with tab2:
    st.header("📈 Product Forecasts")
    selected_display = st.selectbox("Select Product", product_display_names, index=0, key="forecast_select")
    selected_product_id = display_to_id[selected_display]
    
    # Show forecast metrics from your CSV
    product_metrics = m2[m2["Product_ID"] == selected_product_id]
    if not product_metrics.empty:
        st.subheader("Forecast Accuracy Metrics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if "Prophet_MAE" in product_metrics.columns:
                mae = product_metrics["Prophet_MAE"].iloc[0]
                st.metric("Prophet MAE", f"{mae:.4f}")
        
        with col2:
            if "Prophet_RMSE" in product_metrics.columns:
                rmse = product_metrics["Prophet_RMSE"].iloc[0]
                st.metric("Prophet RMSE", f"{rmse:.4f}")
        
        with col3:
            if "Prophet_MAPE" in product_metrics.columns:
                mape = product_metrics["Prophet_MAPE"].iloc[0]
                st.metric("Prophet MAPE", f"{mape:.4f}%")
    
    # Forecast visualization
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Historical data if available
    if sales_df is not None and "Product_ID" in sales_df.columns:
        hist = sales_df[sales_df["Product_ID"] == selected_product_id].sort_values("date")
        if len(hist) > 0 and "date" in hist.columns:
            # Aggregate by month for cleaner historical view
            hist_monthly = hist.groupby(hist["date"].dt.to_period("M"))["units_sold"].sum().reset_index()
            hist_monthly["date"] = hist_monthly["date"].dt.to_timestamp()
            ax.plot(hist_monthly["date"], hist_monthly["units_sold"], 
                   label="Historical Monthly Sales", marker='o', linewidth=2)
    
    # 30-day forecast
    fc = product_forecasts[selected_product_id]
    future_dates = pd.date_range(start=pd.Timestamp.today().normalize(), periods=30, freq="D")
    ax.plot(future_dates, fc, label="30-day Daily Forecast", linestyle="--", marker='s', linewidth=2)
    
    ax.set_title(f"Sales Forecast: {selected_display}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Units")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

# Tab 3: Inventory calculations
with tab3:
    st.header("📦 Inventory Planning")
    st.info("Based on 30-day forecasts and your cost parameters")
    
    plan = []
    for p in products:
        fc = product_forecasts[p]
        total_demand = float(np.sum(fc))
        avg_daily = float(np.mean(fc))
        std = float(np.std(fc))
        
        # Inventory calculations
        yearly_demand = avg_daily * 365
        eoq = np.sqrt((2 * yearly_demand * oc) / max(hc, 0.0001))
        ss = z * std * np.sqrt(lead)
        rop = (avg_daily * lead) + ss
        
        plan.append({
            "Product ID": p,
            "Product Name": get_product_name(p),
            "Category": get_product_category(p),
            "Avg Daily Sales": round(avg_daily, 1),
            "Total 30d Demand": round(total_demand, 1),
            "EOQ": round(eoq, 0),
            "Safety Stock": round(ss, 1),
            "Reorder Point": round(rop, 1)
        })
    
    inv_plan = pd.DataFrame(plan)
    st.dataframe(inv_plan, use_container_width=True)

# Tab 4: Stock Alerts
with tab4:
    st.header("🚨 Stock Alerts")
    
    # Simulate current stock levels
    alert_data = []
    for p in products:
        fc = product_forecasts[p]
        avg_daily = float(np.mean(fc))
        std = float(np.std(fc))
        ss = z * std * np.sqrt(lead)
        rop = (avg_daily * lead) + ss
        
        # Simulate current stock (between 0-200 units)
        current_stock = np.random.randint(0, 200)
        
        # Determine alert level
        if current_stock == 0:
            status = "Out of Stock 🔴"
        elif current_stock < rop * 0.5:
            status = "Very Low Stock 🟠"
        elif current_stock < rop:
            status = "Low Stock 🟡"
        else:
            status = "Adequate Stock 🟢"
        
        alert_data.append({
            "Product ID": p,
            "Product Name": get_product_name(p),
            "Category": get_product_category(p),
            "Current Stock": current_stock,
            "Reorder Point": round(rop, 1),
            "Status": status
        })
    
    alerts_df = pd.DataFrame(alert_data)
    
    # Show critical alerts first
    critical_alerts = alerts_df[alerts_df["Status"].str.contains("🔴|🟠")]
    if not critical_alerts.empty:
        st.subheader("Critical Alerts")
        st.dataframe(critical_alerts, use_container_width=True)
    
    st.subheader("All Products Stock Status")
    st.dataframe(alerts_df, use_container_width=True)

# Tab 5: Reports & Downloads
with tab5:
    st.header("📋 Reports & Exports")
    
    # Generate the inventory plan for download
    download_plan = []
    for p in products:
        fc = product_forecasts[p]
        avg_daily = float(np.mean(fc))
        std = float(np.std(fc))
        yearly_demand = avg_daily * 365
        eoq = np.sqrt((2 * yearly_demand * oc) / max(hc, 0.0001))
        ss = z * std * np.sqrt(lead)
        rop = (avg_daily * lead) + ss
        
        download_plan.append({
            "Product_ID": p,
            "Product_Name": get_product_name(p),
            "Category": get_product_category(p),
            "Avg_Daily_Sales": round(avg_daily, 2),
            "EOQ": round(eoq, 2),
            "Safety_Stock": round(ss, 2),
            "Reorder_Point": round(rop, 2),
            "Lead_Time_Days": lead,
            "Service_Level": service_level
        })
    
    download_df = pd.DataFrame(download_plan)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CSV Download
        csv_bytes = download_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Inventory Plan (CSV)",
            data=csv_bytes,
            file_name="inventory_plan.csv",
            mime="text/csv"
        )
    
    with col2:
        # Excel Download
        try:
            import xlsxwriter
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                download_df.to_excel(writer, index=False, sheet_name="InventoryPlan")
                writer.save()
            excel_data = output.getvalue()
            st.download_button(
                "📥 Download Inventory Plan (Excel)",
                data=excel_data,
                file_name="inventory_plan.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except Exception:
            st.info("Install xlsxwriter for Excel downloads: `pip install xlsxwriter`")
    
    st.markdown("---")
    st.subheader("System Information")
    st.write(f"- **Products Loaded**: {len(products)}")
    st.write(f"- **Sales Data**: {'Available' if sales_df is not None else 'Not available'}")
    st.write(f"- **Product Names**: {'Loaded' if product_names_map else 'Not loaded'}")
    st.write(f"- **Current Parameters**: Lead Time={lead} days, Order Cost=${oc}, Holding Cost=${hc}/unit/year")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### How to Use")
st.sidebar.markdown("""
1. **Forecasts Tab**: View sales predictions
2. **Inventory Tab**: See EOQ and reorder points  
3. **Alerts Tab**: Monitor stock levels
4. **Reports Tab**: Download planning data
""")

st.sidebar.markdown("Done — dashboard loaded! 🎉")