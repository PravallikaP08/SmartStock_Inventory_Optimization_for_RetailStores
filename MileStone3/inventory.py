
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt


#SETUP

st.set_page_config(page_title="Smart Inventory Optimization Dashboard", layout="wide")

st.title("📦 Milestone 3 — Advanced Inventory Optimization Dashboard")

# Load your forecast dataset
df = pd.read_csv(r"C:\Users\PRAVALLIKA\Smart_Stock_Project\MileStone2\data\forecast_results.csv")

if "product_id" not in df.columns:
    st.error("❌ Missing 'product_id' column in dataset.")
    st.stop()

# SIDEBAR INPUTS

st.sidebar.header("🔧 Control Panel")

lead_time = st.sidebar.slider("Lead Time (days)", 1, 30, 7)
ordering_cost = st.sidebar.slider("Ordering Cost ($)", 10, 200, 50)
holding_cost = st.sidebar.slider("Holding Cost ($/unit/year)", 1, 20, 5)
unit_cost = st.sidebar.slider("Unit Purchase Cost ($)", 10, 200, 50)
stockout_cost = st.sidebar.slider("Stockout Cost ($/unit)", 10, 100, 25)

service_levels = {"90%": 1.28, "95%": 1.65, "99%": 2.33}
service_choice = st.sidebar.selectbox("Service Level", list(service_levels.keys()), 1)
z = service_levels[service_choice]

products = df["product_id"].unique()
selected_product = st.sidebar.selectbox("Select Product", products)

#     DEMAND & FORECAST LOGIC

inventory_plan = []

for product in products:
    prod_row = df[df["product_id"] == product].iloc[0]

    # Simulate average daily demand (better realism)
    avg_demand = np.random.randint(10, 60)
    std_demand = np.random.uniform(2, 10)

    # Identify best model
    if prod_row["Prophet_MAPE"] < prod_row["LSTM_MAPE"]:
        best_model = "Prophet"
        reliability = 100 - prod_row["Prophet_MAPE"]
    else:
        best_model = "LSTM"
        reliability = 100 - prod_row["LSTM_MAPE"]

    demand = avg_demand * 30  # monthly demand
    eoq = np.sqrt((2 * demand * ordering_cost) / (holding_cost))
    ss = z * std_demand * np.sqrt(lead_time)
    rop = (avg_demand * lead_time) + ss
    annual_cost = ((demand / eoq) * ordering_cost) + ((eoq / 2) * holding_cost) + (demand * unit_cost)

    inventory_plan.append({
        "Product": product,
        "Best_Model": best_model,
        "Forecast_Reliability(%)": round(reliability, 2),
        "AvgDailyDemand": round(avg_demand, 2),
        "Demand_StdDev": round(std_demand, 2),
        "EOQ": round(eoq, 2),
        "SafetyStock": round(ss, 2),
        "ReorderPoint": round(rop, 2),
        "Annual_Cost($)": round(annual_cost, 2)
    })

inv_df = pd.DataFrame(inventory_plan)


#        ABC + XYZ ANALYSIS

inv_df["Annual_Value"] = inv_df["AvgDailyDemand"] * 365 * unit_cost
inv_df = inv_df.sort_values(by="Annual_Value", ascending=False)
inv_df["Cumulative%"] = inv_df["Annual_Value"].cumsum() / inv_df["Annual_Value"].sum() * 100

def abc_category(x):
    if x <= 20: return "A"
    elif x <= 50: return "B"
    else: return "C"

inv_df["ABC"] = inv_df["Cumulative%"].apply(abc_category)

def xyz_category(std):
    if std < 4: return "X"
    elif std < 7: return "Y"
    else: return "Z"

inv_df["XYZ"] = inv_df["Demand_StdDev"].apply(xyz_category)
inv_df["Class"] = inv_df["ABC"] + inv_df["XYZ"]


#DASHBOARD TABS

tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📈 Analytics", "🧠 Recommendations"])

with tab1:
    st.subheader("📈 Inventory Overview")
    row = inv_df[inv_df["Product"] == selected_product].iloc[0]
    weeks = np.arange(1, 9)
    inv_level = np.linspace(100, 30, 8)
    plt.figure(figsize=(8, 4))
    plt.plot(weeks, inv_level, label="Inventory Level")
    plt.axhline(y=row["ReorderPoint"], color="orange", linestyle="--", label="Reorder Point")
    plt.axhline(y=row["SafetyStock"], color="red", linestyle="--", label="Safety Stock")
    plt.title(f"Inventory Simulation for {selected_product}")
    plt.xlabel("Weeks")
    plt.ylabel("Units")
    plt.legend()
    st.pyplot(plt.gcf())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🛒 EOQ", f"{row['EOQ']:.2f}")
    c2.metric("⚠️ Safety Stock", f"{row['SafetyStock']:.2f}")
    c3.metric("📦 Reorder Point", f"{row['ReorderPoint']:.2f}")
    c4.metric("💰 Annual Cost", f"${row['Annual_Cost($)']:.2f}")

with tab2:
    st.subheader("🔍 ABC–XYZ Classification Summary")
    st.dataframe(inv_df[["Product", "Best_Model", "Forecast_Reliability(%)", "ABC", "XYZ", "Class"]])

    st.subheader("💲 Total Annual Cost Distribution")
    plt.figure(figsize=(8, 4))
    plt.bar(inv_df["Product"], inv_df["Annual_Cost($)"])
    plt.xticks(rotation=90)
    plt.ylabel("Annual Cost ($)")
    st.pyplot(plt.gcf())

with tab3:
    st.subheader("🧠 Smart Recommendations")
    recs = []
    for _, r in inv_df.iterrows():
        if r["Forecast_Reliability(%)"] < 80:
            rec = f"🔸 {r['Product']}: Improve forecast accuracy (low reliability)."
        elif r["SafetyStock"] < (0.1 * r["ReorderPoint"]):
            rec = f"⚠️ {r['Product']}: Increase safety stock to prevent stockouts."
        elif r["EOQ"] > 500:
            rec = f"💼 {r['Product']}: Reduce order quantity to minimize holding costs."
        else:
            rec = f"✅ {r['Product']}: Inventory parameters are optimized."
        recs.append(rec)
    st.write("\n".join(recs))

    st.download_button("📥 Download Full Inventory Plan", inv_df.to_csv(index=False), "advanced_inventory_plan.csv")
