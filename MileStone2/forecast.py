import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler


file_path = r"C:\Users\PRAVALLIKA\Smart_Stock_Project\MileStone1\cleaned_retail_dataset_single_store_2020_2025.csv"
output_path = r"C:\Users\PRAVALLIKA\Smart_Stock_Project\MileStone2\data"
os.makedirs(output_path, exist_ok=True)


df = pd.read_csv(file_path)
date_col = "date"
df[date_col] = pd.to_datetime(df[date_col])


def mae(y_true, y_pred):
    min_len = min(len(y_true), len(y_pred))
    return np.mean(np.abs(y_true[:min_len] - y_pred[:min_len]))

def rmse(y_true, y_pred):
    min_len = min(len(y_true), len(y_pred))
    return np.sqrt(np.mean((y_true[:min_len] - y_pred[:min_len]) ** 2))

def mape(y_true, y_pred):
    min_len = min(len(y_true), len(y_pred))
    y_true, y_pred = y_true[:min_len], y_pred[:min_len]
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if np.all(y_true != 0) else np.nan


def create_sequences(data, n_steps=5):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i+n_steps])
        y.append(data[i+n_steps])
    return np.array(X), np.array(y)

def train_lstm(train_series, n_steps=5):
    values = train_series.values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)

    X, y = create_sequences(scaled, n_steps)
    if X.size == 0:
        return None, None
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_steps, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=20, verbose=0)
    return model, scaler

def forecast_lstm(model, scaler, series, steps=30, n_steps=5):
    if model is None:
        return np.array([])
    input_seq = scaler.transform(series.values.reshape(-1, 1))[-n_steps:]
    input_seq = input_seq.reshape((1, n_steps, 1))
    yhat_list = []
    for _ in range(steps):
        yhat = model.predict(input_seq, verbose=0)  # shape (1,1)
        yhat_list.append(scaler.inverse_transform(yhat)[0, 0])
        yhat_reshaped = yhat.reshape(1, 1, 1)  # (batch=1, step=1, feature=1)
        input_seq = np.concatenate([input_seq[:, 1:, :], yhat_reshaped], axis=1)
    return np.array(yhat_list)



results = []

for product in df['product_id'].unique():
    print(f"🔄 Training Prophet & LSTM for product {product}...")
    product_df = df[df['product_id'] == product][[date_col, 'units_sold']]
    product_df = product_df.rename(columns={'units_sold': 'Sales'}).dropna()

   
    if len(product_df) < 2:
        print(f"⚠️ Skipping {product} (not enough data)")
        continue

    
    prophet_df = product_df.rename(columns={date_col: "ds", "Sales": "y"})
    model_p = Prophet()
    model_p.fit(prophet_df)
    future = model_p.make_future_dataframe(periods=30)
    forecast_p = model_p.predict(future)

    # LSTM
   
    sales_series = product_df.set_index(date_col)['Sales']
    lstm_model, scaler = train_lstm(sales_series)
    yhat_l = forecast_lstm(lstm_model, scaler, sales_series, steps=30)

  
    # Metrics
   
    actual = prophet_df['y'].values
    yhat_p = forecast_p['yhat'].values[:-30]  # match training part

    mae_p = mae(actual, yhat_p)
    rmse_p = rmse(actual, yhat_p)
    mape_p = mape(actual, yhat_p)

    mae_l = mae(actual, sales_series.values[-len(yhat_l):]) if len(yhat_l) > 0 else np.nan
    rmse_l = rmse(actual, sales_series.values[-len(yhat_l):]) if len(yhat_l) > 0 else np.nan
    mape_l = mape(actual, sales_series.values[-len(yhat_l):]) if len(yhat_l) > 0 else np.nan

    print(f"Prophet MAE:{mae_p:.2f} RMSE:{rmse_p:.2f} MAPE:{mape_p:.1f}%")
    print(f"LSTM    MAE:{mae_l:.2f} RMSE:{rmse_l:.2f} MAPE:{mape_l:.1f}%")
    print("--------------------------------------------------")

    # Save results
    results.append([product, mae_p, rmse_p, mape_p, mae_l, rmse_l, mape_l])

   
    # Plotting

    plt.figure(figsize=(10, 6))
    plt.plot(product_df[date_col], product_df['Sales'], label="Actual", color="black")
    plt.plot(forecast_p['ds'], forecast_p['yhat'], label="Prophet Forecast", color="blue")
    if len(yhat_l) > 0:
        future_dates = pd.date_range(start=product_df[date_col].iloc[-1], periods=30, freq="D")
        plt.plot(future_dates, yhat_l, label="LSTM Forecast", color="red")
    plt.title(f"Forecast for Product {product}")
    plt.xlabel("Date")
    plt.ylabel("Units Sold")
    plt.legend()
    plt.tight_layout()

    # Save each plot
    plt.savefig(os.path.join(output_path, f"forecast_{product}.png"))
    plt.close()


# Save results to CSV

results_df = pd.DataFrame(results, columns=[
    "product_id", "Prophet_MAE", "Prophet_RMSE", "Prophet_MAPE",
    "LSTM_MAE", "LSTM_RMSE", "LSTM_MAPE"
])
results_df.to_csv(os.path.join(output_path, "forecast_results.csv"), index=False)

print("\n✅ Forecast results saved at:")
print(os.path.join(output_path, "forecast_results.csv"))