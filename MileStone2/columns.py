import pandas as pd

file_path = r"C:\Users\PRAVALLIKA\Smart_Stock_Project\MileStone1\cleaned_retail_dataset_single_store_2020_2025.csv"
df = pd.read_csv(file_path)

print("Columns in dataset:")
print(df.columns.tolist())
