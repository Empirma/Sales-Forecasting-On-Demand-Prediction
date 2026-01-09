
from utils.preprocessing import *


file_path = 'data/depi_v0.csv'  # Replace with your file path
print(f"Loading data from {file_path}...")
df = load_data(file_path)

# 2. Investigate Data
print("\nInvestigating data...")
investigating_data(df)

# 3. Split Numeric and Categorical Columns (Optional Step for Analysis)
numeric_df, categorical_df = split_numeric_categorical(df)
print(f"\nNumeric Columns: {numeric_df.columns.tolist()}")
print(f"Categorical Columns: {categorical_df.columns.tolist()}")

# 4. Handle Unneeded Columns
print("\nHandling unneeded columns...")
df = handle_unneeded_columns(df)

# 5. Handle Missing Columns
print("\nHandling missing columns...")
df = handle_missing_columns(df)

# 6. Handle Missing Rows
print("\nHandling missing rows...")
df = handle_missing_rows(df)

df.to_csv("data/depi_ungrouped.csv",index=False)

# 7. Group Data
print("\nGrouping data...")
df_grouped = grouping_data(df)


# 8. Feature Engineering
print("\nPerforming feature engineering...")
df_features = feature_engineering(df_grouped)
df_features.to_csv("data/depi_grouped.csv",index=False)  

# 9. Time-Series Feature Engineering
print("\nCreating time-series features...")
df_time_series = time_series_features(df_features)
df_time_series.to_csv('data/depi_time_series.csv', index=False)


print("\nPipeline completed successfully!")



