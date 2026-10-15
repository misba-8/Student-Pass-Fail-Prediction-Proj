
import pandas as pd
import glob
import os

files = glob.glob("*.csv")

if not files:
    print("CSV file nahi mili. Dataset project folder mein rakho.")
else:
    print("CSV files:", files)

    df = pd.read_csv(files[0])

    print("\nDataset Shape:")
    print(df.shape)

    print("\nColumn Names:")
    print(df.columns.tolist())

    print("\nFirst 5 Rows:")
    print(df.head())

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDataset Information:")
    df.info()

# Create total score
df["total_score"] = (
    df["math score"]
    + df["reading score"]
    + df["writing score"]
)

# Create Pass/Fail column
df["result"] = (
    (df["math score"] >= 40)
    & (df["reading score"] >= 40)
    & (df["writing score"] >= 40)
).map({True: "Pass", False: "Fail"})

# Display results
print("\nPass/Fail Results:")
print(df[["math score", "reading score",
          "writing score", "result"]].head(10))

print("\nResult Count:")
print(df["result"].value_counts())


print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDataset Size:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

print("\nData Types:")
print(df.dtypes)

