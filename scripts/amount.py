import pandas as pd

INPUT_CSV = "raw.csv"
OUTPUT_CSV = "1m.csv"

df = pd.read_csv(INPUT_CSV)

# add amount column at the end
df["amount"] = 0

df.to_csv(OUTPUT_CSV, index=False)

print(f"'amount' column added. Saved to {OUTPUT_CSV}")
