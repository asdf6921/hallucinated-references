import pandas as pd

# File path
input_path = "/Users/jerry/Desktop/CSE Capstone/hallucinated-references/code/airports_work/output_llama2_7b.csv"
output_path = input_path  # overwrite the same file

# Load CSV
df = pd.read_csv(input_path)

# Strip column names to remove accidental whitespace
df.columns = df.columns.str.strip()

# Add new column 'bing_return'
df["bing_return"] = df["neural_ans1_prob"] > 0

# Save back to the same file
df.to_csv(output_path, index=False)

print("Updated 'bing_return' column added and file saved.")
