import pandas as pd

"""
takes JEOPARDY_CSV.csv and gets 1000 rows, and the columns are question, answer
"""
# Load JEOPARDY_CSV.csv
jeopardy = pd.read_csv('DQ_model_mistral7b.csv')

# Print columns to check their names
print("Columns in CSV:", jeopardy.columns.tolist())

# Select required columns (update these names if needed)
selected_columns = ['Question', 'Answer']  # Change to match actual column names (note: we removed the spaces in actual subset)
jeopardy_subset = jeopardy[selected_columns]

jeopardy_subset.to_csv('model_answers_mistral7b.csv', index=False)

# import pandas as pd

# # Load the dataset
# jeopardy = pd.read_csv('JEOPARDY_CSV.csv')

# # Print column names to verify
# print("Columns in CSV:", jeopardy.columns.tolist())

# # Use the correct column names (update if necessary)
# selected_columns = [' Question', ' Answer']
# jeopardy = jeopardy[selected_columns].copy()

# # Filter out rows where either column contains "<a href="
# mask = ~jeopardy.apply(lambda row: row.astype(str).str.contains('<a href=', regex=False), axis=1).any(axis=1)
# jeopardy_cleaned = jeopardy[mask]

# # Sample 1000 random rows after cleaning
# jeopardy_sample = jeopardy_cleaned.sample(n=1000, random_state=42)

# # Save to CSV
# jeopardy_sample.to_csv('jeopardy_subset.csv', index=False)

# print(f"Saved cleaned and sampled data with {len(jeopardy_sample)} rows to 'jeopardy_subset.csv'")