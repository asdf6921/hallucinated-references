import pandas as pd

"""
takes JEOPARDY_CSV.csv and gets 1000 rows, and the columns are question, answer
"""
# Load JEOPARDY_CSV.csv
jeopardy = pd.read_csv('JEOPARDY_CSV.csv')

# Print columns to check their names
print("Columns in CSV:", jeopardy.columns.tolist())

# Select required columns (update these names if needed)
selected_columns = [' Question', ' Answer']  # Change to match actual column names (note: we removed the spaces in actual subset)
jeopardy_subset = jeopardy[selected_columns].sample(n=1000, random_state=42)

jeopardy_subset.to_csv('jeopardy_subset.csv', index=False)