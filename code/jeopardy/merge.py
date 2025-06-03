# import pandas as pd

# # Load the three CSV files
# df1 = pd.read_csv('DQ_acc_gemma3_4b.csv')
# df2 = pd.read_csv('DQ_model_gemma3_4b.csv')
# df3 = pd.read_csv('DQ_wrg_gemma3_4b.csv')

# # Keep the Question column from the first file only
# questions = df1[['Question']]

# # Remove the Question column from all files to avoid duplication
# df1 = df1.drop(columns=['Question'])
# df2 = df2.drop(columns=['Question'])
# df3 = df3.drop(columns=['Question'])

# # Rename columns for df1
# df1.columns = [
#     f"Answer1" if col == "Answer" else
#     f"neural_ans1_prob" if col == "Correctness" else
#     f"{col}1" for col in df1.columns
# ]

# # Rename columns for df2
# df2.columns = [
#     f"Answer2" if col == "Answer" else
#     f"neural_ans2_prob" if col == "Correctness" else
#     f"{col}2" for col in df2.columns
# ]

# # Rename columns for df3
# df3.columns = [
#     f"Answer3" if col == "Answer" else
#     f"neural_ans3_prob" if col == "Correctness" else
#     f"{col}3" for col in df3.columns
# ]

# # Concatenate horizontally
# merged_df = pd.concat([questions, df1, df2, df3], axis=1)

# # Save to CSV
# merged_df.to_csv('merged_output.csv', index=False)

import pandas as pd

# Load the original CSVs
df1 = pd.read_csv('DQ_acc_mistral7b.csv')
df2 = pd.read_csv('DQ_model_mistral7b.csv')
df3 = pd.read_csv('DQ_wrg_mistral7b.csv')
model_df = pd.read_csv('mistral7b_evaluated_fuzzy.csv')

# Extract and keep only 'Question' from df1
questions = df1[['Question']]

# Convert model_answer column to boolean and rename it to 'bing_return'
model_df['bing_return'] = model_df['model_answer'].astype(str).str.strip().str.lower().map({'true': True, 'false': False})

# Drop Question from all answer DataFrames
df1 = df1.drop(columns=['Question'])
df2 = df2.drop(columns=['Question'])
df3 = df3.drop(columns=['Question'])

# Rename df1 columns
df1.columns = [
    'Answer1' if col == 'Answer' else
    'neural_ans1_prob' if col == 'Correctness' else
    f"{col}1" for col in df1.columns
]

# Rename df2 columns
df2.columns = [
    'Answer2' if col == 'Answer' else
    'neural_ans2_prob' if col == 'Correctness' else
    f"{col}2" for col in df2.columns
]

# Rename df3 columns
df3.columns = [
    'Answer3' if col == 'Answer' else
    'neural_ans3_prob' if col == 'Correctness' else
    f"{col}3" for col in df3.columns
]

# Combine all parts
merged_df = pd.concat([questions, df1, df2, df3, model_df[['bing_return']]], axis=1)

# Reorder columns to the specified format
ordered_columns = (
    ['Question'] +
    ['Answer1', 'T11', 'T21', 'T31', 'T41', 'T51', 'neural_ans1_prob'] +
    ['Answer2', 'T12', 'T22', 'T32', 'T42', 'T52', 'neural_ans2_prob'] +
    ['Answer3', 'T13', 'T23', 'T33', 'T43', 'T53', 'neural_ans3_prob'] +
    ['bing_return']
)

merged_df = merged_df[ordered_columns]

# Save final output
merged_df.to_csv('mistral7b_merged_output.csv', index=False)

# import pandas as pd
# import sys

# def remove_duplicates(input_file, output_file):
#     try:
#         df = pd.read_csv(input_file)
#         df_cleaned = df.drop_duplicates()
#         df_cleaned.to_csv(output_file, index=False)
#         print(f"Duplicates removed. Cleaned file saved to: {output_file}")
#     except Exception as e:
#         print(f"Error: {e}")

# if __name__ == "__main__":
#     if len(sys.argv) != 3:
#         print("Usage: python remove_duplicates.py input.csv output.csv")
#     else:
#         input_file = sys.argv[1]
#         output_file = sys.argv[2]
#         remove_duplicates(input_file, output_file)