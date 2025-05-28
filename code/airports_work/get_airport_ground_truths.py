import pandas as pd

def compare(generatedResult):
  # Load airport names from CSV
  df = pd.read_csv('airportst_subset.csv')
  airport_names = set(df['name'].astype(str).str.strip())

  # Split input into lines and extract names
  lines = generatedResult.strip().split('\n')
  results = []
  for line in lines:
    parts = line.split('.', 1)
    name = parts[1].strip() if len(parts) > 1 else line.strip()
    results.append(name in airport_names)
  return results