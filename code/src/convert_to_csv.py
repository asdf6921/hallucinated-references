import pandas as pd
import json

def convert_to_csv(json_path: str):
    # Load the JSON
    data = json.load(open(json_path, "r"))
    
    # Create a list of rows
    rows = []
    for entry in data:
        title = entry.get("title", "")
        gen_list = entry.get("gen_list", [])
        
        for generated in gen_list:
            rows.append({
                "title": title,
                "generated_reference": generated
            })
    
    # Create DataFrame
    df = pd.DataFrame(rows)

    # Save CSV
    csv_path = json_path.replace(".json", ".csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV to {csv_path}")

json_path = "/Users/jerry/Desktop/CSE Capstone/hallucinated-references/code/src/output.json"
convert_to_csv(json_path)