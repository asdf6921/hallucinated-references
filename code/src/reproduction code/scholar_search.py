import pandas as pd
import re
import requests
import time
import concurrent.futures

# === CONFIGURATION ===
csv_path = "/Users/jerry/Desktop/CSE Capstone/hallucinated-references/code/src/qwen 3 4b/output.csv"
start_index = 0  # Change this to resume from a different row
checkpoint_every = 50  # Save after every 10 rows

# === Text Normalization ===
def normalize_text(text):
    text = re.sub(r'<.*?>', '', text)  # Remove tags
    text = re.sub(r'[^a-z\s]', '', text.lower())  # Lowercase and keep letters/spaces only
    return set(text.split())


def safe_normalize(text, timeout=2):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(normalize_text, text)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            print("⏳ normalize_text timed out!")
            return None

# === OpenAlex Query + Grounding Check ===
def is_grounded_openalex(reference):
    url = "https://api.openalex.org/works"
    params = {"search": reference, "per-page": 5}

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        results = response.json().get("results", [])

        for item in results:
            title = item.get("title", "")

            print(f"🔍 Comparing:\n→ REF:   \"{reference}\"\n→ TITLE: \"{title}\"")
            
            ref_words = normalize_text(reference)
            title_words = safe_normalize(title)

            if ref_words.issubset(title_words) or title_words.issubset(ref_words):
                return 'G'
    except Exception as e:
        print(f"⚠️ Error querying OpenAlex: {e}")
    
    return 'H'

# === Load CSV ===
df = pd.read_csv(csv_path)

# Ensure 'label' column exists
if 'label' not in df.columns:
    df['label'] = ''

# === Process References ===
for i in range(start_index, len(df)):
    raw_ref = str(df.at[i, 'generated_reference']).strip()

    if not raw_ref:
        print(f"{i+1}/{len(df)}: Empty reference → H")
        df.at[i, 'label'] = 'H'
        continue

    # Clean off prefix like <3>. " or similar
    clean_ref = re.sub(r'^<\d+>\.\s*["“”]?', '', raw_ref).rstrip('."” ').strip()

    try:
        label = is_grounded_openalex(clean_ref)
    except Exception as e:
        print(f"⚠️ Error at index {i}: {e}")
        label = 'H'

    df.at[i, 'label'] = label
    print(f"{i+1}/{len(df)}: {clean_ref[:60]}... → {label}")

    # Periodic checkpoint
    if (i + 1) % checkpoint_every == 0:
        df.to_csv(csv_path, index=False)
        print(f"💾 Saved checkpoint at row {i+1}")
        time.sleep(1)  # polite pause

# === Final Save ===
df.to_csv(csv_path, index=False)

# === Summary ===
hallucinated = (df['label'] == 'H').sum()
total = len(df)
print(f"\n🧠 Hallucination rate: {hallucinated}/{total} ({100 * hallucinated / total:.2f}%)")