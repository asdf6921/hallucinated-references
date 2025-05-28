import pandas as pd
import re
import random
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager

# === Text Normalization ===
def normalize_text(text):
    text = re.sub(r'<.*?>', '', text)  # Remove <TITLE>...</TITLE> etc.
    text = re.sub(r'[^a-z\s]', '', text.lower())  # Lowercase, remove non-letter
    return set(text.split())

# === Grounding Check ===
def is_grounded(reference, results):
    ref_words = normalize_text(reference)
    for result in results:
        try:
            title_elem = result.find_element(By.TAG_NAME, "h3")
            title = title_elem.text
            title_words = normalize_text(title)
            print(f"🔍 Comparing:\n→ REF:   {ref_words}\n→ TITLE: {title_words}")
            if ref_words.issubset(title_words):
                return True
        except Exception as e:
            print(f"Skipping a result due to error: {e}")
            continue
    return False

# === Scholar Search ===
def search_google_scholar(query, num_results=10):
    options = webdriver.ChromeOptions()

    # Set user-agent (macOS Chrome)
    

    # You can toggle headless mode if CAPTCHA is giving you trouble
    # options.add_argument("--headless")
    options.add_argument("--guest")  
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    grounded = False
    try:
        driver.get("https://scholar.google.com")
        time.sleep(2)

        search_box = driver.find_element(By.NAME, "q")
        search_box.send_keys(query)
        search_box.send_keys(Keys.RETURN)
        time.sleep(random.uniform(30, 35))  # May need to tweak this if pages load slowly

        results = driver.find_elements(By.CLASS_NAME, "gs_ri")[:num_results]
        grounded = is_grounded(query, results)

    except Exception as e:
        print(f"⚠️ Error searching for: {query[:60]}... → {e}")
    finally:
        driver.quit()

    return 'G' if grounded else 'H'

# === Load CSV ===
csv_path = "/Users/jerry/Desktop/CSE Capstone/hallucinated-references/code/src/llama 2 7b res/outputQ.csv"
df = pd.read_csv(csv_path)

# === Starting Index ===
start_index = 0  # Change this to resume from a different row

# Ensure 'label' column exists
if 'label' not in df.columns:
    df['label'] = ''

# === Process References ===
for i in range(start_index, len(df)):
    row = df.iloc[i]
    ref = str(row['generated_reference'])
    cleaned_ref = re.sub(r'[^a-z\s]', '', ref.lower())
    
    try:
        label = search_google_scholar(cleaned_ref)
    except Exception as e:
        print(f"⚠️ Error at index {i}: {e}")
        label = 'H'  # Default label if something breaks

    df.at[i, 'label'] = label
    print(f"{i+1}/{len(df)}: {ref[:60]}... → {label}")

    # Periodic checkpoint saves
    if (i + 1) % 10 == 0:
        print(f"💾 Checkpoint save at row {i+1}")
        df.to_csv(csv_path, index=False)

# === Final Save ===
df.to_csv(csv_path, index=False)

# === Summary ===
hallucinated = (df['label'] == 'H').sum()
total = len(df)
print(f"\n🧠 Hallucination rate: {hallucinated}/{total} ({100 * hallucinated / total:.2f}%)")