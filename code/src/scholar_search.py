from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
import time

def search_google_scholar(query, num_results=10):
    # Set up Chrome WebDriver with Service
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # Comment this out if you want to see the browser
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    try:
        # Go to Google Scholar
        driver.get("https://scholar.google.com")

        # Find the search box and submit the query
        search_box = driver.find_element(By.NAME, "q")
        search_box.send_keys(query)
        search_box.send_keys(Keys.RETURN)
        
        time.sleep(2)  # Let the page load

        # Get result elements
        results = driver.find_elements(By.CLASS_NAME, "gs_ri")[:num_results]

        for i, result in enumerate(results):
            title_elem = result.find_element(By.TAG_NAME, "h3")
            title = title_elem.text
            link = title_elem.find_element(By.TAG_NAME, "a").get_attribute("href") if title_elem.find_elements(By.TAG_NAME, "a") else "No link available"
            snippet = result.find_element(By.CLASS_NAME, "gs_rs").text if result.find_elements(By.CLASS_NAME, "gs_rs") else "No snippet available"

            print(f"\nResult {i+1}:")
            print(f"Title: {title}")
            print(f"Link: {link}") 
            print(f"Snippet: {snippet}")
    finally:
        driver.quit()

# Example usage
if __name__ == "__main__":
    search_query = "Digital Evidence and Computer Forensics"
    search_google_scholar(search_query)