import sys
# Remove hardcoded path since we're using default Ollama installation
# sys.path.append('/Users/jerry/.ollama/models')
from typing import Optional
import ollama
import json
import os
# from bing_search import query_bing_return_mkt
import pandas as pd
import logging
import numpy as np
import re
import csv
from prompts import *
import time
import ast
import re
from convert_to_csv import convert_to_csv

def reference_query(generator, prompt, concept, top_p, temperature, LOG_PATH):
    dialogs = [prompt(concept)]

    for dialog in dialogs:
        # Retry loop
        max_retries = 3
        attempt = 0
        response = None

        while attempt < max_retries:
            response = generator.chat(
                model='gemma3:4b', #CHANGE_HERE
                messages=dialog,
                options={
                    "temperature": temperature,
                    "top_p": top_p
                }
            )

            if response and 'message' in response and response['message']['content'].strip():
                break
            else:
                attempt += 1

        if not response or 'message' not in response or not response['message']['content'].strip():
            model_content = "No response generated."
        else:
            model_content = response['message']['content']

    # Now parse the model_content based on newlines
    lines = model_content.split('\n')
    cleaned_list = []

    for line in lines:
        line = line.strip()
        if line:
            # Remove leading numbers like "1. ", "<1>. ", etc.
            line = line.lstrip('0123456789.<> ').strip()
            # Remove quotes if any
            if line.startswith('"') and line.endswith('"'):
                line = line[1:-1]
            # Take only first 3 characters
            line = line[:3]
            cleaned_list.append(line)

    result = {
        "title": concept,
        "gen_list": cleaned_list
    }

    return result

def IQ_query(generator, num_gen, gen_title, top_p, temperature, LOG_PATH, IQ_query_type):
    # Construct the question based on the provided format
    # question = f"Who were the authors of the reference {gen_title}? Please, list only the author names, formatted as - AUTHORS: <firstname> <lastname>, separated by commas. Do not mention the reference in the answer. "
    question = ""
    if (IQ_query_type == 1):
        question = f"What country is the airport with IATA code {gen_title} in? Please list only the exact official country name with no hyphens or accents, just alphabetic letters, formatted as - COUNTRY: <country_name>. Do not mention the airport in the answer."
    elif (IQ_query_type == 2):
        question = f"What is the name of the airport with IATA code {gen_title}? Please list only the exact airport name with no hyphens or accents, just alphabetic letters, formatted as - AIRPORT: <airport_name>. Do not mention the IATA code in the answer."
    elif (IQ_query_type == 3):
        question = f"What continent is the airport with IATA code {gen_title} in? Please list only the exact official continent name with no hyphens or accents, just alphabetic letters, formatted as - CONTINENT: <continent_name>. Do not mention the airport in the answer."

    model_ans = []
    for j in range(num_gen):
        # Generating the response using the Ollama API (or other model as per your use case)
        print(question, temperature)
        response = generator.chat(
            model='gemma3:4b',  #CHANGE_HERE
            messages=[{"role": "user", "content": question}],
            options={
                "temperature": temperature,
                "top_p": top_p
            }
        )

        # Extract the response content and add it to the list of answers
        ans = response['message']['content']
        model_ans.append(ans)

    return model_ans

def DQ_query(generator, num_gen, gen_title, top_p, temperature, LOG_PATH):
    # Construct the question based on the provided format
    # question = f"Who were the authors of the reference {gen_title}? Please, list only the author names, formatted as - AUTHORS: <firstname> <lastname>, separated by commas. Do not mention the reference in the answer. "
    question = f"Does the airport with IATA code {gen_title} exist? Just output yes/no."

    model_ans = []
    for j in range(num_gen):
        # Generating the response using the Ollama API (or other model as per your use case)
        print(question, temperature)
        response = generator.chat(
            model='gemma3:4b',  #CHANGE_HERE
            messages=[{"role": "user", "content": question}],
            options={
                "temperature": temperature,
                "top_p": top_p
            }
        )

        # Extract the response content and add it to the list of answers
        ans = response['message']['content']
        if "yes" in ans.lower():
            ans = "yes"
        elif "no" in ans.lower():
            ans = "no"
        model_ans.append(ans)

    return model_ans

def main_IQ(
    temperature: float = 0.0,
    num_gen: int = 1,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_gen_len: Optional[int] = None,
    read_path: str = None,
    LOG_PATH: str = None,
    start_index: int = None,
    how_many: int = None,  # -1 for all
    IQ_query_type: int = None
):
    assert start_index is not None
    assert how_many is not None
    assert IQ_query_type is not None

    generator = ollama

    df = pd.read_csv(read_path)

    counter = 0
    for i, row in df.iterrows():
        if i < start_index:
            continue
        if how_many != -1 and counter >= how_many:
            break
        counter += 1
        res = IQ_query(generator, num_gen, row["generated_reference"], top_p, temperature, LOG_PATH, IQ_query_type)

        for j in range(num_gen):
            if res[j] is not None:
                # swap \n with <br> here
                processed_text = res[j].replace('\n', '<br>')
                processed_text = re.sub(r"<think>.*?</think>", "", processed_text, flags=re.DOTALL).strip()
                processed_text = processed_text.replace("- COUNTRY:", "").replace("- AIRPORT:", "").replace("- CONTINENT:", "").replace("COUNTRY:", "").replace("AIRPORT:", "").replace("CONTINENT:", "").strip()
                df.loc[i, f"IQ_{IQ_query_type}_ans{j+1}"] = processed_text
            else:
                df.loc[i, f"IQ_{IQ_query_type}_ans{j+1}"] = ""

        df.to_csv(read_path, index=False, quoting=csv.QUOTE_ALL)
        print(i, "saved")


    df.to_csv(read_path, index=False, quoting=csv.QUOTE_ALL)

def main_DQ(
    temperature: float = 0.0,
    num_gen: int = 1,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_gen_len: Optional[int] = None,
    read_path: str = None,
    LOG_PATH: str = None,
    start_index: int = None,
    how_many: int = None,  # -1 for all
):
    assert start_index is not None
    assert how_many is not None

    generator = ollama

    df = pd.read_csv(read_path)

    counter = 0
    for i, row in df.iterrows():
        if i < start_index:
            continue
        if how_many != -1 and counter >= how_many:
            break
        counter += 1
        res = DQ_query(generator, num_gen, row["generated_reference"], top_p, temperature, LOG_PATH)

        for j in range(num_gen):
            if res[j] is not None:
                # swap \n with <br> here
                processed_text = res[j].replace('\n', '<br>')
                processed_text = re.sub(r"<think>.*?</think>", "", processed_text, flags=re.DOTALL).strip()
                df.loc[i, f"DQ_ans{j+1}"] = processed_text
            else:
                df.loc[i, f"DQ_ans{j+1}"] = ""

        df.to_csv(read_path, index=False, quoting=csv.QUOTE_ALL)
        print(i, "saved dq stuff")


    df.to_csv(read_path, index=False, quoting=csv.QUOTE_ALL)

def get_agreement_frac(s: str):
    "Extract agreement percentage from string and return it as a float between 0 and 1"
    x = s.strip().upper()
    if x.startswith("ANSWER"):
        x = x[6:].strip()
    if x.startswith("ANS"):
        x = x[3:].strip()
    x = x.strip(": ")
    # match an initial integer or float
    m = re.match(r"^[0-9]+\.?[0-9]*", x)
    if not m:
        return 0.0
    return min(float(m.group(0)) / 100.0, 1.0)

def consistency_check_airport_name(airport_list,generator):
    PROMPT = """Below is a list of airport names: <LIST>. On a scale of 0-100%, how much overlap is there in the airport names? Ignore minor variations in the name, as long as the airport name is correct. Answer with a number between 0 and 100. Output format should be ANS: <ans>."""

    prompt = PROMPT.replace("<LIST>", str(airport_list))
    response = generator.chat(
        model='gemma3:4b',  #CHANGE_HERE
        messages=[{"role": "user", "content": prompt}],
        options={
                    "temperature": 0.0,
                    "top_p": 0.95
                }
    )
    # Wrap the response like LLaMA would return
    results = {
        'generation': {
            'role': 'assistant',
            'content': response['message']['content']
        }
    }
    ans = results["generation"]["content"]
    return get_agreement_frac(ans)

# works for country and continent
def consistency_check_place(country_list,generator):
    # Convert all strings to lowercase and strip whitespace
    countries = [str(c).lower().strip() for c in country_list]
    print('COUNTRIES IN CHECK WERE: ', str(countries))

    # Count occurrences of each country
    country_counts = {}
    for country in countries:
        country_counts[country] = country_counts.get(country, 0) + 1

    # Find the maximum number of matches for any country
    max_matches = max(country_counts.values()) if country_counts else 0

    # Return fraction based on most frequent country
    return max_matches / len(countries)

def new_consistency_check_airport_name(airport_list,correct_airport_name, generator):
    PROMPT = """Below is a list of airport names: <LIST>. The correct expected airport name is <CORRECT_AIRPORT_NAME>. Return the number of airport names in the list which are the same as the correct airport name (small variations in hyphens, accents, spelling, etc. are allowed, but the core name should be correct). Return only the number of matches in the list. Output format should be ANS: <ans>."""
    
    prompt = PROMPT.replace("<LIST>", str(airport_list)).replace("<CORRECT_AIRPORT_NAME>", correct_airport_name)
    response = generator.chat(
        model='gemma3:4b',  # or whichever model you're using in Ollama
        messages=[{"role": "user", "content": prompt}],
        options={
                    "temperature": 0.0,
                    "top_p": 0.95
                }
    )
    # Wrap the response like LLaMA would return
    results = {
        'generation': {
            'role': 'assistant',
            'content': response['message']['content']
        }
    }
    ans = results["generation"]["content"]
    res = re.search(r'ANS:\s*(\d+)', ans)
    if res is None:
        cleaned_ans = 0
    else:
        cleaned_ans = int(res.group(1))
    return cleaned_ans


def new_consistency_check_place(place_list, iata_code, type: 'country' or 'continent'):
    places = [str(c).lower().strip() for c in place_list]
    place_counts = {}
    for place in places:
        place_counts[place] = place_counts.get(place, 0) + 1

    airports_df = pd.read_csv('/Users/jerry/Desktop/CSE Capstone/hallucinated-references/code/src/airports_code/new_airports.csv', encoding='latin1')

    matching_rows = airports_df[airports_df['IATA CODE'] == iata_code]
    if len(matching_rows) == 0:
        print(f"Warning: IATA code {iata_code} not found in airports database")
        return 0

    # Find the row where IATA CODE matches and get the Country Name
    correct_country = matching_rows['Country Name'].iloc[0].lower().strip()

    # get correct place from iata_code
    if type == 'country':
        if correct_country in place_counts:
            return place_counts[correct_country] / len(places)
        else:
            return 0
    elif type == 'continent':
        # get continent from iata_code
        # Read the continents CSV
        continents_df = pd.read_csv('/Users/jerry/Desktop/CSE Capstone/hallucinated-references/code/src/airports_code/Countries_by_continents.csv')

        matching_continent = continents_df[continents_df['Country'].apply(lambda x: x.lower().strip()) == correct_country]
        if len(matching_continent) == 0:
            print(f"Warning: Country {correct_country} not found in continents database")
            return 0

        # Find the continent for the correct country
        correct_continent = matching_continent['Continent'].iloc[0].lower().strip()
        
        if correct_continent in place_counts:
            return place_counts[correct_continent] / len(places)
        else:
            return 0

def new_consistency_check_DQ(dq_list,iata_code):
    # Check if IATA code exists in new_airports.csv
    airports_df = pd.read_csv('/Users/jerry/Desktop/CSE Capstone/hallucinated-references/code/src/airports_code/new_airports.csv', encoding='latin1')
    correct_answer = "yes"
    if len(airports_df[airports_df['IATA CODE'].apply(lambda x: x.strip() if isinstance(x, str) else str(x).strip()) == iata_code]) > 0:
        print(f"IATA code {iata_code} exists in new_airports.csv")
        correct_answer = "yes"
    else:
        print(f"IATA code {iata_code} does not exist in new_airports.csv")
        correct_answer = "no"
    
    # Count occurrences of each "yes" or "no"
    ans_counts = {}
    for ans in dq_list:
        ans_counts[ans] = ans_counts.get(ans, 0) + 1

    if correct_answer in ans_counts:
        return ans_counts[correct_answer] / len(dq_list)
    else:
        return 0

def addNewCCIQ(IQ_query_type):
    # Read the CSV file
    df = pd.read_csv('/Users/jerry/Desktop/CSE Capstone/hallucinated-references/code/airports_work/output_mistral_7b.csv') # TODO: CHANGE TO BE YOUR OUTPUT.CSV
    generator = ollama # TODO: DON'T NEED TO VARY THE MODEL, BUT CAN CHANGE IT TO SMALLEST MODEL FOOR EFFICIENCY
    
    # Loop through each row
    for index, row in df.iterrows():
        # Get the IATA code
        iata_code = row['generated_reference'].strip()
        
        # Get all answers for this IQ query type
        ans_columns = [col for col in df.columns if col.startswith(f' IQ_{IQ_query_type}_ans')]
        answers = [str(row[col]).lower().strip() for col in ans_columns]
        consistency = new_consistency_check_IQ(answers, iata_code, IQ_query_type)
            
        # Update the probability column
        df.at[index, f' IQ_{IQ_query_type}_llama_prob'] = consistency
    
    # Save the updated dataframe back to CSV
    df.to_csv('/Users/jerry/Desktop/CSE Capstone/hallucinated-references/code/airports_work/output_mistral_7b.csv', index=False)

def new_consistency_check_IQ(place_list, iata_code, IQ_query_type):
    places = [str(c).lower().strip() for c in place_list]
    place_counts = {}
    for place in places:
        place_counts[place] = place_counts.get(place, 0) + 1

    airports_df = pd.read_csv('/Users/jerry/Desktop/CSE Capstone/hallucinated-references/code/src/airports_code/new_airports.csv', encoding='latin1')

    matching_rows = airports_df[airports_df['IATA CODE'] == iata_code]
    if len(matching_rows) == 0:
        print(f"Warning: IATA code {iata_code} not found in airports database")
        return 0

    # Find the row where IATA CODE matches and get the Country Name
    correct_airport_name = matching_rows['Airport Name'].iloc[0].lower().strip()
    correct_country = matching_rows['Country Name'].iloc[0].lower().strip()

    # get correct place from iata_code
    if IQ_query_type == 1:
        if correct_country in place_counts:
            return place_counts[correct_country] / len(places)
        else:
            return 0
    elif IQ_query_type == 2:
        if correct_airport_name in place_counts:
            return place_counts[correct_airport_name] / len(places)
        else:
            return 0
    elif IQ_query_type == 3:
        # get continent from iata_code
        # Read the continents CSV
        continents_df = pd.read_csv('/Users/jerry/Desktop/CSE Capstone/hallucinated-references/code/src/airports_code/Countries_by_continents.csv')

        matching_continent = continents_df[continents_df['Country'].apply(lambda x: x.lower().strip()) == correct_country]
        if len(matching_continent) == 0:
            print(f"Warning: Country {correct_country} not found in continents database")
            return 0

        # Find the continent for the correct country
        correct_continent = matching_continent['Continent'].iloc[0].lower().strip()
        
        if correct_continent in place_counts:
            return place_counts[correct_continent] / len(places)
        else:
            return 0
        
def addNewCCDQ():
    # Read the CSV file
    df = pd.read_csv('/Users/jerry/Desktop/CSE Capstone/hallucinated-references/code/airports_work/output_mistral_7b.csv')
    
    # Loop through each row
    for index, row in df.iterrows():
        # Get the IATA code
        iata_code = row['generated_reference'].strip()
        # Get all answers for this DQ query
        ans_columns = [col for col in df.columns if col.startswith(f'DQ_ans')]
        answers = [str(row[col]).lower().strip() for col in ans_columns]
        consistency = new_consistency_check_DQ(answers, iata_code)
        # Update the probability column
        df.at[index, f' DQ_llama_prob'] = consistency

    # Save the updated dataframe back to CSV
    df.to_csv('/Users/jerry/Desktop/CSE Capstone/hallucinated-references/code/airports_work/output_mistral_7b.csv', index=False)

def consistency_check_DQ(dq_list,generator):
    print('DQS IN CHECK WERE: ', str(dq_list))

    # Count occurrences of each "yes" or "no"
    ans_counts = {}
    for ans in dq_list:
        ans_counts[ans] = ans_counts.get(ans, 0) + 1

    # Find the maximum number of matches for any "yes" or "no"
    max_matches = max(ans_counts.values()) if ans_counts else 0

    # Return fraction based on most frequent "yes" or "no"
    return max_matches / len(dq_list)

def main_CC(
    read_path: str,
    LOG_PATH: str,
    start_index: int,
    how_many: int,  # -1 for all
    IQ_query_type: int,
):
    generator = ollama

    start = time.time()
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    df = pd.read_csv(read_path)
    counter = 0
    log_lines = []

    for i, row in df.iterrows():
        if i < start_index:
            continue
        if how_many != -1 and counter >= how_many:
            break
        counter += 1

        fraction_same = None
        if IQ_query_type == 1 or IQ_query_type == 3:
            fraction_same = consistency_check_place(
                [row.get(f"IQ_{IQ_query_type}_ans{j}") for j in range(1, 4)], # the 4-1 is the number of generations we do for each IQ query
                generator
            )
        elif IQ_query_type == 2:
            fraction_same = consistency_check_airport_name(
                [row.get(f"IQ_{IQ_query_type}_ans{j}") for j in range(1, 4)],
                generator
            )

        log_lines.append(f"title: {row['title']}")
        log_lines.append(f"fraction_same: {fraction_same}\n")
        # log_lines.append(f"records: {records}\n")

        df.loc[i, f"IQ_{IQ_query_type}_llama_prob"] = fraction_same
        # df.loc[i, "IQ_llama_ans_list"] = str(records)

    with open(LOG_PATH, "w") as log_f:
        for line in log_lines:
            log_f.write(line + "\n")
        log_f.write(f"\nDone at {time.ctime()}\n\n")

    df.to_csv(read_path, index=False)
    print(f"Wrote {counter:,} entries to {LOG_PATH} in {time.time() - start:.2f}s")


def main_DQ_CC(
    read_path: str,
    start_index: int,
    how_many: int,  # -1 for all
):
    generator = ollama

    start = time.time()

    df = pd.read_csv(read_path)
    counter = 0

    for i, row in df.iterrows():
        if i < start_index:
            continue
        if how_many != -1 and counter >= how_many:
            break
        counter += 1

        fraction_same = consistency_check_DQ(
            [row.get(f"DQ_ans{j}") for j in range(1, 11)], # the 3 is the number of generations we do for each DQ query
            generator
        )

        df.loc[i, f"DQ_llama_prob"] = fraction_same

    df.to_csv(read_path, index=False)

# this function generates 200 references for each cs topic in acm_ccs_200.titles
# it writes the results to output.json
def main_Q(
    temperature: float = 0.0,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_gen_len: Optional[int] = None,
    read_path: str = None,
    write_json_path: str = None,
    log_path: str = None,
    start_index: int = None,
    how_many: int = None,  # -1 for all
):
    assert start_index is not None
    assert how_many is not None

    title_list = open(read_path, "r").readlines()
    res = []
    generator = ollama

    if start_index > 0:
        try:
            res = json.load(open(write_json_path, "r"))
        except FileNotFoundError:
            print(f"No file found at {write_json_path}, starting from scratch")

    counter = 0
    for i, entry in enumerate(title_list[start_index:]):
        if how_many != -1 and counter >= how_many:
            break
        counter += 1

        title = entry.strip()
        print(f"Processing title: {title} index: {i}")
        log_results = reference_query(generator, prompt_Q, title, top_p, temperature, log_path)

        # Keep only the last 200 items in gen_list
        if "gen_list" in log_results and len(log_results["gen_list"]) > 200:
            log_results["gen_list"] = log_results["gen_list"][-200:]

        res.append(log_results)

        if (i + 1) % 20 == 0:
            json.dump(res, open(write_json_path, "w"), indent=2, ensure_ascii=False)

    json.dump(res, open(write_json_path, "w"), indent=2, ensure_ascii=False)

def main(
    gen_type: str = None,
    temperature: float = 0.0,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_gen_len: Optional[int] = None,
    read_path: str = None,
    write_json_path: str = None,
    log_path: str = None,
    start_index: int = None,
    num_gen: int = None,
    how_many: int = None,
    dq_type: int = None,
    IQ_query_type: int = None
):
    if gen_type == "Q":
        main_Q(temperature, top_p, max_seq_len, max_gen_len,
               read_path, write_json_path, log_path, start_index, how_many)

    elif gen_type == "IQ":
        main_IQ(temperature, num_gen, top_p, max_seq_len, max_gen_len,
                read_path, log_path, start_index, how_many, IQ_query_type)
        print("IQ done")
        main_CC(read_path, log_path, start_index, how_many, IQ_query_type)

    elif gen_type == "DQ":
        main_DQ(num_gen, temperature, top_p, max_seq_len, max_gen_len,
                read_path, log_path, start_index, how_many)


if __name__ == "__main__":
    # modeln = '_gemma3_4b'
    # main(
    #     gen_type="Q",
    #     temperature=0.0,
    #     top_p=0.9,
    #     max_seq_len=512,
    #     max_gen_len=200,
    #     read_path="/Users/saket/Documents/481/hallucinated-references/code/src/airports_code/acm_ccs_200.titles",  # Change this path to your CSV file
    #     write_json_path='/Users/saket/Documents/481/hallucinated-references/code/airports_work/output'+ modeln + '.json',     # This will be your output CSV
    #     log_path="/Users/saket/Documents/481/hallucinated-references/code/src/logs/air_log1"+modeln+".txt",
    #     start_index=0,
    #     how_many=-1
    # )
    #     #     read_path="/Users/medhagupta/Documents/GitHub/hallucinated-references/code/src/acm_ccs_200.titles",
    #     # write_json_path="output.json",
    # convert_to_csv('/Users/saket/Documents/481/hallucinated-references/code/airports_work/output'+ modeln + '.json')
    # main(
    #     gen_type="IQ",
    #     temperature=0.3,
    #     top_p=0.9,
    #     max_seq_len=512,
    #     max_gen_len=200,
    #     read_path='/Users/saket/Documents/481/hallucinated-references/code/airports_work/output'+ modeln + '.csv',  # make sure this is a CSV
    #     log_path="/Users/saket/Documents/481/hallucinated-references/code/src/logs/air_log2"+modeln+".txt",
    #     start_index=0,
    #     num_gen=3,  # or 3
    #     how_many=-1,
    #     IQ_query_type=1
    # )
    # main(
    #     gen_type="IQ",
    #     temperature=0.3,
    #     top_p=0.9,
    #     max_seq_len=512,
    #     max_gen_len=200,
    #     read_path='/Users/saket/Documents/481/hallucinated-references/code/airports_work/output'+ modeln + '.csv',  # make sure this is a CSV
    #     log_path="/Users/saket/Documents/481/hallucinated-references/code/src/logs/air_log3"+modeln+".txt",
    #     start_index=0,
    #     num_gen=3,  # or 3
    #     how_many=-1,
    #     IQ_query_type=2
    # )
    # main(
    #     gen_type="IQ",
    #     temperature=0.3,
    #     top_p=0.9,
    #     max_seq_len=512,
    #     max_gen_len=200,
    #     read_path='/Users/saket/Documents/481/hallucinated-references/code/airports_work/output'+ modeln + '.csv',
    #     log_path="/Users/saket/Documents/481/hallucinated-references/code/src/logs/air_log4"+modeln+".txt",
    #     start_index=0,
    #     num_gen=3,  # or 3
    #     how_many=-1,
    #     IQ_query_type=3
    # )
    # main_DQ(
    #    num_gen=5,  # or 10
    #    temperature=0.0,
    #    top_p=0.9,
    #    max_seq_len=512,
    #    max_gen_len=200,
    #    read_path='/Users/saket/Documents/481/hallucinated-references/code/airports_work/output'+ modeln + '.csv',
    #    LOG_PATH="/Users/saket/Documents/481/hallucinated-references/code/src/logs/air_log5"+modeln+".txt",
    #    start_index=0,
    #    how_many=-1,
    # )
    # main_DQ_CC(
    #     read_path='/Users/saket/Documents/481/hallucinated-references/code/airports_work/output'+ modeln + '.csv',
    #     start_index=0,
    #     how_many=-1
    # )
    # df2 = pd.read_csv('/Users/medhagupta/Documents/GitHub/hallucinated-references/code/src/airports_code/new_airports.csv', encoding='latin1')
    # print(df2.columns)
    addNewCCIQ(1)
    addNewCCIQ(2)
    addNewCCIQ(3)
    addNewCCDQ()