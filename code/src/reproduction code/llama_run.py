import sys
sys.path.append('/Users/jerry/.ollama/models')
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


def process_search_query(s):
    s = s.strip().strip("\"").strip("“").strip("”").strip("’").strip("‘").strip("\"").strip(".").strip()
    return s

def extract_answers(ans,is_pair=False):
    ans_lis = ans.split("\n")
    gen_pubs = []
    for line in ans_lis:

        match_pub = re.search(r'(\d+)\.\s*(?:“|")(.+)(?:"|”)',line)
        if match_pub:
            line_new = match_pub.group(2)
            gen_pubs.append(line_new)
        else:
            # optionally log or just skip
            continue
    return gen_pubs


def extract_authors_from_ans(title, ans):
    if pd.isna(ans) or ans == "" or ans == "None" or ans == "nan":
        return None
    ans = str(ans)
    authors = ans.replace(title, "provided reference").replace("AUTHORS:", "")
    return authors

def df_extract_authors(read_path):
    df = pd.read_csv(read_path)
    for i in range(1, 4):
        full_ans_col = f"IQ_full_ans{i}"
        ans_col = f"IQ_ans{i}"
        if full_ans_col in df.columns:
            df[ans_col] = df.apply(lambda x:
                                   extract_authors_from_ans(x["generated_reference"], x[full_ans_col]), axis=1)
    df.to_csv(read_path, index=False)

def reference_query(generator, prompt, concept, top_p, temperature, LOG_PATH):
    dialogs = [prompt(concept)]

    for dialog in dialogs:
        # Retry loop
        max_retries = 3
        attempt = 0
        response = None

        while attempt < max_retries:
            response = generator.chat(
                model='mistral:7b',
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
            cleaned_list.append(line)

    result = {
        "title": concept,
        "gen_list": cleaned_list
    }

    return result

def direct_query(generator,prompt,title,max_gen_len,top_p,temperature,LOG_PATH,i=None,all_ans=None):

    if i is not None and all_ans is not None:
        assert title in all_ans
        prompt = prompt(all_ans,(i%5)+1)
    else:
        prompt = prompt(title)

    alias = title.replace(" ","_").replace(":","").replace("/","_")

    logging.basicConfig(filename=LOG_PATH + alias + ".log",filemode='w',level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',force=True)
    logging.info("temperature: " + str(temperature))

    dialogs = [prompt]

    logging.info("Sending query\n {}".format(dialogs))

    response = generator.chat(
        model='mistral:7b',  # or whichever model you're using in Ollama
        messages=messages,
        options={
            "temperature": temperature,
            "top_p": top_p
        }
    )

    # Wrap the response like LLaMA would return
    results = {
        'generation': {
            'role': 'assistant',
            'content': response['message']['content']
        }
    }

    #there is only one dialog, we are not batching right now
    for dialog, result in zip(dialogs, results):
        for msg in dialog:
            logging.info(f"{msg['role'].capitalize()}: {msg['content']}\n")
        logging.info(
            f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}")
        logging.info("\n==================================\n")
        logging.info("Yes probability: {}".format(result['yes_prob']))
        logging.info("Token with respect to which prob is calculated (0) mean default\n {}".format(result['prob_token']))
        #logging.info("Model Answer\n {}".format(ans))
    print(result['generation']['content'],result['yes_prob'].item())
    return result['generation']['content'],result['yes_prob'].item(),result['prob_token']

def DQ_query_sample(generator, prompt_fn, gen_title, max_gen_len, top_p, temperature, LOG_PATH=None, i=None, all_ans=None, num_gen=None):
    if i is not None:
        prompt_text = prompt_fn(gen_title, (i % 5) + 1)
    else:
        prompt_text = prompt_fn(gen_title)

    model_ans = []
    for _ in range(num_gen):
        try:
            response = generator.chat(
                model='mistral:7b',
                messages=prompt_text,
                options={
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_gen_len
                }
            )
            assistant_content = response['message']['content']
            model_ans.append(assistant_content)
        except Exception as e:
            model_ans.append(f"ERROR: {str(e)}")

    # Post-process answers
    n_prob = 0
    for ans in model_ans:
        if "yes" in ans.lower():
            n_prob += 1

    n_prob = n_prob / num_gen if num_gen else 0

    return model_ans, n_prob

def correct_sample_dq(file_path):
    df = pd.read_csv(file_path)
    def return_prob(x):
        n_prob = 0
        num_gen = len(x)
        for i in range(num_gen):
            if "yes" in x[i].lower():
                n_prob = n_prob + 1
        n_prob = n_prob/num_gen
        return n_prob
    df["DQ1_prob_sample"] = df["DQ1_ans_sample"].apply(lambda x: return_prob(ast.literal_eval(x)))
    # df["DQ2_prob_sample"] = df["DQ2_ans_sample"].apply(lambda x: return_prob(ast.literal_eval(x)))
    # df["DQ3_prob_sample"] = df["DQ3_ans_sample"].apply(lambda x: return_prob(ast.literal_eval(x)))
    df.to_csv(file_path,index=False)

def IQ_query(generator, num_gen, gen_title, top_p, temperature, LOG_PATH):
    alias = gen_title.replace(" ", "_").replace(":", "").replace("/", "_")
    # Construct the question based on the provided format
    question = f"Who were the authors of the reference {gen_title}? Please, list only the author names, formatted as - AUTHORS: <firstname> <lastname>, separated by commas. Do not mention the reference in the answer. "

    model_ans = []
    for j in range(num_gen):
        # Generating the response using the Ollama API (or other model as per your use case)
        print(question, temperature)
        response = generator.chat(
            model='mistral:7b',  # or whichever model you're using in Ollama
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

def main_DQ(
    num_gen: int = None,
    temperature: float = 0.0,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_gen_len: Optional[int] = None,
    read_path: str = None,
    LOG_PATH: str = None,
    start_index: int = None,
    how_many: int = None, #-1 for all
    dq_type: int = None
):
    generator = ollama

    df = pd.read_csv(read_path)
    suffix = ""
    if num_gen is not None:
        suffix = "_sample"
    if dq_type == 1:
        prompt = prompt_DQ1 if num_gen is None else prompt_DQ1_sample
        verbose_ans = f"DQ1_ans{suffix}"
        prob_ans = f"DQ1_prob{suffix}"
        prob_ans_token = "DQ1_prob_token"
    elif dq_type == 2:
        prompt = prompt_DQ2 if num_gen is None else prompt_DQ2_sample
        verbose_ans = f"DQ2_ans{suffix}"
        prob_ans = f"DQ2_prob{suffix}"
        prob_ans_token = "DQ2_prob_token"
    elif dq_type == 3: #need to check some implementations
        prompt = prompt_DQ3 if num_gen is None else prompt_DQ3_sample
        verbose_ans = f"DQ3_ans{suffix}"
        prob_ans = f"DQ3_prob{suffix}"
        prob_ans_token = "DQ3_prob_token"
    counter = 0
    i = start_index
    while i < len(df):
        if how_many != -1 and counter >= how_many:
            break

        # DQ1 and DQ2 logic untouched
        if dq_type in [1, 2] or num_gen is None:
            row = df.iloc[i]
            if dq_type == 1 or dq_type == 2:
                if num_gen is None:
                    ans, prob, prob_token = direct_query(generator, prompt, row["generated_reference"], max_gen_len, top_p, temperature, LOG_PATH)
                else:
                    ans, prob = DQ_query_sample(generator, prompt, row["generated_reference"], max_gen_len, top_p, temperature, LOG_PATH, i=None, all_ans=None, num_gen=num_gen)
            elif dq_type == 3:
                if num_gen is None:
                    ans, prob, prob_token = direct_query(generator, prompt, row["generated_reference"], max_gen_len, top_p, temperature, LOG_PATH, i, row["model_answer_main_query"])
                else:
                    # DQ3 with sampling is handled below
                    pass

            if dq_type != 3 or num_gen is None:
                df.loc[i, verbose_ans] = str(ans)
                df.loc[i, prob_ans] = prob
                if num_gen is None:
                    df.loc[i, prob_ans_token] = prob_token

                print(i, "done in method", dq_type)
                if i % 20 == 0:
                    df.to_csv(read_path, index=False)
                    print(i, "saved")

                i += 1
                counter += 1
                continue

        # --- New logic for DQ3 with sampling (group of 5 rows) ---
        if dq_type == 3 and num_gen is not None:
            titles = []
            indices = []
            for j in range(5):
                if i + j < len(df):
                    titles.append(df.iloc[i + j]["generated_reference"])
                    indices.append(i + j)

            combined_title = " \n ".join(titles)
            ans, prob = DQ_query_sample(generator, prompt, combined_title, max_gen_len, top_p, temperature, LOG_PATH, i=i, all_ans=None, num_gen=num_gen)

            for idx in indices:
                df.loc[idx, verbose_ans] = str(ans)
                df.loc[idx, prob_ans] = prob

            print(f"{i}-{i+len(indices)-1} done in DQ3 sample group")
            if i % 20 == 0:
                df.to_csv(read_path, index=False)
                print(i, "saved")

            i += 5
            counter += 1

    df.to_csv(read_path, index=False)

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
        res = IQ_query(generator, num_gen, row["generated_reference"], top_p, temperature, LOG_PATH)

        for j in range(num_gen):
            if res[j] is not None:
                # swap \n with <br> here
                processed_text = res[j].replace('\n', '<br>')
                processed_text = re.sub(r"<think>.*?</think>", "", processed_text, flags=re.DOTALL).strip()
                df.loc[i, f"IQ_full_ans{j+1}"] = processed_text
            else:
                df.loc[i, f"IQ_full_ans{j+1}"] = ""

        df.to_csv(read_path, index=False, quoting=csv.QUOTE_ALL)
        print(i, "saved")


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

def consistency_check_pair(list1,list2,generator):
    PROMPT = """Below are what should be two lists of authors. On a scale of 0-100%, how much overlap is there in the author names (ignore minor variations such as middle initials or accents)? Answer with a number between 0 and 100. Also, provide a justification. Note: if either of them is not a list of authors, output 0. Output format should be ANS: <ans> JUSTIFICATION: <justification>.

    1. <NAME_LIST1>
    2. <NAME_LIST2>"""

    list1 = str(list1).strip().replace("<br>", " ")
    list2 = str(list2).strip().replace("<br>", " ")
    prompt = PROMPT.replace("<NAME_LIST1>", list1).replace("<NAME_LIST2>", list2)
    response = generator.chat(
        model='mistral:7b',  # or whichever model you're using in Ollama
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
    return (get_agreement_frac(ans), ans)

def consistency_check(auth_lists,generator):
    n = len(auth_lists)
    assert n >= 2
    records = []
    fracs = []
    for i in range(n):
        for j in range(i):
            frac,ans = consistency_check_pair(auth_lists[i],auth_lists[j],generator)
            records.append(ans)
            fracs.append(frac)
    mean = sum(fracs)/len(fracs)
    return mean, records

def main_CC(
    read_path: str,
    LOG_PATH: str,
    start_index: int,
    how_many: int  # -1 for all
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

        mean, records = consistency_check(
            [row.get(f"IQ_ans{j}") for j in range(1, 4)],
            generator
        )

        log_lines.append(f"title: {row['title']}")
        log_lines.append(f"mean: {mean}")
        log_lines.append(f"records: {records}\n")

        df.loc[i, "IQ_llama_prob"] = mean
        df.loc[i, "IQ_llama_ans_list"] = str(records)

    with open(LOG_PATH, "w") as log_f:
        for line in log_lines:
            log_f.write(line + "\n")
        log_f.write(f"\nDone at {time.ctime()}\n\n")

    df.to_csv(read_path, index=False)
    print(f"Wrote {counter:,} entries to {LOG_PATH} in {time.time() - start:.2f}s")


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

        # Keep only the last 5 items in gen_list
        if "gen_list" in log_results and len(log_results["gen_list"]) > 5:
            log_results["gen_list"] = log_results["gen_list"][-5:]

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
    dq_type: int = None
):
    if gen_type == "Q":
        main_Q(temperature, top_p, max_seq_len, max_gen_len,
               read_path, write_json_path, log_path, start_index, how_many)

    elif gen_type == "IQ":
        main_IQ(temperature, num_gen, top_p, max_seq_len, max_gen_len,
                read_path, log_path, start_index, how_many)
        print("IQ done")
        df_extract_authors(read_path)
        main_CC(read_path, log_path, start_index, how_many)

    elif gen_type == "DQ":
        main_DQ(num_gen, temperature, top_p, max_seq_len, max_gen_len,
                read_path, log_path, start_index, how_many, dq_type)
        correct_sample_dq(read_path)



if __name__ == "__main__":
    # main(
    #     gen_type="Q",
    #     temperature=0.0,
    #     top_p=0.9,
    #     max_seq_len=512,
    #     max_gen_len=200,
    #     read_path="/Users/jerry/Desktop/CSE Capstone/hallucinated-references/code/src/acm_ccs_200.titles",
    #     write_json_path="output.json",
    #     log_path="log.txt",
    #     start_index=0,
    #     how_many=-1
    # )
    # convert_to_csv("/Users/jerry/Desktop/CSE Capstone/hallucinated-references/code/src/output.json")
    # main(
    #     gen_type="IQ",
    #     temperature=0.3,
    #     top_p=0.9,
    #     max_seq_len=512,
    #     max_gen_len=200,
    #     read_path="/Users/jerry/Desktop/CSE Capstone/hallucinated-references/code/src/output.csv",  # make sure this is a CSV
    #     log_path="log.txt",
    #     start_index=302,
    #     num_gen=5,  # or however many generations you want
    #     how_many=-1
    # )
    main_DQ(
       num_gen=3,  # or however many generations you want per prompt
       temperature=0.7,
       top_p=0.9,
       max_seq_len=512,
       max_gen_len=200,
       read_path="/Users/jerry/Desktop/CSE Capstone/hallucinated-references/code/src/reproduction code/reproduction results/Mistral 7b res/output.csv",
       LOG_PATH="log.txt",
       start_index=0,
       how_many=-1,
       dq_type=3
    )