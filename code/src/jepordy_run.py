import sys
sys.path.append('/Users/saket/.ollama/models')
from typing import Optional
import ollama
import json
import os
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


def reference_query(generator, prompt, concept, answer, top_p, temperature, LOG_PATH):
    if answer is None:
      dialogs = [prompt(concept)]
    else:
        dialogs = [prompt(concept,answer)]

    for dialog in dialogs:
        # Retry loop
        max_retries = 3
        attempt = 0
        response = None

        while attempt < max_retries:
            response = generator.chat(
                model='llama2:13b', # CHANGE THIS
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

def main_DQ(
    temperature: float = 0.0,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_gen_len: Optional[int] = None,
    read_path: str = None,          # CSV input file path
    write_csv_path: str = None,     # CSV output file path
    log_path: str = None,
    start_index: int = None,
    how_many: int = None,           # -1 for all
    question_column: str = None,    # Name of the column in CSV containing questions
    answer_column: str = None,       # Name of the column in CSV containing answers
    insert: bool = False,
):
    assert start_index is not None
    assert how_many is not None
    assert read_path is not None
    assert write_csv_path is not None
    assert question_column is not None
    assert answer_column is not None

    # Read CSV questions and answers
    with open(read_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    generator = ollama

    # # Define the prompt for checking answer correctness
    # prompt_DQ = lambda x, ans: [{
    #     "role": "user",
    #     "content": f"""Is this the correct answer to the question "{x}"? Answer either "yes" or "no" for this answer: "{ans}"."""
    # }]

    # Open output CSV and write header
    with open(write_csv_path, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.writer(outfile)
        # Write header: Question, Answer, 10 Answer Verification Columns, and Final Correctness Column
        header = ["Question", "Answer"] + [f"T{i+1}" for i in range(5)] + ["Correctness"]
        writer.writerow(header)

        # Store results for each row across 10 iterations
        verification_results = {i: [] for i in range(len(rows))}

        # Loop through 10 iterations
        for attempt in range(5):
            print(f"Iteration {attempt + 1} of 5...")

            # For each row, query the model and store results for this iteration (Verification_1, Verification_2, etc.)
            for i, row in enumerate(rows[start_index:]):
                if how_many != -1 and i >= how_many:
                    break

                question = row[question_column].strip()
                answer = row[answer_column].strip()
                print(f"Processing question: {question} index: {i}")

                # Query model for the correctness of the answer
                log_results = reference_query(generator, prompt_DQ_J, question, answer, top_p, temperature, log_path)
                model_content = log_results.get('gen_list', [])[0] if 'gen_list' in log_results else "no"
                model_content = model_content.strip().lower()

                # Store the verification result ('yes' or 'no') for the current iteration
                if model_content in ['yes', 'no']:
                    verification_results[i].append(model_content)
                else:
                    verification_results[i].append("no")  # Default to "no" if response is unexpected

        # After 5 iterations, write the results back to the output CSV
        for i, row in enumerate(rows[start_index:]):
            question = row[question_column].strip()
            answer = row[answer_column].strip()

            # Calculate correctness as decimal percentage (e.g., 0.6 for 60%)
            correctness_percentage = verification_results[i].count("yes") / 5.0
            if insert:
              correctness_percentage = 1-correctness_percentage

            # Write the row to the output CSV with the verification results and correctness
            row_data = [question, answer] + verification_results[i] + [correctness_percentage]
            writer.writerow(row_data)

            if (i + 1) % 20 == 0:
                print(f"Written {i+1} rows to {write_csv_path}")


    # # Read CSV questions and answers
    # with open(read_path, "r", encoding="utf-8") as f:
    #     reader = csv.DictReader(f)
    #     rows = list(reader)

    # generator = ollama

    # # Define the prompt for checking answer correctness
    # # prompt_DQ = lambda x, ans: [{
    # #     "role": "user",
    # #     "content": f"""Is this the correct answer to the question "{x}"? Answer either "yes" or "no" for this answer: "{ans}"."""
    # # }]

    # # Open output CSV and write header
    # with open(write_csv_path, "w", newline="", encoding="utf-8") as outfile:
    #     writer = csv.writer(outfile)
    #     # Write header: Question, 10 Answer Verification Columns, and Final Correctness Column
    #     header = ["Question", "Answer"] + [f"Verification_{i+1}" for i in range(5)] + ["Correctness"]
    #     writer.writerow(header)

    #     counter = 0
    #     for i, row in enumerate(rows[start_index:]):
    #         if how_many != -1 and counter >= how_many:
    #             break
    #         counter += 1

    #         question = row[question_column].strip()
    #         answer = row[answer_column].strip()
    #         print(f"Processing question: {question} index: {i}")

    #         # Store answers for 10 iterations
    #         verification_results = []

    #         # Run the verification 10 times
    #         for attempt in range(5):
    #             log_results = reference_query(generator, prompt_DQ, question, answer, top_p, temperature, log_path)

    #             # Extract the response (either 'yes' or 'no')
    #             model_content = log_results.get('gen_list', [])[0] if 'gen_list' in log_results else "no"
    #             model_content = model_content.strip().lower()

    #             # Store the verification result ('yes' or 'no')
    #             if model_content in ['yes', 'no']:
    #                 verification_results.append(model_content)
    #             else:
    #                 verification_results.append("no")  # Default to "no" if response is unexpected

    #         # Calculate correctness (how many "yes" responses)
    #         correctness_count = verification_results.count("yes") / 5.0
    #         if insert:
    #             correctness_count = 1-correctness_count

    #         # Write the row to the output CSV
    #         row_data = [question, answer] + verification_results + [correctness_count]
    #         writer.writerow(row_data)

    #         if (i + 1) % 20 == 0:
    #             print(f"Written {i+1} rows to {write_csv_path}")


def main_Q(
    temperature: float = 0.0,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_gen_len: Optional[int] = None,
    read_path: str = None,          # CSV input file path
    write_csv_path: str = None,     # CSV output file path
    log_path: str = None,
    start_index: int = None,
    how_many: int = None,           # -1 for all
    question_column: str = None      # Name of the column in CSV containing questions
):
    assert start_index is not None
    assert how_many is not None
    assert read_path is not None
    assert write_csv_path is not None
    assert question_column is not None

    # Read CSV questions
    with open(read_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    generator = ollama

    # Open output CSV and write header
    with open(write_csv_path, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["Question", "Answer"])  # Header row

        counter = 0
        for i, row in enumerate(rows[start_index:]):
            if how_many != -1 and counter >= how_many:
                break
            counter += 1

            question = row[question_column].strip()
            print(f"Processing question: {question} index: {i}")

            log_results = reference_query(generator, prompt_Q_J, question, None, top_p, temperature, log_path)

            # Extract answer — assumes prompt_Q returns single-word answer as first item in gen_list
            answer = ""
            if "gen_list" in log_results and len(log_results["gen_list"]) > 0:
                answer = log_results["gen_list"][0]

            writer.writerow([question, answer])

            if (i + 1) % 20 == 0:
                print(f"Written {i+1} rows to {write_csv_path}")


def main(
    gen_type: str = None,
    temperature: float = 0.0,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_gen_len: Optional[int] = None,
    read_path: str = None,
    write_csv_path: str = None,
    log_path: str = None,
    start_index: int = None,
    num_gen: int = None,
    how_many: int = None,
    dq_type: int = None,
    question_column: str = None,
    answer_column: str = None,
    insert: bool = False,

):
    if gen_type == "Q":
        main_Q(temperature, top_p, max_seq_len, max_gen_len,
               read_path, write_csv_path, log_path, start_index, how_many, question_column)
    if gen_type == "DQ":
        main_DQ(temperature, top_p, max_seq_len, max_gen_len,
               read_path, write_csv_path, log_path, start_index, how_many, question_column, answer_column, insert)


if __name__ == "__main__":
    #change model name:
    modeln = '_llama2_13b'
    main(
        gen_type="Q",
        temperature=0.0,
        top_p=0.9,
        max_seq_len=512,
        max_gen_len=200,
        read_path="/Users/saket/Documents/481/hallucinated-references/code/jeopardy/1kq.csv",  # Change this path to your CSV file
        write_csv_path='/Users/saket/Documents/481/hallucinated-references/code/jeopardy/model_answers'+ modeln + '.csv',     # This will be your output CSV
        log_path="/Users/saket/Documents/481/hallucinated-references/code/src/log1"+modeln+".txt",
        start_index=0,
        how_many=-1,
        question_column="Question"
    )
    print("Starting DQ's\n")
    main(
        gen_type="DQ",
        temperature=0.0,
        top_p=0.9,
        max_seq_len=512,
        max_gen_len=200,
        read_path='/Users/saket/Documents/481/hallucinated-references/code/jeopardy/model_answers'+ modeln + '.csv',  # Change this path to your CSV file
        write_csv_path="/Users/saket/Documents/481/hallucinated-references/code/jeopardy/DQ_model"+ modeln +".csv",     # This will be your output CSV
        log_path="/Users/saket/Documents/481/hallucinated-references/code/src/log2"+ modeln +".txt",
        start_index=0,
        how_many=-1,
        question_column="Question",                # Change this if your CSV uses a different column name
        answer_column="Answer",
        insert = False
    )
    main(
        gen_type="DQ",
        temperature=0.0,
        top_p=0.9,
        max_seq_len=512,
        max_gen_len=200,
        read_path="/Users/saket/Documents/481/hallucinated-references/code/jeopardy/1kqawa.csv",  # Change this path to your CSV file
        write_csv_path="/Users/saket/Documents/481/hallucinated-references/code/jeopardy/DQ_acc"+ modeln +".csv",     # This will be your output CSV
        log_path="/Users/saket/Documents/481/hallucinated-references/code/src/log3"+ modeln +".txt",
        start_index=0,
        how_many=-1,
        question_column="Question",                # Change this if your CSV uses a different column name
        answer_column="Answer",
        insert = False
    )
    main(
        gen_type="DQ",
        temperature=0.0,
        top_p=0.9,
        max_seq_len=512,
        max_gen_len=200,
        read_path="/Users/saket/Documents/481/hallucinated-references/code/jeopardy/1kqawa.csv",  # Change this path to your CSV file
        write_csv_path="/Users/saket/Documents/481/hallucinated-references/code/jeopardy/DQ_wrg"+ modeln +".csv",     # This will be your output CSV
        log_path="/Users/saket/Documents/481/hallucinated-references/code/src/log3"+ modeln +".txt",
        start_index=0,
        how_many=-1,
        question_column="Question",                # Change this if your CSV uses a different column name
        answer_column="WA1",
        insert = True
    )
