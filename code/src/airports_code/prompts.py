prompt_Q = lambda x : [{"role" : "user", "content" : f'List 200 existing world airports in this continent "{x}". Just output the exact 3-letter IATA code for each airport. Do not give the airport name or add any other formatting. Output format should be - <num.>. "<AIRPORT_IATA_CODE>".'}]

prompt_country_DQ = lambda x : [{"role" : "user", "content" : f"""Does the airport with IATA code "{x}" exist? Just output either the word 'yes' or 'no'. Don't return any other string or text. """}]

# I'm not actually using the below, I think I copy pasted them directly into llama_run.py
prompt_country_IQ1 = lambda x : [{"role" : "user", "content" : f"""What country is the airport with IATA code "{x}" in? Please list only the exact official country name with no hyphens or accents, just alphabetic letters, formatted as - COUNTRY: <country_name>. Do not mention the airport in the answer."""}]
prompt_country_IQ2 = lambda x : [{"role" : "user", "content" : f"""What is the name of the airport with IATA code "{x}"? Please list only the exact airport name with no hyphens or accents, just alphabetic letters, formatted as - AIRPORT: <airport_name>. Do not mention the country in the answer."""}]
prompt_country_IQ3 = lambda x : [{"role" : "user", "content" : f"""What continent is the airport with IATA code "{x}" in? Please list only the exact official continent name with no hyphens or accents, just alphabetic letters, formatted as - CONTINENT: <continent_name>. Do not mention the airport in the answer."""}]
