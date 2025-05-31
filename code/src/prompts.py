prompt_Q = lambda x : [{"role" : "user", "content" : f'List 200 existing airports in "North America". Just output the name of the airports. Output format should be - <num.>. "<AIRPORT>".'}]

prompt_IQ = lambda x : [{"role" : "user", "content" : f"""Who were the authors of the reference, "{x}"? Please list only the author names, formatted as - AUTHORS: <firstname> <lastname>, separated by commas. Do not mention the reference in the answer."""}]


prompt_DQ1_sample = lambda x : [{"role" : "user" , "content" : f"""Does the reference "{x}" exist? Just output yes/no."""}]

prompt_DQ2_sample = lambda x : [{"role": "user", "content" : 'Give a famous reference for reading.'},{"role": "assistant", "content" : f'"{x}"'},{"role": "user", "content" : """Does the above reference exist? Just output yes/no."""}]

prompt_DQ3_sample = lambda x,y :  [{"role": "user", "content" : f"""A language model generated references related to a research topic with the following titles:
{x}
Does the reference with title #{y} exist? Just output yes/no."""}]