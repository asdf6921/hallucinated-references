### Setup
- `python==3.8`
- `requests`
- `ollama`
- `pandas>=1.4.4`
- `numpy>=1.23.0`
- download `auc_delong_xu.py` from [RaulSanchezVazquez/roc_curve_with_confidence_intervals](https://github.com/RaulSanchezVazquez/roc_curve_with_confidence_intervals)
- download `ollama` from https://ollama.com/download


### Running the experiments

#### For ollama models
- `ollama run <model_name>` to download the model desired to run


#### For Reproducing the Experiment
- `cd src/reproduction code`
- Under llama_run.py, run the main gen_type = 'Q' code to generate references based on the acm_ccs_200.titles
- Under llama_run.py, run the convert_to_csv function to save the json from the previous generated results to a csv file for IQ and DQ later on.
- Under llama_run.py, run the main gen_type="IQ" to generate answers to indirect questions
- Under llama_run.py, run the main gen_type="DQ" to generate answers to direct questions.  `dq_type` specifies which direct question to ask the model.
- For a specific model, replace every `model='mistral:7b'` with the desired model based on model prompt from https://ollama.com/
- To obtain the ground truth, run `scholar_search.py` and changing the csv path to the correct csv file
- Variable `num_gen` determines the number of repetition for a question
- Variable `temperature` decides the temperature the model will respond with
- Variable `read_path` decides the csv file to read and save the questions from and to.
- Variable `start_index` decides which row of the csv file to start querying the model. This is used in case of a break whilst running to prevent regenerating the entire dataset.

### Metrics

- `code//metrics.ipynb` notebook could be run by
- setting the `PATH` variable to the correct csv file to get the ROC and FDR curves.
- changing the csv columns header to `neural_ans{i}_prob` where 1 is for IQ, and 2, 3, 4 for DQ1, DQ2, DQ3 respectively
- changing the csv column header `label` to `bing_return`

