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

### Running the Extensions

#### Airports Extension.
- `cd code/src/airports_code/` to access the code
- All the commands to running the airports extension are the same as steps in the section above: For Reproducing the Experiment
- to change the prompts to be asked to the model, go to `cd code/src/airports_code/prompts.py` to enter custom prompts.
- For a specific model, input the desired model everywhere there is a `model=` with the desired model based on model prompt from https://ollama.com/. (ex: `model=mistral:7b`)
- Access to the dataset of airports is located: https://www.kaggle.com/datasets/sanjeetsinghnaik/world-airport-dataset?resource=download
- To run the metrics for the output from the airports run's, follow the Metrics section above with `code/src/airports_code/airportMetrics.ipynb`. make sure to set the `PATH` variable to the correct csv file to get the ROC and FDR curves.



#### Jeopardy Extension.
- `cd code/src/jeopardy_run.py` to access the code to run the prompts.
- To access the jeopardy data and the csv files, `cd code/jeopardy/`
- All the commands to running the jeopardy extension are the same as steps in the section above: For Reproducing the Experiment
- No need to change any inputs or flags, just need to run `./jepordy_run.py`
- make sure to change the file path of where you are going to save the output files.
- to change the prompts to be asked to the model, go to `cd src/prompts_jeoprdy.py` to enter custom prompts.
- For a specific model, input the desired model everywhere there is a `model=` with the desired model based on model prompt from https://ollama.com/. (ex: `model=mistral:7b`)
- Access to the dataset of jeopardy prompts is located: https://www.kaggle.com/datasets/tunguz/200000-jeopardy-questions
- To run the metrics for the output from the airports run's, follow the Metrics section above with `src/jeoMetrics.ipynb`. make sure to set the `PATH` variable to the correct csv file to get the ROC and FDR curves.



