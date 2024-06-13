### Global Microplastic Transport Model

This repository contains all the code required to run the global microplastic transport model, to collect the results, and to create plotting functions. Details about the contents of each of the files is given below. 
It should be noted that most of the files require input data to function correctly. Please contact arthur.ronner@live.nl if you would like to use this model for more detailed information on input data.

## Code overview

__model_discrete_time.py__: Contains the model and data preparation classes
- data preparation module: reformats the input data such that it works with the model
- model: everything required to initialise and run one instance of the model

__run_model_batch.py__: Contains the code required to run multiple instances of the model, using an already existing uncertainty sample.

__samples_util.py__: Translates a uncertainty ranges file to an uncertainty sample that can be used by the run_model_batch.py file.

__prepare_data.py__: Calculates the nearest neighbor approximation of washingmachine ownership per country, based on country HDI.
Also generates the corresponding plot.

__filter_data.py__: Collects the data from multiple runs into a numpy array that can be used by the analysis.py module.

__collect_geo_data.py__: Collects the data from multiple runs into a numpy array that can be used by the geo_analysis.py and execute_geo_plots.py

__analysis.py__: Contains most of the plotting functions of the report.

__geo_analysis.py__: Contains the geographical plotting functions of the report.

__execute_geo_plots.py__: Contains the functions that execute the figures for the report.


## Data overview

The data folder contains a couple of key excel files that were used to run the model.

