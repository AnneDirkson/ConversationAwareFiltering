# Conversation-Aware Filtering
This repository contains the code for Conversation Aware Filtering for Online Patient Forums

# Prerequisites 
For installing the required packages:

*Using pip* pip install --upgrade pip pip install -r ./env/requirements.txt

*Using conda* conda env create -f ./env/environment.yml

# Usage 

**Feature Extraction** 

out= FeatureExtractor().main(datapath, outpath)

* datapath is the path to the input data (/data/ExampleData.tsv is the Medical Misinformation Data reformatted correctly).
* outpath is the path where you want to save the data with features

This script does not compute the Dialogue Act Features, because this classifier is not publicly available but can be requested by emailing Giuliano Tortoreto giuliano.tortoreto@unitn.it [1]

*Using Jupyter Notebook/Lab*

/src/DiscourseFeatureExtraction.ipynb

*Using Python*

download /src/DiscourseFeatureExtraction.py into folder

from DiscourseFeatureExtraction import FeatureExtractor 

**Running BERT + CRF + Features**
ConvAwareModel().main(self, data, outpath, da_acts = False): 

* data is the loaded pandas DataFrame with data (/data/ExampleDatawithFeat.tsv is the Medical Misinformation Data reformatted correctly after Feature Extraction).
* outpath is the folder where you wish all results to go 
* da_acts can be set to True if you compute the Dialogue Acts (see Feature Extraction for more explanation)

This script will: 
* Split the data into 10 folds based on discussion threads. 
* Run a BERT classifier over 10 folds. Here the number of epochs (3,4) are tuned per fold.
* Run a Blended CRF + BERT using the BERT predictions from the previous step. Here hyperparameters are tuned for the CRF. 
* Using the average performance and parameters from previous step, the script will perform greedy forward feature selection to see if the F1 score can be improved. 

The script will output: 
* in the main folder: main results for BERT, blended BERT and feature selection
* in /Folds/ the fold dictionaries using the thread_ids 
* in /BERTprobs/ the probabilities for train, dev and test sets for each fold according to the initial BERT 
* in /BERTpredictors/ the BERT predictors for the different folds
* in /FeatureSelect/ the progression of feature selection over each round

*Using Jupyter Notebook/Lab*

/src/BERT_CRF_DiscourseFeatures.ipynb

*Using Python*

download /src/BERT_CRF_DiscourseFeatures.py into folder

from BERT_CRF_DiscourseFeatures import ConvAwareModel



# References: 
[1] Giuliano Tortoreto, Evgeny A Stepanov, Alessandra Cervone, Mateusz Dubiel, and Giuseppe Riccardi. 2019.
Affective Behaviour Analysis of On-line User Interactions: Are On-line Support Groups more Therapeutic than
Twitter? In Proceedings ofthe 4th Social Media Mining for Health Applications (#SMM4H)Workshop & Shared
Task, pages 79–88.
