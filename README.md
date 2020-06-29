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

*Can use the Jupyter Notebook DiscourseFeatureExtraction.ipynb or from DiscourseFeatureExtraction import FeatureExtractor in script when placing .py file in folder

**Running BERT + CRF + Features 





# References: 
[1] Giuliano Tortoreto, Evgeny A Stepanov, Alessandra Cervone, Mateusz Dubiel, and Giuseppe Riccardi. 2019.
Affective Behaviour Analysis of On-line User Interactions: Are On-line Support Groups more Therapeutic than
Twitter? In Proceedings ofthe 4th Social Media Mining for Health Applications (#SMM4H)Workshop & Shared
Task, pages 79–88.
