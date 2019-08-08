# deep-learning #genomicprediction 
scripts and examples
Scripts for genomic prediction with deep learning. Application of deep learning to agriculture. 

Autoencoder_gridsearch.R Function that uses an autoencoder machine learning approach for dimensionality reduction of data.This function has an bottleneck middle layer that "encodes" the data for dimensional reduction without the need to add the known predicted value. It is particularly useful as an alternative to Principal Components Analysis.

Prediction_DLpipeline.R Code from Enciso-see repository DLpipeline- written in Python translated to Keras R.While most of the code is verbatim, the hyperparameter tuning has been adapted to R capabilities by implementing a random search.
