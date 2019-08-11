## deep-learning for genomic prediction :seedling: :seedling: :seedling:

Scripts for genomic prediction with deep learning . Application of deep learning to agriculture. 
Comments, suggestions, corrections :mailbox: duniapc77@gmail.com


## Prediction_DLpipeline.R 
Code from Enciso-see repository DLpipeline- written in Python translated to Keras R.While most of the code is verbatim, the hyperparameter tuning has been adapted to R capabilities by implementing a grid search instead of a random search.

## Prediction Multi-environment Genomic Prediction Montesinos.R
Code from Appendix script in the paper Montesinos et al 2018
Typical feedforward neural network also known as multilayer perceptron, which does not assume a specific structure in the input features. The input layer neurons correspond to the number of features (called independent variables by the statistics community)and hidden layer neurons are generally used to perform non-linear transformation of the original input attributes.The number of output neurons corresponds to the number of response variables (traits in plant breeding) you wish to predict or classify and they receive as input the output of hidden neurons and produce as output the prediction values of interest

## Multi-trait, Multi-environment Genomic Prediction Montesinos.R
Code from Appendix script in the paper Montesinos et al 2018
The basic structure ofa densely connected network consists of an input layer, L output layers (for multi-trait modeling) and multiple hidden layers between the input and output layers.This type of neural network is also known as a feedforward neural network.

## DeepGS_Wenlong2017.R https://github.com/cma2015/DeepGS
From paper DeepGS Predicting phenotypes from genotypes using Deep learning. The paper is interesting because they followed a different approach, instead of a MLP, ANN or NN they use a convolutional neural network approach (CNN) and it is also not implemented in Keras. The code is implemented using MXNet, if you want to know more about the differences between DL/AI frameworks see https://skymind.com/wiki/comparison-frameworks-dl4j-tensorflow-pytorch. In brief MXNet is faster and has higher accuracy 🔥 🔥 and it can be implemented as a Keras backend, I plan to convert the DeepGS code to Keras R just because I plan to stick to Keras ..for the moment.

## Autoencoder_gridsearch.R 
Function that uses an autoencoder machine learning approach for dimensionality reduction of data.This function has an bottleneck middle layer that "encodes" the data for dimensional reduction without the need to add the known predicted value. It is particularly useful as an alternative to Principal Components Analysis.

## datacamp_deeplearning.py
Code from exercises in the datacamp website to obtain a certificate on deep learning python. Completed :mortar_board:


## nn_scratch.R
Neural network built from scratch, build your own neural network function and compare it with the results of the neuralnet R package. As seen on http://gradientdescending.com/deep-neural-network-from-scratch-in-r/


