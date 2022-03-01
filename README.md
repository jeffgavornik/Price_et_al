# Price_et_al
Analysis and stimulus-generation code used in Price et al. 2022

## Data Download
Data from the paper (necessary to run the code) can be found in MATLAB format in the Data folder (SeqRFExp_DataForMbTDR...). This is pre-processed data, along with the optimal model fit from MbTDR (SeqRFExp_ModelFitAIC...).
NWB format forthcoming, with both raw and pre-processed data.

## Example Script
Download the data and this repository. Run "RunStatAnalyses.m" from FigureStats folder to create dot plots and run permutation tests from the paper.

## Model-based targeted dimensionality reduction (MbTDR)
MbTDR folder contains code to fit the MbTDR model. If you are interested in using the model on your own data, we highly recommend using code from the original paper by Mikio Aoi, Valerio Mante, & Jonathan Pillow: https://github.com/pillowlab/mTDRdemo

Decoding folder contains code to perform decoding analyses, given the MbTDR fit.

## Reference
Price, Jensen, Khoudary, Gavornik 2022: Expectation violations produce error signals in mouse V1.
https://www.biorxiv.org/content/10.1101/2021.12.31.474652v1
