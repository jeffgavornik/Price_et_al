# Data

Data from the paper is in this folder. For usage, see files such as RunStatAnalyses.m or DecodeWrapper.m . 

The "DataForMbTDR" file contains all of the data, from 140 multi-channels, with 600 trials / channel. "DataForMbTDRDecoder" is held-out data that was not used during the MbTDR fitting procedure and reserved for decoding / estimation of explained variance. "ModelFitAIC" gives parameters and other outputs from the optimal MbTDR fit.  

Variable naming conventions:  
Z - neural data  
X - design matrix (info on each predictor is in DesignCellCodes)  
N - number of units  
P - total number of predictors  
R - rank of optimal MbTDR fit for each predictor  
S, W - as defined in the paper for MbTDR  
binSize - size of neural data bins in seconds (25ms bins)  
