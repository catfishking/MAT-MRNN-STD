function [numSamp,dimSamp,D,sampPeriod] = read_mat(filename)

D = [];
load(filename)
numSamp = size(feature_dbn,1);
dimSamp = size(feature_dbn,2);
samplePeriod = 1;
D = feature_dbn';
clear feature*

