%% Predict Concrete Properties Using a Trained MATLAB Model
% This script loads a trained model file and predicts performance indicators
% for a new concrete mixture.

clear; clc;

%% Locate project folders
scriptPath = mfilename('fullpath');
matlabFolder = fileparts(scriptPath);
projectRoot = fileparts(matlabFolder);

modelFile = fullfile(projectRoot, 'models', 'matlab', 'gaussian_process_regression_models.mat');

if ~isfile(modelFile)
    error(['Model file not found. Run main_train_models.m first. Missing file: ', modelFile]);
end

load(modelFile, 'modelStruct', 'featureNames', 'targetNames');

%% Define a new concrete mixture
% Values must use the same feature names and units as the training dataset.
newMixture = table( ...
    380, ... % cement_kg_m3
    180, ... % water_kg_m3
    720, ... % fine_aggregate_kg_m3
    980, ... % coarse_aggregate_kg_m3
    30,  ... % recycled_coarse_aggregate_pct
    8,   ... % silica_fume_pct
    10,  ... % fly_ash_pct
    1.2, ... % superplasticizer_pct
    28,  ... % curing_days
    85,  ... % saturation_ratio_pct
    'VariableNames', featureNames);

%% Predict all target variables
predictedValues = zeros(1, numel(targetNames));

for t = 1:numel(targetNames)
    targetName = targetNames{t};
    model = modelStruct.(targetName);
    predictedValues(t) = predict(model, newMixture);
end

predictionTable = array2table(predictedValues, 'VariableNames', targetNames);

disp('Predicted concrete performance indicators:');
disp(predictionTable);
