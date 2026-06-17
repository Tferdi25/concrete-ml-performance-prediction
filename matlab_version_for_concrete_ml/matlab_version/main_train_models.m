%% Machine Learning Models for Concrete Performance Prediction
% MATLAB implementation
%
% This script trains several regression models to predict concrete
% mechanical, durability, and environmental performance indicators.
%
% Repository: concrete-ml-performance-prediction
% Note: The sample dataset is synthetic and intended for demonstration of a
% reproducible machine learning workflow. Replace it with experimental data
% for research-grade analysis.

clear; clc; close all;

%% Locate project folders
scriptPath = mfilename('fullpath');
matlabFolder = fileparts(scriptPath);
projectRoot = fileparts(matlabFolder);

% If the script is executed from another location, paths remain valid.
dataFile = fullfile(projectRoot, 'data', 'sample_concrete_dataset.csv');
modelDir = fullfile(projectRoot, 'models', 'matlab');
resultDir = fullfile(projectRoot, 'results', 'matlab');
figureDir = fullfile(resultDir, 'figures');

if ~exist(modelDir, 'dir'); mkdir(modelDir); end
if ~exist(resultDir, 'dir'); mkdir(resultDir); end
if ~exist(figureDir, 'dir'); mkdir(figureDir); end

%% Load dataset
if ~isfile(dataFile)
    error('Dataset not found: %s', dataFile);
end

data = readtable(dataFile);

%% Define input features and target variables
featureNames = {
    'cement_kg_m3', ...
    'water_kg_m3', ...
    'fine_aggregate_kg_m3', ...
    'coarse_aggregate_kg_m3', ...
    'recycled_coarse_aggregate_pct', ...
    'silica_fume_pct', ...
    'fly_ash_pct', ...
    'superplasticizer_pct', ...
    'curing_days', ...
    'saturation_ratio_pct'};

targetNames = {
    'compressive_strength_mpa', ...
    'splitting_tensile_strength_mpa', ...
    'porosity_pct', ...
    'electrical_resistivity_kohm_cm', ...
    'water_absorption_pct', ...
    'co2_emission_kg_m3'};

X = data(:, featureNames);

%% Train-test split
rng(42);
cv = cvpartition(height(data), 'HoldOut', 0.20);
trainIdx = training(cv);
testIdx = test(cv);

XTrain = X(trainIdx, :);
XTest = X(testIdx, :);

%% Model configuration
modelNames = {
    'random_forest', ...
    'gradient_boosting', ...
    'support_vector_regression', ...
    'gaussian_process_regression'};

metrics = table();

% Structures used to store trained models target-by-target.
trainedModels = struct();
for m = 1:numel(modelNames)
    trainedModels.(modelNames{m}) = struct();
end

%% Train one model per target variable
for t = 1:numel(targetNames)
    targetName = targetNames{t};
    yTrain = data{trainIdx, targetName};
    yTest = data{testIdx, targetName};

    fprintf('\nTraining models for target: %s\n', targetName);

    % 1) Random Forest-style bagged regression ensemble
    model_random_forest = fitrensemble(XTrain, yTrain, ...
        'Method', 'Bag', ...
        'NumLearningCycles', 100, ...
        'Learners', templateTree('MinLeafSize', 5));

    % 2) Gradient Boosting regression ensemble
    model_gradient_boosting = fitrensemble(XTrain, yTrain, ...
        'Method', 'LSBoost', ...
        'NumLearningCycles', 100, ...
        'LearnRate', 0.05, ...
        'Learners', templateTree('MinLeafSize', 5));

    % 3) Support Vector Regression
    model_support_vector_regression = fitrsvm(XTrain, yTrain, ...
        'KernelFunction', 'gaussian', ...
        'Standardize', true);

    % 4) Gaussian Process Regression
    model_gaussian_process_regression = fitrgp(XTrain, yTrain, ...
        'KernelFunction', 'matern52', ...
        'Standardize', true);

    currentModels = struct();
    currentModels.random_forest = model_random_forest;
    currentModels.gradient_boosting = model_gradient_boosting;
    currentModels.support_vector_regression = model_support_vector_regression;
    currentModels.gaussian_process_regression = model_gaussian_process_regression;

    for m = 1:numel(modelNames)
        modelName = modelNames{m};
        model = currentModels.(modelName);

        yPred = predict(model, XTest);

        rmse = sqrt(mean((yTest - yPred).^2));
        mae = mean(abs(yTest - yPred));
        r2 = 1 - sum((yTest - yPred).^2) / sum((yTest - mean(yTest)).^2);

        newRow = table( ...
            string(modelName), string(targetName), rmse, mae, r2, ...
            'VariableNames', {'model', 'target', 'rmse', 'mae', 'r2'});
        metrics = [metrics; newRow]; %#ok<AGROW>

        trainedModels.(modelName).(targetName) = model;

        % Predicted vs actual figure
        fig = figure('Visible', 'off');
        scatter(yTest, yPred, 35, 'filled');
        hold on;
        axisMin = min([yTest; yPred]);
        axisMax = max([yTest; yPred]);
        plot([axisMin axisMax], [axisMin axisMax], '--', 'LineWidth', 1.2);
        xlabel('Actual value');
        ylabel('Predicted value');
        title(strrep(modelName, '_', ' ') + " - " + strrep(targetName, '_', ' '));
        grid on;

        figName = sprintf('%s_%s_predicted_vs_actual.png', modelName, targetName);
        saveas(fig, fullfile(figureDir, figName));
        close(fig);
    end
end

%% Save models and metrics
for m = 1:numel(modelNames)
    modelName = modelNames{m};
    modelStruct = trainedModels.(modelName); %#ok<NASGU>
    save(fullfile(modelDir, modelName + "_models.mat"), 'modelStruct', 'featureNames', 'targetNames');
end

writetable(metrics, fullfile(resultDir, 'matlab_model_metrics.csv'));

% Summary: mean performance per model across all targets
summaryMetrics = groupsummary(metrics, 'model', 'mean', {'rmse', 'mae', 'r2'});
writetable(summaryMetrics, fullfile(resultDir, 'matlab_model_summary.csv'));

fprintf('\nTraining completed successfully.\n');
fprintf('Models saved in: %s\n', modelDir);
fprintf('Results saved in: %s\n', resultDir);
disp(metrics);
