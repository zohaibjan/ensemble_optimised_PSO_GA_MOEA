function program = mainProgram()
clear all;
clc;

Problem = dataSetNames();                 % Get list of dataset names

% Problem = {'hepatitis'};

%% Model SETTINGS
params.NCA = false;
params.numOfFolds = 10;                   % Create CROSS VALIDATION FOLDS
params.runPSO = false;                    % for optimization
params.noOfClusters = 5;                  % For nth root of clustering
params.elimination = 5;                   % chances each classifier will be given
params.classifiers = {'ANN', 'SVM', 'KNN', 'DT', 'DISCR', 'NB'};
% params.classifiers = {'ADABOOST'};
params.trainFunctionANN={'trainlm','trainbfg','trainrp','trainscg','traincgb','traincgf','traincgp','trainoss','traingdx'};
params.trainFunctionDiscriminant = {'pseudoLinear','pseudoQuadratic'};
params.kernelFunctionSVM={'gaussian','polynomial','linear'};

%% MAIN LOOP
for i=1:length(Problem)
    p_name = Problem{i};
    disp(p_name);
    results = runTraining(p_name, params);
    saveResults(results, p_name);
end
end





