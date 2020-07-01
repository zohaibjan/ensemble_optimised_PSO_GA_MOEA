function rstx = runTraining(p_name , params)
warning('off','all');

%% Create CROSS VALIDATION FOLDS
numOfFolds = params.numOfFolds;
data = load([pwd,filesep,'P-Data',filesep, p_name]);
data = [data.X, data.y];  %data = [normalize(data.X, 2) , data.y];
data = rmmissing(data);
cvFolds = cvpartition(data(:,end), 'KFold', numOfFolds);

%% RECORD KEEPING VARIABLES
avgAccuracy = [];
avgAccuracy_without_BCS = [];
sizeBeforeBCS = [];
sizeAfterBCS = [];
K = [];
rstx = {};

%% ITERATE OVER THE NUMBER OF FOLDS
for fold=1:numOfFolds
    classifierIndex = 1;
    classifiers = {};
    trainData = data(cvFolds.training(fold),:);
    testData = data(cvFolds.test(fold),:);
    
    %% DIMENSIONALITY REDUCTION WITH FEATURE SELECTION NCA-NEIGHBOURHOOD COMPONENT ANALYSIS
    if params.NCA == true
        nca = fscnca(trainData(:,1:end-1),trainData(:,end),'FitMethod','exact','Lambda',0,...
            'Solver','sgd','Standardize',true);
        L = loss(nca,testData(:,1:end-1),testData(:,end));
        tol= L;
        selidx = find(nca.FeatureWeights > tol*max(1,max(nca.FeatureWeights)));
        trainData = [trainData(:,selidx'), trainData(:, end)];
        testData = [testData(:,selidx'), testData(:, end)];
    end
   
    %% SEPARATE VALIDATION DATA PER FOLD
    cv_vali_folds = cvpartition(trainData(:,end), 'holdout', 0.1);
    validationDataIndex = test(cv_vali_folds);
    validationData = trainData(validationDataIndex,:);
    trainingSet =  ~validationDataIndex;
    trainData = trainData(trainingSet, :);
    
    %% HOW MANY CLUSTERS UPPER BOUND - K
    noOfClusters = round(nthroot(size(trainData,1),params.noOfClusters));
    
    allClusters = zeros(length(trainData),1);
    totalClusters = 1;
    %% GENERATE ALL CLUSTERS - from k = 1, 2, ..., K
    for clusters=1:noOfClusters
        %% CLUSTERINGS
        % STANDARDIZE DATA BEFORE CLUSTERING
        if length(trainData) > clusters
            clusterIds = kmeans(normalize(trainData(:,1:end-1),2), clusters, 'MaxIter', 2400);                                                                          % 'MaxIter',500, 'Replicates',5,  'dist','sqeuclidean'
            for j=1:clusters
                allClusters(:,totalClusters) = (clusterIds == j); % indexes of clusters
                totalClusters = totalClusters + 1;
            end
        end
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%% - END OF CLUSTERING
    
    generatedClusters(fold) = totalClusters;
    
    %% TRAIN ON CLUSTERS
    for j = allClusters
        trainCluster = trainData(find(j),:);
        noOfRecords = size(trainCluster,1);
        noOfClasses = length(unique(trainCluster(:,end)));
        if  noOfClasses > 1 && noOfRecords >= size(trainCluster,2) %https://arxiv.org/abs/1211.1323
            % TRY WITH 10/25/58/75 test samples
            all = trainClassifiers(trainCluster, params);
            for temp = 1:length(all)
                all = trainClassifiers(trainCluster, params);
                for temp = 1:length(all)
                    classifiers{classifierIndex} = all{1,temp};
                    classifierIndex = classifierIndex + 1;
                end
            end
            
        end
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%% - END OF TRAINING
    
    for i=1:length(classifiers)
       classifiers{1,i}.chance =  3;
    end
    
    topPerformingClassifiers = rBCS(classifiers, validationData, params);
    avgAccuracy(fold) = fusion(classifiers, topPerformingClassifiers, testData);    
    avgAccuracy_without_BCS(fold) = fusion(classifiers, ones(1, length(classifiers)), testData);    
    K(fold) = noOfClusters;
    sizeAfterBCS(fold) = sum(topPerformingClassifiers);
    sizeBeforeBCS(fold) = length(classifiers);
end
rstx.avgAccuracy = mean(avgAccuracy);
rstx.avgAccuracy_without_BCS = mean(avgAccuracy_without_BCS);
rstx.K = mean(K);
rstx.originalSize = mean(sizeBeforeBCS);
rstx.sizeAfterBCS = mean(sizeAfterBCS);
rstx.stdDevWithBCS = std(avgAccuracy);
rstx.stdDevWithoutBCS = std(avgAccuracy_without_BCS);
end
