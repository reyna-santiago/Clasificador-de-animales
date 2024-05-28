clear all, close all, clc
imds = imageDatastore("DATASET", ...
    'IncludeSubfolders',true, 'LabelSource','foldernames');

[imdsTrain,imdsValidation] = splitEachLabel(imds,0.9,...
    'randomized');
%%

layers=[ imageInputLayer([227 227 1]) %TAMAÑO DE
    convolution2dLayer(3,8,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,64,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,128,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,256,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,512,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,1024,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    convolution2dLayer(3,2048,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    convolution2dLayer(3,4096,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(3) %Numero de clases
    fullyConnectedLayer(3) %Numero de clases
    fullyConnectedLayer(3) %Numero de clases

    softmaxLayer
    classificationLayer]; 



% Especificar las opciones de entrenamiento
options = trainingOptions('sgdm', 'InitialLearnRate',0.01, ...
 'MaxEpochs', 100, ...
 'Shuffle','every-epoch',...
 'ValidationData',imdsValidation,...
 'ValidationFrequency', 10,...
 'Verbose', false, ...
 'Plots','training-progress');


net = trainNetwork(imdsTrain,layers,options);

[YPred,probs] = classify(net,imdsValidation);
YValidation = imdsValidation.Labels;
accuracy = sum(YPred == YValidation)/numel(YValidation)

%% Prueba con una imagen 
I = imread("DATASET\1.jpg");
[YPred,probs] = classify(net,I);
imshow(I)
label=YPred;
title(string(label)+ ", "+num2str(100*max(probs),3)+"%");


%para salvar save('net.mat','net')
