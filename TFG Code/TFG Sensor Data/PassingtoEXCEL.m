clc;
clear;
close all;

%% Clay_1

% Load the data from an .mat archive
load('medidas_clay_15455552_20220101_20220331.mat');

% We convert the data from a structure to a matrix
Cells=struct2table(val);

% We transcript the data to an .csv archive
writetable(Cells,'Clay_1.csv','Delimiter',';');

%% Clay_2

% Load the data from an .mat archive
load('medidas_clay_15473012_20220101_20220331.mat');

% We convert the data from a structure to a matrix
Cells=struct2table(val);

% We transcript the data to an .csv archive
writetable(Cells,'Clay_2.csv','Delimiter',';');

%% Sand_1

% Load the data from an .mat archive
load('medidas_sand_15442283_20220101_20220331.mat');

% We convert the data from a structure to a matrix
Cells=struct2table(val);

% We transcript the data to an .csv archive
writetable(Cells,'Sand_1.csv','Delimiter',';');

%% Sand_2

% Load the data from an .mat archive
load('medidas_sand_15482234_20220101_20220331.mat');

% We convert the data from a structure to a matrix
Cells=struct2table(val);

% We transcript the data to an .csv archive
writetable(Cells,'Sand_2.csv','Delimiter',';');
