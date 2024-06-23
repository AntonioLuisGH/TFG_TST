clc;
clear;
close all;

%% Clay_1
mat_to_csv('medidas_clay_15455552_20220101_20220331.mat', 'Clay_1.csv');

%% Clay_2
mat_to_csv('medidas_clay_15473012_20220101_20220331.mat', 'Clay_2.csv');

%% Sand_1
mat_to_csv('medidas_sand_15442283_20220101_20220331.mat', 'Sand_1.csv');

%% Sand_2
mat_to_csv('medidas_sand_15482234_20220101_20220331.mat', 'Sand_2.csv');
