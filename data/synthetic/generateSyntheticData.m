clc;clear all;close all;
%Thomas Koch

%Synthetic Dataset made according to the description in the COPA paper by
%Vapnik

num = 1e3;
dim = 2; %number of features
num_train = .8 * num; %percentage of dataset to be training
num_test = num-num_train;
epsilon = 1e-1; %hyperparameter "noise parameter"

train = 2 * (rand(dim,num_train) - .5); %uniformly distributed
p_i = sum(train,1) + epsilon*randn(1,num_train); %privileged info
train_label = sum(train,1) > 0;


test = 2 * (rand(dim,num_test) - .5);
test_label = sum(test,1) > 0;

