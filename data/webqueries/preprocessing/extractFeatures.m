%% Extract visual deep learning features from jpg images

clear all; close all; clc;

% Install and compile MatConvNet (needed once).
untar('http://www.vlfeat.org/matconvnet/download/matconvnet-1.0-beta25.tar.gz') ;
cd matconvnet-1.0-beta25
run matlab/vl_compilenn ;

% Download a pre-trained CNN from the web (needed once).
urlwrite(...
  'http://www.vlfeat.org/matconvnet/models/imagenet-vgg-s.mat', ...
  'imagenet-vgg-s.mat') ;

% Setup MatConvNet.
run matlab/vl_setupnn ;

% Load a model and upgrade it to MatConvNet current version.
net = load('imagenet-vgg-s.mat') ;
net = vl_simplenn_tidy(net) ;

outerDir = '../../data/webqueries/train/' ;

for i = 0:354
    innerDir = strcat('query_', int2str(i), '/') ;
    files = dir(strcat(outerDir, innerDir, '*.jpg')) ;
    
    for file = files'
        file.name
        
        try
            % Obtain and preprocess an image.
            im = imread(strcat(outerDir, innerDir, file.name)) ;
            im_ = single(im) ; % note: 255 range
            im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
            im_ = im_ - net.meta.normalization.averageImage ;

            % Run the CNN.
            res = vl_simplenn(net, im_) ;
            
            % Save results as csv.
            % FIXME not sure if this res(end-4) is the fc6 layer...
            csvwrite(strcat(outerDir, innerDir, file.name(1:end-4), '.csv'), res(end-4).x) ;
        catch
            warning(file.name) ;
        end
    end
end
