function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% set minError to be a very large value initially
minError = 100000000000;

for i = 0:8
	for j = 0:8
		% increase by multiple of three each time
		tempC = 0.01 .* (3 .^ i);
		tempSigma = 0.01 .* (3 .^ j);
		% train the model with this combo of C and sigma
		model = svmTrain(X, y, tempC, @(x1, x2)gaussianKernel(x1, x2, tempSigma));
		% compute the error for this combo of C and sigma
		predictions = svmPredict(model, Xval);
		predictionError = mean(double(predictions ~= yval));
		if (predictionError < minError)
			C = tempC;
			sigma = tempSigma;
			minError = predictionError;
		endif
	endfor
endfor


% =========================================================================

end
