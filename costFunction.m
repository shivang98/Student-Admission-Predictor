function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples
J = 0;
grad = zeros(size(theta));

h = X * theta;
J = (1 / m) * sum(-y' * log(sigmoid(h)) - (1-y)' * log(1-sigmoid(h)));

% grad = (1 / m) * (X' * (sigmoid(h)-y));
grad = (1 / m) * sum((sigmoid(h)-y).*X)';

end
