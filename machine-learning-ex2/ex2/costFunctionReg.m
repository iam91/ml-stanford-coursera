function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h = sigmoid(X * theta);
yy = [-y; -(1 - y)];
xx = [log(h); log(1 - h)];
theta2 = theta .^ 2;

J = yy' * xx / m + lambda * sum(theta2(2:end)) / 2 / m;

mask = ones(size(theta), 1);
mask(1) = 0;
grad = sum(X' * (h - y), 2) / m + (theta .* mask) * lambda / m;

% =============================================================

end
