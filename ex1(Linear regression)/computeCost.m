function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
hyp = X * theta; % m*2 * 2*1 = m*1  //calulating hypothesis
eqn1 = hyp - y;
eqn2 = sum(eqn1.^2);
J = eqn2/2/m;



% =========================================================================

end
