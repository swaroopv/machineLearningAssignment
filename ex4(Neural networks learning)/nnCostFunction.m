function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
a1 = [ones(m, 1), X];%[m,n] = [5000,401]

z2 = a1 * Theta1' ; % [5000, 401]*[401 * 25] = [5000, 25]

% hidden Layer
a2 = sigmoid(z2);  
a2 = [ ones(size(a2, 1),1)  a2];

%output layer
z3 = a2 * Theta2'; % [5000 * 26] * [26 * 10]  = [5000 * 10]
a3 = sigmoid(z3); 

yd = eye(num_labels);
y = yd(y,:); % nice trick!!

hyp = a3;
Theta1WOBias = Theta1(:, 2:end);
Theta2WOBias = Theta2(:, 2:end);

Theta1WOBiasSquared = Theta1(:, 2:end) .^ 2;
Theta2WOBiasSquared = Theta2(:, 2:end) .^ 2;
J = 1/m .* sum(sum( -y .* log(hyp) - (1-y).*log(1 - hyp))) ...
    + lambda/(2*m) *  ( sum(Theta1WOBiasSquared(:)) + sum(Theta2WOBiasSquared(:)) ); % regularization term

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementatterion is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

Delta1 = 0;
Delta2 = 0;
for i = 1:m
	% step 1
	a1 = [1 X(i,:)]; %[ 1 * 401]
	z2 = Theta1 * a1'; % [25*401]*[401*1]=[25*1]
	a2 = [1; sigmoid(z2)]; %[26 * 1]
	z3 = Theta2 * a2; % [10*26][26*1] = [10*1]
	a3 = sigmoid(z3); %[10*1]

	% step 2
	yi = y(i,:);

	d3 = a3 - yi'; %[10*1]

	% step 3
	d2 = (Theta2WOBias' * d3) .* sigmoidGradient(z2);%[25*10][10*1]= [25,1]
	%d2 = d2(2:end);


	% step 4
	Delta2 = Delta2 + (d3 * a2');%[10*1][1*26]=[10*26] % here bias unit is included, so error of bias unit is 
                                                       %also calculated
	Delta1 = Delta1 + (d2 * a1);%[25*1][1*401]=[25*401]
end

Theta1_grad = (1/m) * Delta1;
Theta2_grad = (1/m) * Delta2;  %so bias term should not be included in regularization and delta calculation

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% -------------------------------------------------------------

% =========================================================================

Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + ((lambda / m) * Theta1WOBias);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + ((lambda / m) * Theta2WOBias);

% Unroll gradients

grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
