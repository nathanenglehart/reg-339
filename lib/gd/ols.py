import numpy as np

# Nathan Englehart (Summer, 2022)

class ols_regression():
	
	def __init__(self, alpha=0.01, epoch=2500, l1_penalty=0, l2_penalty=0):
		
		""" OLS regression class based on sklearn functionality 
			
			Args:
				alpha::[Float]
					Learning rate for batch gradient descent algorithm

				epoch::[Int]
					Number of iterations for batch gradient descent algorithm

				l1_penalty::[Float]
					L1 penalty for lasso regression (optional)

                l2_penalty::[[Float]]
                    L2 penalty for ridge regression (optional)

		"""

		self.alpha = alpha
		self.epoch = epoch
		self.l1_penalty = l1_penalty
		self.l2_penalty = l2_penalty

	def fit(self,X,t):

		""" Fits OLS regression model with given regressor/train matrix and target vector 

			Args:
				X::[Numpy Array]
					Regressor/train matrix that already has column of ones for intercept

				t::[Numpy Array]
					Target vector

		"""

		self.bgd(X, t)
		self.coef_ = self.theta

		return self
	
	def bgd(self, X, t):
	
		""" Performs batch gradient descent to find optimal coefficients for OLS model within fit function

			Args:
				X::[Numpy Array]
					Regressor/train matrix that already has column of ones for intercept
				
				t::[Numpy Array]
					Target vector

		"""

		self.theta = np.zeros(X.shape[1]) 

		for i in range(self.epoch):
			
			t_hat = self.predict(X)

			gradient = (np.dot(X.T, (t_hat - t)) / t.size) + (self.l1_penalty * np.sign(self.theta)) + (self.l2_penalty * np.square(self.theta))
				
			self.theta = self.theta - (self.alpha * gradient)

		return self
	
	def predict(self, X):


		""" Generates predictions for the given matrix based on OLS model

			Args:
				X::[Numpy Array]
					Test matrix that already has column of ones for intercept

		"""

		return X.dot(self.theta) 



