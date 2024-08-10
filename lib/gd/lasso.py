import numpy as np

# Nathan Englehart (Summer, 2022)

class lasso_regression():
	
	def __init__(self, alpha=0.01, epoch=2500, lam=0.1):
		
		""" Lasso regression class based on sklearn functionality 
			
			Args:
				alpha::[Float]
					Learning rate for batch gradient descent algorithm

				epoch::[Int]
					Number of iterations for batch gradient descent algorithm

				lam::[Float]
					L1 penalty for lasso regression

		"""

		self.alpha = alpha
		self.epoch = epoch
		self.lam = lam

	def fit(self,X,t):

		""" Fits lasso regression model with given regressor/train matrix and target vector 

			Args:
				X::[Numpy Array]
					Regressor/train matrix that already has column of ones for intercept

				t::[Numpy Array]
					Target vector

		"""

		self.gd(X, t)
		self.coef_ = self.theta

		return self
	
	def gd(self, X, t):
	
		""" Performs batch gradient descent to find optimal coefficients for lasso model within fit function

			Args:
				X::[Numpy Array]
					Regressor/train matrix that already has column of ones for intercept
				
				t::[Numpy Array]
					Target vector

		"""

		self.theta = np.zeros(X.shape[1]) 

		for i in range(self.epoch):
			
			t_hat = self.predict(X)

			gradient = (np.dot(X.T, (t_hat - t)) / t.size) + (self.lam * np.sign(self.theta))
				
			self.theta = self.theta - (self.alpha * gradient)

		return self
	
	def predict(self, X):


		""" Generates predictions for the given matrix based on lasso model

			Args:
				X::[Numpy Array]
					Test matrix that already has column of ones for intercept

		"""

		return X.dot(self.theta) 



