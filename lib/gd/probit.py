import numpy as np
from scipy.stats import norm

# Nathan Englehart (adapted from Autumn 2022)

class probit_regression():
	
	def __init__(self, alpha=0.1, epoch=1000, l1_penalty = 0, l2_penalty = 0):
		
		""" Probit regression class based on sklearn functionality 
			
			Args:
				alpha::[Float]
					Learning rate for batch gradient descent algorithm

				epoch::[Int]
					Number of iterations for batch gradient descent algorithm

                l1_penalty::[Int]
                    L1 penalty for lasso (optional)

                l2_penalty::[Int]
                    L2 penalty for ridge (optional)
		"""

		self.alpha = alpha
		self.epoch = epoch
		self.l1_penalty = l1_penalty
		self.l2_penalty = l2_penalty

	def fit(self,X,t):

		""" Fits probit regression model with given regressor/train matrix and target vector 

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
	
		""" Performs batch gradient descent (also known as vanilla gradient descent) to find optimal coefficients for logit model within fit function

			Args:
				X::[Numpy Array]
					Regressor/train matrix that already has column of ones for intercept
				
				t::[Numpy Array]
					Target vector

		"""
		
		self.theta = np.zeros(X.shape[1]) 

		for i in range(self.epoch):
			
			gradient = (np.dot(X.T, (self.predict_proba(X) - t)) / t.size) + (self.l1_penalty * np.sign(self.theta)) + (self.l2_penalty * np.square(self.theta)) 
				
			self.theta -= (self.alpha * gradient)
		return self

	def Phi(self, z):
		
		""" Cumulative distribution function for the standard normal distribution. """
		
		return norm.cdf(-z) 

	def predict_proba(self, X):


		""" Generates probability predictions for the given matrix based on model

			Args:
				X::[Numpy Array]
					Test matrix that already has column of ones for intercept

		"""

		return 1 - self.Phi(np.dot(X,self.theta))

	def predict(self, X):
			
		""" Generates classification predictions for the given matrix based on the model 

			Args:
				X::[Numpy Array]
					Test matrix that already has column of ones for intercept

		"""

		return self.predict_proba(X).round()

	def log_likelihood(self,X,t,theta):
		
		""" Compute log likelihood given inputs. """

		y_star = np.dot(X,theta)
		return np.sum(-np.log(1+np.exp(y_star))) + np.sum(t * y_star) 
