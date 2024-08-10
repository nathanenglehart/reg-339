import numpy as np

# Nathan Englehart (Summer, 2022)

class poisson_regression():
	
	def __init__(self, alpha=0.1, epoch=1000):
		
		""" Poisson regression class based on sklearn functionality 
			
			Args:
				alpha::[Float]
					Learning rate for batch gradient descent algorithm

				epoch::[Int]
					Number of iterations for batch gradient descent algorithm

		"""

		self.alpha = alpha
		self.epoch = epoch

	def fit(self,X,t):

		""" Fits poisson regression model with given regressor/train matrix and target vector 

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
	
		""" Performs batch gradient descent (also known as vanilla gradient descent) to find optimal coefficients for poisson model within fit function

			Args:
				X::[Numpy Array]
					Regressor/train matrix that already has column of ones for intercept
				
				t::[Numpy Array]
					Target vector

		"""
		
		self.theta = np.zeros(X.shape[1]) 

		for i in range(self.epoch):
			
			gradient = np.dot(X.T, (self.predict_proba(X) - t)) / t.size
				
			self.theta -= (self.alpha * gradient)

		return self

"""
	def sigmoid(self,z):
		
		""" Sigmoid function (also called the logistic function) where P(t = 1) = sigma(coefs * x) and P(t = 0) = 1 - sigma(coefs * x) """

		return 1.0 / (1.0 + np.exp(z))
"""

    def poisson_pmf(self,h):
        
        return () / np.factorial(h) 

	def predict_proba(self, X):


		""" Generates probability predictions for the given matrix based on model

			Args:
				X::[Numpy Array]
					Test matrix that already has column of ones for intercept

		"""

		return 1 - self.sigmoid(np.dot(X,self.theta))

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
