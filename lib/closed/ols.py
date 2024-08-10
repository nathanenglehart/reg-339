import numpy as np

# Nathan Englehart (Summer, 2022)

class ols_regression():

  def __init__(self, l2_penalty):
      
      """ OLS regression class based on sklearn functionality (for modular use) 
          Args:

              l2_penalty::[[Float]]
                L2 penalty for lasso regression (optional)

      """

      self.l2_penalty = l2_penalty

  def fit(self, X, t):

      """ Fits ridge regression model with given regressor/train matrix and target vector
      
		Args:
			
			X::[Numpy Array]
				Regressor/train matrix (must be built before entering as parameter with method such as sklearn.PolynomialFeatures fit_transform)
			
			t::[Numpy Array]
				Target vector
      """

      theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(t)
      
      if(self.l2_penalty != 0): I = np.identity(X.shape[1]); I[0,0] = 0; lam_matrix = self.l2_penalty * I; theta = np.linalg.inv(X.T.dot(X) + lam_matrix).dot(X.T).dot(t)
 
      self.theta = theta
      self.coef_ = theta
      
      return self

  def predict(self, X):
      
      """ Generates predictions for the given matrix based on model
      
      		Args:
			
			X::[Numpy Array]
				Test matrix (must be built before entering as parameter with method such as sklearn.PolynomialFeatures fit_transform)
      """

      self.predictions = X.dot(self.theta)

      return self.predictions
