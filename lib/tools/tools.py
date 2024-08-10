import numpy as np

def compute_classification_error_rate(t,t_hat):

	""" Computes error rate for classification methods such as logistic regression.

		Args:
			
			t::[Numpy Array]
				Truth values

			t_hat::[Numpy Array]
				Prediction values
	
	"""

	error_rate = 0

	for i in range(len(t)):
		
		if(t[i] != t_hat[i]):
			error_rate += 1
	
	return error_rate / len(t) 

def efron_r_squared(t,t_probs):
	""" Returns Efron's psuedo R-Squared for logistic regression. 

		Args:

			t::[Numpy Array]
				Truth values

			t_probs::[Numpy Array]
				Prediction value probabilities

	"""

	return 1.0 - ( np.sum(np.power(t - t_probs, 2.0)) / np.sum(np.power((t - (np.sum(t) / float(len(t)))), 2.0)) ) 

def mcfadden_r_squared(theta, X, t, model):

	""" Returns McFadden's psuedo R-Squared for logistic regression 
	
		Args:
			
			theta::[Numpy Array]
				Weights/coefficients for the given logistic regression model
			
			X::[Numpy Array]
				Regressor matrix

			t::[Numpy Array]
				Truth values corresponding to regressor matrix

	"""

	L_ul = model.log_likelihood(X,t,theta)
	theta_0 = np.zeros(theta.size)
	theta_0[0] = theta[0]
	L_0 = model.log_likelihood(X, t, theta_0)

	return 1 - (L_ul / L_0)

def r_squared(t, t_hat):
	
	""" Returns R-Squared for model with given truth values and prediction values. 

		Args:
			
			t::[Numpy Array]
				Truth values

			t_hat::[Numpy Array]
				Prediction values

	"""

	t_bar = t.mean()

	return 1 - ((((t-t_hat)**2).sum()) / (((t-t_bar)**2).sum()))

def adj_r_squared(t,t_hat,m):

	""" Returns adjusted R-Squared for model with given truth values and prediction values.

		Args:
				
			t::[Numpy Array]
				Truth values

			t_hat::[Numpy Array]
				Prediction values
			
			m::[Integer]
				Number of features in the dataset
	
	"""

	n = len(t)
	return 1 - ((1 - r_squared(t,t_hat)) * ((n-1) / (n-m)))

