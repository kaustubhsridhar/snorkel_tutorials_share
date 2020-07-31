# imports for ModVarma_InCov function
import numpy as np

################################################################################################
# Modified Varma et al Dependence Monitor Main Code that thresholds inverse of covariance matrix
################################################################################################
def ModVarma_InCov(L_dev, Y_dev, thresh=1.5):
	# create pd dataframe
	mod_varma_deps = []
	count = 0
	for I in range(L_dev.shape[1]):
		for J in [k for k in range(I, L_dev.shape[1]) if k!=I]:
			count += 1
			M = np.zeros((L_dev.shape[0],3))
			M[:,0] = L_dev[:,I]; M[:,1] = L_dev[:,J]
			M[:,2] = np.array(Y_dev).T
			P = np.cov(M, rowvar=False)
			if np.linalg.det(P) == 0: # singular P ==> not conditional dependency
				continue
			K = np.linalg.inv(P)
			# Check if K[0,1] (aka K[1,0]) is greater than threshold 
			# [0,1] corresponds to LF_I \perp LF_J | Y_gold
			if np.abs(K[0,1]) > thresh:
				mod_varma_deps.append((I, J))

	return mod_varma_deps