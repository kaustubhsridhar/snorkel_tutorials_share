# imports for DM function
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, chi2
from ipywidgets import interact, fixed, IntSlider, FloatSlider

# Simple Dependence Monitor Main Code
def DM(L_dev, sig = 0.01, verbose = False):
	# create pd dataframe
	df = pd.DataFrame(data=L_dev, columns=["LF_"+str(i) for i in range(L_dev.shape[1])])

	def create_CT_tables(df, L_dev):
		"""create all combinations of contingency table's of {LF_i, LF_j}"""
		CT_list = []
		for i in range(L_dev.shape[1]):
			for j in [k for k in range(i, L_dev.shape[1]) if k!=i]:
				CT = pd.crosstab(df['LF_'+str(i)], df['LF_'+str(j)], margins = False) 
				CT_list.append(CT)
		return CT_list

	def show_CT(k, CT_list):
		"""function to return CT at index k of CT_list"""
		return CT_list[k-1]

	# create and show all CT's with slider applet
	CT_list = create_CT_tables(df, L_dev) # create all combinations of Contingency Tables
	if verbose:
		print("No of tables = Choosing 2 from "+str(L_dev.shape[1])+" LFs = "+str(L_dev.shape[1])+"C2 = "+str(len(CT_list)))
		print("Note: Showing subtables where CI is not clearly evident")
		interact(show_CT, k=IntSlider(min=1, max=len(CT_list), value=0, step=1), CT_list=fixed(CT_list));

	class bcolors:
		""" Custom class for storing colours of warnings, errors, etc """
		HEADER = '\033[95m'
		OKBLUE = '\033[94m'
		OKGREEN = '\033[92m'
		WARNING = '\033[93m'
		FAIL = '\033[91m'
		ENDC = '\033[0m'

	def get_p_vals_222_tables(CT_list, delta = 0, sig = 0.05):
		"""peform 3-way table chi-square independence test and obtain test statistic chi_not-square 
		(or corresponding p-value) for each CT table where each CT-table has two 3x3 (i.e. 233) tables
		https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html"""
		p_val_list = []
		LFs_that_have_deps = []
		count = 0; n_bad = 0
		for CT in CT_list:
			count+=1
			CT_reshaped = np.reshape(CT.values, (1,2,2)) 
			
			# check for any zero columns / rows in both 2x2 matrices in CT
			bad_table = False
			for i,j in [(0,0), (0,1)]:
				if ~np.all(CT_reshaped[i,:,:].any(axis=j)):
					bad_table = True
					n_bad += 1
			if bad_table:
				if delta!=0:
					# to prevent 0 row/col in exp_freq table which in turn prevents division by 0 in statistic
					CT_reshaped = np.reshape(CT.values, (1,2,2)) + delta
					if verbose:
						print("Adding delta to table ", count)
				else:
					if verbose:
						print(bcolors.WARNING + "Error : table ",count," has a zero column/row in one (or both) of its 2x2 matrices!" + bcolors.ENDC)
					continue
			
			chi2stat1, p1, dof1, exp_freq1 = chi2_contingency(CT_reshaped[0,:,:])
			
			p_val_list.append(p1)
			# checking if total p_value is lesser than chosen sig
			if p1 < sig: 
				if verbose:
					print("table: {0:<15} chi-sq {1:<15} p-value: {2:<20} ==> ~({3} __|__ {4})".format(count, np.around(chi2stat1,4), np.around(p1,6), str(CT.index.name), str(CT.columns.name)))
				LFs_that_have_deps.append( (int(CT.index.name[-1]), int(CT.columns.name[-1])) )
			else:
				if verbose:
					print("table: {0:<15} chi-sq {1:<15} p-value: {2:<20}".format(count, np.around(chi2stat1,4), np.around(p1,6)))
		#print("\nSimple Dependecy Graph Edges: ", LFs_that_have_deps)
		
		if n_bad!=0 and delta == 0 and verbose:
			print(bcolors.OKBLUE+"\nNote"+bcolors.ENDC+": Either tune delta (currently "+str(delta)+") or increase datapoints in dev set to resolve"+bcolors.WARNING+" Errors"+bcolors.ENDC)
		
		return LFs_that_have_deps

	# retrieve CTs / combos of LFs that are conditionally independent
	LFs_that_have_deps = get_p_vals_222_tables(CT_list, delta = 1, sig = sig)
	
	return LFs_that_have_deps
