# imports for CDM function
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, chi2
from ipywidgets import interact, fixed, IntSlider, FloatSlider, SelectionSlider, Dropdown
import itertools
from collections import Counter

class bcolors:
	""" Custom class for storing colours of warnings, errors, etc """
	HEADER = '\033[95m'
	OKBLUE = '\033[94m'
	OKGREEN = '\033[92m'
	WARNING = '\033[93m'
	FAIL = '\033[91m'
	ENDC = '\033[0m'

# Conditional Dependence Monitor Main Code
def NM_fast(L_dev, Y_dev, k = 2, sig = 0.01, policy = 'new', verbose = False, return_more_info = False):
	# create pd dataframe
	Complete_dev = np.concatenate((np.array([Y_dev]).T, L_dev), axis=1)
	df = pd.DataFrame(data=Complete_dev, columns=["GT"] + ["LF_"+str(i) for i in range(L_dev.shape[1])])
	no_other_LFs = L_dev.shape[1]-2

	def create_CT_tables(df, L_dev, k):
		"""create all combinations of contingency table's of {LF_i, LF_j, [LF_all others]}"""
		CT_list = []
		for i in range(L_dev.shape[1]):
			for j in [k for k in range(i, L_dev.shape[1]) if k!=i]:
				other_LFs_list = [df['LF_'+str(m)] for m in range(L_dev.shape[1]) if (m!=i and m!=j)]
				CT = pd.crosstab([df['GT']] + other_LFs_list + [df['LF_'+str(i)]], df['LF_'+str(j)], margins = False) 
				#k_list = [i-1 for i in range(k+1)]; GT_indices = [i for i in range(k)]
				#CT = CT.reindex(index=list(itertools.product(GT_indices, *[tuple(k_list)] * (no_other_LFs+1))), columns=k_list, fill_value=0)

				# prep to reindex only the LF column closest to values (this is new in the fast version)
				indices_other_LFs = [] # first get current indices of other LF columns
				for itera in range(len(CT.index)):
					indices_other_LFs.append(CT.index[itera][:-1])
				indices_other_LFs = list(set(indices_other_LFs)) # get unique values only
				indices_closest_LF = [(i-1,) for i in range(k+1)]
				all_indices = [ ind1+ind2 for ind1 in indices_other_LFs for ind2 in indices_closest_LF]
				k_list = [i-1 for i in range(k+1)]
				# reindex only the LF column closest to values
				CT = CT.reindex(index=all_indices, columns=k_list, fill_value=0)
				CT_list.append(CT)
		return CT_list
	
	def show_CT(q, CT_list):
		"""function to return qth CT at index q-1 of CT_list"""
		return CT_list[q-1]

	# create and show all CT's (if verbose) with slider applet
	CT_list = create_CT_tables(df, L_dev, k) # create all combinations of Contingency Tables
	if verbose:
		print("No of tables = Choosing 2 from "+str(L_dev.shape[1])+" LFs = "+str(L_dev.shape[1])+"C2 = "+str(len(CT_list)))
		#print("Note: Showing subtables where CI is not clearly evident")
		interact(show_CT, q=IntSlider(min=1, max=len(CT_list), value=0, step=1), CT_list=fixed(CT_list))

	def get_conditional_deps(CT_list, sig, k, delta = 0):
		"""peform 3-way table chi-square independence test and obtain test statistic chi_square 
		(or corresponding p-value) for each CT table where each CT-table is a ({k+1}^{no of other LFs})x(k+1)x(k+1) matrix
		https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html"""
		CD_edges = []; CD_nodes = []; CD_edges_p_vals = []; p_vals_sum_dict = {}; CT_reduced_list = []
		count = 0; #n_bad = 0
		for CT in CT_list:
			count+=1; Z = int(len(CT.values)/3) # no of rows/3 (this is new in the fast version)
			CT_reshaped = np.reshape(CT.values, (Z,k+1,k+1)) 

			if policy == 'old':
				p_val_tot, CT_reduced = get_p_total_old_policy(CT_reshaped, k, Z, delta, sig, count, verbose, return_more_info) # Older policy of one round of 0 row/col reduction and then adding delta
			else:
				p_val_tot, CT_reduced = get_p_total_new_policy(CT_reshaped, k, Z, sig, count, verbose, return_more_info) # newer policy of complete reduction
			CT_reduced_list.append(CT_reduced)
			# checking if total p_value is lesser than chosen sig
			if p_val_tot < sig: 
				digits_LF1 = CT.index.names[1+no_other_LFs][3:] # add 1 for the GT column
				digits_LF2 = CT.columns.name[3:] # 3rd index onwards to remove LF_ (3 characters)
				CD_edges.append( (int(digits_LF1), int(digits_LF2)) )
				CD_edges_p_vals.append(p_val_tot)

		#printing info
		edges_info_dict = {}
		edges_info_dict['CD_edges'] = CD_edges; edges_info_dict['CD_edges_p_vals'] = CD_edges_p_vals
		if verbose:
			edges_df = pd.DataFrame(edges_info_dict)
			print(edges_df)

		return edges_info_dict, CT_reduced_list

	# retrieve modified CTs & tuples of LFs that are conditionally independent
	edges_info_dict, CT_reduced_list = get_conditional_deps(CT_list, sig = sig, k = k, delta = 1)

	# display reduced contingency tables' submatrices whose all elements = delta is not true
	if verbose:
		non_delta_tuple_indices = []
		for q in range(len(CT_reduced_list)):
			for r in range(len(CT_reduced_list[q])):
				delta = 1
				if ~(CT_reduced_list[q][r]==delta).all():
					non_delta_tuple_indices.append( ((q,r), (q,r)) ) # apending a tuple of tuples because first part of outer tuple is key and second is value passed gy slider to fn

		def show_CT_sub_matrices(t, CT_reduced_list):
			"""function to return qth CT at index q-1 of CT_list"""
			return CT_reduced_list[t[0]][t[1]]
		print("The reduced and modified contingency tables with non-delta values are given below")
		interact(show_CT_sub_matrices, t=Dropdown(description='index tuples (table number, submatrix number)', options=non_delta_tuple_indices), CT_reduced_list=fixed(CT_reduced_list))
	
	if return_more_info:
		return edges_info_dict
	else:
		return edges_info_dict['CD_edges']


##################################################################
# Heuristic Policies to reduce Contingency Tables and get P values
##################################################################
def get_p_total_old_policy(CT_reshaped, k, Z, delta, sig, count, verbose, return_more_info):
	# check for "both 0 row and column" in all Z submatrices of size (k+1)x(k+1); reduce from Zx(k+1)x(k+1) to Zx(k)x(k) matrix
	zero_col_counter = np.zeros(k+1); zero_row_counter = np.zeros(k+1)
	reduced_matrix = False
	# obtain count of no of 0 columns when (k+1)x(k+1) matrices are vstacked; ~ count of no of 0 rows when (k+1)x(k+1) matrices are hstacked
	# For eg.,			[[[a,b,0], [0,0,0], [0,0,0]],	gives		(zero_col_counter, zero_row_counter) as below
	#  2x3x3 matrix		 [[0,c,d], [0,0,0], [0,0,0]]]				[1,0,1], [0,2,2]
	for i in range(Z): 
		zero_col_counter += (CT_reshaped[i,:,:]==0).all(axis=0) # find bools representing 0 columns/vertical dirn (axis = 0); add bool result to counter
		zero_row_counter += (CT_reshaped[i,:,:]==0).all(axis=1) # similarly for row
	# checking if any elements of zero_col_counter and zero_row_counter are equal to Z (no of submatrices)
	# ie checking for atleast 1 zero row and col
	if (zero_col_counter==Z).any() and (zero_row_counter==Z).any():
		zero_col_indices = np.where(zero_col_counter == Z)[0] # get indices of cols that are 0 in all Z submatrices of size (k+1)x(k+1)
		zero_row_indices = np.where(zero_row_counter == Z)[0] # similarly for row
		if k == 2: # if submatrices are 3x3
		#if min(len(zero_col_indices), len(zero_row_indices)) == 1: # if only 1 set of both 0 row & 0 col
			CT_reshaped_2 = np.zeros((Z,k,k))
			k_red = k
			for i in range(Z): 									   # reshape
				temp = np.delete(CT_reshaped[i,:,:], zero_col_indices[0], axis=1) # delete *first* 0 col in all Z submatrices of size (k+1)x(k+1)
				temp2 = np.delete(temp, zero_row_indices[0], axis=0)
				CT_reshaped_2[i,:,:] = temp2
				# if submatrix still has a zero row/col, then add delta
				if ~np.all(CT_reshaped_2[i,:,:].any(axis=0)) or ~np.all(CT_reshaped_2[i,:,:].any(axis=1)):
					CT_reshaped_2[i,:,:] += delta
			
		if k == 3: # if submatrices are 4x4
			if min(len(zero_col_indices), len(zero_row_indices)) == 1: # if only 1 set of both 0 row & 0 col
				CT_reshaped_2 = np.zeros((Z,k,k))
				k_red = k
				for i in range(Z): 									   # reshape
					temp = np.delete(CT_reshaped[i,:,:], zero_col_indices[0], axis=1) # delete *first* 0 col in all Z submatrices of size (k+1)x(k+1)
					temp2 = np.delete(temp, zero_row_indices[0], axis=0)
					CT_reshaped_2[i,:,:] = temp2
					# if submatrix still has a zero row/col, then add delta
					if ~np.all(CT_reshaped_2[i,:,:].any(axis=0)) or ~np.all(CT_reshaped_2[i,:,:].any(axis=1)):
						CT_reshaped_2[i,:,:] += delta
			elif min(len(zero_col_indices), len(zero_row_indices)) == 2: # if 2 sets of both 0 row & 0 col
				CT_reshaped_2 = np.zeros((Z,k-1,k-1))
				k_red = k-1
				for i in range(Z): 									   # reshape
					temp = np.delete(CT_reshaped[i,:,:], zero_col_indices[0:2], axis=1) # delete *first two* 0 col in all Z submatrices of size (k+1)x(k+1)
					temp2 = np.delete(temp, zero_row_indices[0:2], axis=0)
					CT_reshaped_2[i,:,:] = temp2
					# if submatrix still has a zero row/col, then add delta
					if ~np.all(CT_reshaped_2[i,:,:].any(axis=0)) or ~np.all(CT_reshaped_2[i,:,:].any(axis=1)):
						CT_reshaped_2[i,:,:] += delta
		reduced_matrix = True

	if reduced_matrix:
		CT_to_use = CT_reshaped_2
		#if verbose and return_more_info: print(CT_reshaped_2)
	else:
		CT_to_use = CT_reshaped
		k_red = k+1

	# calculate statistic for each (k)x(k) matrix and sum
	chi2_sum = 0; actual_Z = 0
	for i in range(Z):
		if ~((CT_to_use[i,:,:]==delta).all()): # if not (all elements of kxk matrix are 0+delta), then
			chi2stat, p, dof, exp_freq = chi2_contingency(CT_to_use[i,:,:])
			if p<sig: # if any submatrix is dependent, whole CT is dependent
				if verbose: print("left at submatrix ", i," of CT ", count)
				return p, CT_to_use
			chi2_sum += chi2stat; actual_Z += 1
	if verbose: print("There are ",Z-actual_Z,"/",Z," ",k,"x",k," zero matrices that weren't used in chi^2 computation")
	p_val_tot = 1-chi2.cdf(chi2_sum, actual_Z*(k_red-1)*(k_red-1))
	return p_val_tot, CT_to_use

def get_p_total_new_policy(M, k, Z, sig, count, verbose, return_more_info):
	def is0D(m):
		return m.shape[0] == 0 or m.shape[1] == 0
	def is1D(m):
		return m.shape[0] == 1 or m.shape[1] == 1
	def remove_all_0_rows_cols(m):
		# remove all 0 columns
		m2 = m[:, ~np.all(m == 0, axis=0)]
		# remove all 0 rows
		m3 = m2[~np.all(m2 == 0, axis=1)] 
		return m3
	#if verbose: print(count)
	chi2_sum = 0
	dof_sum = 0; no_1D = 0; no_0D = 0; M_reduced = []
	for i in range(Z):
		m_new = remove_all_0_rows_cols(M[i])
		M_reduced.append(m_new)
		#if verbose: print(m_new)

	for i in range(Z):
		m_new = M_reduced[i]
		if is0D(m_new):
			no_0D += 1
			#if verbose: print("skipping 0d")
			continue
		elif is1D(m_new):
			no_1D += 1 # assume all 1D reduced matrices are conditionally independent
			#if verbose: print("skipping 1d")
			continue
		else: # square and not square matrices (~2x3 or 3x2)
			n_rows = m_new.shape[0]
			n_cols = m_new.shape[1]

			chi2stat, p, dof, exp_freq = chi2_contingency(m_new)
			if p<sig: # if any submatrix is dependent, whole CT is dependent
				if verbose: print("left at submatrix ", i," of CT ", count)
				return p, M_reduced
			chi2_sum += chi2stat
			dof_sum += (n_rows-1)*(n_cols-1)
	if no_0D+no_1D != Z: # if all reduced matrices are not 1d, 0d
		p_val_tot = 1-chi2.cdf(chi2_sum, dof_sum)
	else: 
		p_val_tot = 1 # to be considered independent, set any value > alpha here
	return p_val_tot, M_reduced