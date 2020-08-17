# imports for CDM function
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, chi2
from ipywidgets import interact, fixed, IntSlider, FloatSlider, SelectionSlider, Dropdown
from collections import Counter

# imports for Informed_LabelModel class
from snorkel.labeling.model import LabelModel
from snorkel.labeling.model.label_model import _CliqueData
from typing import Any, Dict
import torch
import networkx as nx
from itertools import chain, product
from scipy.sparse import issparse

class bcolors:
	""" Custom class for storing colours of warnings, errors, etc """
	HEADER = '\033[95m'
	OKBLUE = '\033[94m'
	OKGREEN = '\033[92m'
	WARNING = '\033[93m'
	FAIL = '\033[91m'
	ENDC = '\033[0m'

##########################################
# Conditional Dependence Monitor Main Code
##########################################
def CDM(L_dev, Y_dev, k = 2, sig = 0.01, policy = 'new', verbose = False, return_more_info = False):
	# create pd dataframe
	Complete_dev = np.concatenate((np.array([Y_dev]).T, L_dev), axis=1)
	df = pd.DataFrame(data=Complete_dev, columns=["GT"] + ["LF_"+str(i) for i in range(L_dev.shape[1])])

	def create_CT_tables(df, L_dev, k):
		"""create all combinations of contingency table's of {LF_i, LF_j, GT}"""
		CT_list = []
		for i in range(L_dev.shape[1]):
			for j in [k for k in range(i, L_dev.shape[1]) if k!=i]:
				CT = pd.crosstab([df['GT'], df['LF_'+str(i)]], df['LF_'+str(j)], margins = False) 
				CT = CT.reindex(index=[(m,n) for m in range(0,k) for n in range(-1,k)], columns=list(range(-1,k)), fill_value=0)
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
		interact(show_CT, q=IntSlider(min=1, max=len(CT_list), value=0, step=1), CT_list=fixed(CT_list));

	def get_conditional_dependencies(CT_list, sig, k, delta = 0):
		"""peform 3-way table chi-square independence test and obtain test statistic chi_square 
		(or corresponding p-value) for each CT table where each CT-table is a ({k+1}^{no of other LFs})x(k+1)x(k+1) matrix
		https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html"""
		CD_edges = []; CD_nodes = []; CD_edges_p_vals = []; p_vals_sum_dict = {}; CT_reduced_list = []
		count = 0; #n_bad = 0
		for CT in CT_list:
			count+=1; Z = k; # there are (k) GT values
			CT_reshaped = np.reshape(CT.values, (Z,k+1,k+1))
			
			if policy == 'old':
				p_val_tot, CT_reduced = get_p_total_old_policy(CT_reshaped, k, Z, delta, count, verbose, return_more_info) # Older policy of one round of 0 row/col reduction and then adding delta
			else:
				p_val_tot, CT_reduced = get_p_total_new_policy(CT_reshaped, k, Z, count, verbose, return_more_info) # newer policy of complete reduction
			CT_reduced_list.append(CT_reduced)
			# checking if total p_value is lesser than chosen sig
			if p_val_tot < sig: 
				digits_LF1 = CT.index.names[1][3:]
				digits_LF2 = CT.columns.name[3:] # 3rd index onwards to remove LF_ (3 characters)
				CD_edges.append( (int(digits_LF1), int(digits_LF2)) )
				CD_edges_p_vals.append(p_val_tot)

		#print info
		edges_info_dict = {}
		edges_info_dict['CD_edges'] = CD_edges; edges_info_dict['CD_edges_p_vals'] = CD_edges_p_vals
		if verbose:
			edges_df = pd.DataFrame(edges_info_dict)
			print(edges_df)

		return edges_info_dict, CT_reduced_list

	# retrieve modified CTs & tuples of LFs that are conditionally independent
	edges_info_dict, CT_reduced_list = get_conditional_dependencies(CT_list, sig = sig, k = k, delta = 1)

	# display reduced contingency tables' submatrices whose all elements = delta is not true
	if verbose:
		non_delta_tuple_indices = []
		for q in range(len(CT_reduced_list)):
			for r in range(len(CT_reduced_list[0])):
				delta = 1
				if ~(CT_reduced_list[q][r]==delta).all():
					non_delta_tuple_indices.append( ((q,r), (q,r)) ) # apending a tuple of tuples because first part of outer tuple is key and second is value passed gy slider to fn

		def show_CT_sub_matrices(t, CT_reduced_list):
			"""function to return qth CT at index q-1 of CT_list"""
			return CT_reduced_list[t[0]][t[1]]
		print("The reduced and modified contingency tables with non-delta values are given below")
		interact(show_CT_sub_matrices, t=Dropdown(description='index tuples (table number, submatrix number)', options=non_delta_tuple_indices), CT_reduced_list=fixed(CT_reduced_list));
	
	if return_more_info:
		return edges_info_dict
	else:
		return edges_info_dict['CD_edges']


##################################################################
# Heuristic Policies to reduce Contingency Tables and get P values
##################################################################
def get_p_total_old_policy(CT_reshaped, k, Z, delta, count, verbose, return_more_info):
	# check for "both 0 row and column" in all Z submatrices of size (k+1)x(k+1); reduce from Zx(k+1)x(k+1) to Zx(k)x(k) matrix
	zero_col_counter = np.zeros(k+1); zero_row_counter = np.zeros(k+1); 
	reduced_matrix = False
	# obtain count of no of 0 columns when (k+1)x(k+1) matrices are vstacked; ~ count of no of 0 rows when (k+1)x(k+1) matrices are hstacked
	# For eg.,			[[[a,b,0], [0,0,0], [0,0,0]],	gives		(zero_col_counter, zero_row_counter) as below
	#  2x3x3 matrix		 [[0,c,d], [0,0,0], [0,0,0]]]				[1,0,1], [0,2,2]
	for i in range(Z): 
		zero_col_counter += (CT_reshaped[i,:,:]==0).all(axis=0) # find bools representing 0 columns/vertical dirn (axis = 0); add bool result to counter
		zero_row_counter += (CT_reshaped[i,:,:]==0).all(axis=1) # similarly for row
	# checking if any elements of zero_col_counter and zero_row_counter are equal to Z (no of submatrices)
	#if (zero_col_counter==0).all() == False and (zero_row_counter==0).all() == False:
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
		if k == 3: # if submatrices are 4x4
			if min(len(zero_col_indices), len(zero_row_indices)) == 1: # if only 1 set of both 0 row & 0 col
				CT_reshaped_2 = np.zeros((Z,k,k))
				k_red = k
				for i in range(Z): 									   # reshape
					temp = np.delete(CT_reshaped[i,:,:], zero_col_indices[0], axis=1) # delete *first* 0 col in all Z submatrices of size (k+1)x(k+1)
					temp2 = np.delete(temp, zero_row_indices[0], axis=0)
					CT_reshaped_2[i,:,:] = temp2
			elif min(len(zero_col_indices), len(zero_row_indices)) == 2: # if 2 sets of both 0 row & 0 col
				CT_reshaped_2 = np.zeros((Z,k-1,k-1))
				k_red = k-1
				for i in range(Z): 									   # reshape
					temp = np.delete(CT_reshaped[i,:,:], zero_col_indices[0:2], axis=1) # delete *first two* 0 col in all Z submatrices of size (k+1)x(k+1)
					temp2 = np.delete(temp, zero_row_indices[0:2], axis=0)
					CT_reshaped_2[i,:,:] = temp2
		reduced_matrix = True

	if reduced_matrix:
		CT_to_use = CT_reshaped_2
		if verbose and return_more_info: print(CT_reshaped_2)
	else:
		CT_to_use = CT_reshaped
		k_red = k+1

	# check for any zero columns / rows in both (k)x(k) matrices in CT; if yes, add delta to all values
	bad_table = False
	for i,j in [(i,j) for i in range(Z) for j in [0,1]]: 
		if ~np.all(CT_to_use[i,:,:].any(axis=j)):
			bad_table = True
			#n_bad += 1
	if bad_table:
		if delta!=0:
			# to prevent 0 row/col in exp_freq table which in turn prevents division by 0 in statistic
			CT_to_use = CT_to_use + delta
			if verbose:
				print("Adding delta to table ", count)
		else:
			if verbose:
				print(bcolors.WARNING + "Error : table ",count," has a zero column/row in one (or both) of its 2x2 matrices!" + bcolors.ENDC)
			return 1

	# calculate statistic for each (k)x(k) matrix and sum
	chi2_sum = 0
	for i in range(Z):
		chi2stat, p, dof, exp_freq = chi2_contingency(CT_to_use[i,:,:])
		chi2_sum += chi2stat
	p_val_tot = 1-chi2.cdf(chi2_sum, Z*(k_red-1)*(k_red-1))
	return p_val_tot, CT_to_use

def get_p_total_new_policy(M, k, Z, count, verbose, return_more_info):
	def isSquare(m): 
		return all(len(row) == len(m) for row in m)
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
		if is0D(m_new):
			no_0D += 1
			#if verbose: print("skipping 0d")
			continue
		elif is1D(m_new):
			no_1D += 1 # assume all 1D reduced matrices are conditionally independent
			#if verbose: print("skipping 1d")
			continue
		elif isSquare(m_new):
			k_red = len(m_new)

			chi2stat, p, dof, exp_freq = chi2_contingency(m_new)
			chi2_sum += chi2stat
			dof_sum += (k_red-1)*(k_red-1)
		else: # not square, 1D, 0D ==> 2x3 or 3x2 matrices
			n_rows = m_new.shape[0]
			n_cols = m_new.shape[1]

			chi2stat, p, dof, exp_freq = chi2_contingency(m_new)
			chi2_sum += chi2stat
			dof_sum += (n_rows-1)*(n_cols-1)
	if no_0D+no_1D != Z: # if all reduced matrices are not 1d, 0d
		p_val_tot = 1-chi2.cdf(chi2_sum, dof_sum)
	else: 
		p_val_tot = 1 # to be considered independent, set any value > alpha here
	return p_val_tot, M_reduced


###############################
# Informed_LabelModel main code
###############################
class Informed_LabelModel(LabelModel):
	def __init__(self, edges, cardinality: int = 2, **kwargs: Any) -> None:
		super().__init__(cardinality, **kwargs)
		self.edges_given_as_input = edges
		
	def get_clique_tree_here(self, nodes, edges):
		"""
		Snorkel source code only allowed for chordal (traingulate-able) graphs. 
		Our modification allows for any graphs (but works in exp. time)! We used nx.algorithms.clique.find_cliques(G1)
		"""
		# Form the original graph G1
		G1 = nx.Graph()
		G1.add_nodes_from(nodes)
		G1.add_edges_from(edges)
		# Create empty graph to store changes
		G2 = nx.Graph()

		# reference : https://networkx.github.io/documentation/networkx-2.4/reference/algorithms/generated/networkx.algorithms.clique.find_cliques.html
		all_graph_cliques = list(nx.algorithms.clique.find_cliques(G1))
		all_graph_cs_sets = [set(inner_list) for inner_list in all_graph_cliques]

		for i, c in enumerate(all_graph_cs_sets):
			G2.add_node(i, members=c)
		for i in G2.nodes:
			for j in G2.nodes:
				S = G2.node[i]["members"].intersection(G2.node[j]["members"])
				w = len(S)
				if w > 0:
					G2.add_edge(i, j, weight=w, members=S)

		# Return a minimum spanning tree of G2
		return nx.minimum_spanning_tree(G2)
		
	# 
	def _create_tree(self) -> None:
		"""
		modified _create_tree function from below to include dependencies (edges) in undirected graphs
		Reference : https://github.com/snorkel-team/snorkel/blob/v0.9.4/snorkel/labeling/model/label_model.py
		"""
		nodes = range(self.m)
		edges = self.edges_given_as_input
		self.c_tree = self.get_clique_tree_here(nodes, edges)
		#self.metal_obj.c_tree = self.get_clique_tree_here(nodes, edges)

	def _create_L_ind_MeTaL(self, L):
		"""
		Reference for this borrowed function :
		https://github.com/HazyResearch/metal/blob/master/metal/label_model/label_model.py
		"""
		# TODO: Update LabelModel to keep L variants as sparse matrices
		# throughout and remove this line.
		if issparse(L):
			L = L.todense()

		L_ind = np.zeros((self.n, self.m * self.cardinality))
		for y in range(1, self.cardinality + 1):
			# A[x::y] slices A starting at x at intervals of y
			# e.g., np.arange(9)[0::3] == np.array([0,3,6])
			L_ind[:, (y - 1) :: self.cardinality] = np.where(L == y, 1, 0)
		return L_ind

	def _get_augmented_label_matrix_MeTaL(self, L, higher_order=False):
		"""
		Reference for this borrowed function :
		https://github.com/HazyResearch/metal/blob/master/metal/label_model/label_model.py
		"""
		self.c_data_MeTaL = {}
		for i in range(self.m):
			self.c_data_MeTaL[i] = {
				"start_index": i * self.cardinality,
				"end_index": (i + 1) * self.cardinality,
				"max_cliques": set(
					[
						j
						for j in self.c_tree.nodes()
						if i in self.c_tree.node[j]["members"]
					]
				),
			}
		L_ind = self._create_L_ind_MeTaL(L)
		# Get the higher-order clique statistics based on the clique tree
		# First, iterate over the maximal cliques (nodes of c_tree) and
		# separator sets (edges of c_tree)
		if higher_order:
			L_aug = np.copy(L_ind)
			for item in chain(self.c_tree.nodes(), self.c_tree.edges()):
				if isinstance(item, int):
					C = self.c_tree.node[item]
					C_type = "node"
				elif isinstance(item, tuple):
					C = self.c_tree[item[0]][item[1]]
					C_type = "edge"
				else:
					raise ValueError(item)
				members = list(C["members"])
				nc = len(members)

				# If a unary maximal clique, just store its existing index
				if nc == 1:
					C["start_index"] = members[0] * self.cardinality
					C["end_index"] = (members[0] + 1) * self.cardinality

				# Else add one column for each possible value
				else:
					L_C = np.ones((self.n, self.cardinality ** nc))
					for i, vals in enumerate(product(range(self.cardinality), repeat=nc)):
						for j, v in enumerate(vals):
							L_C[:, i] *= L_ind[:, members[j] * self.cardinality + v]

					# Add to L_aug and store the indices
					if L_aug is not None:
						C["start_index"] = L_aug.shape[1]
						C["end_index"] = L_aug.shape[1] + L_C.shape[1]
						L_aug = np.hstack([L_aug, L_C])
					else:
						C["start_index"] = 0
						C["end_index"] = L_C.shape[1]
						L_aug = L_C

					# Add to self.c_data_MeTaL as well
					id = tuple(members) if len(members) > 1 else members[0]
					self.c_data_MeTaL[id] = {
						"start_index": C["start_index"],
						"end_index": C["end_index"],
						"max_cliques": set([item]) if C_type == "node" else set(item),
					}
			return L_aug
		else:
			return L_ind
		
	def _get_augmented_label_matrix(self, L: np.ndarray, higher_order: bool = True) -> np.ndarray:
		"""
		We brought over _get_augmented_label_matrix from MeTaL's LabelModel.
		It unlike snorkel's LabelModel class actually changes L_aug based on dependency edges.
		"""
		#self.metal_obj._set_constants(L) # set constants like self.metal_obj.m and .t needed in below
		L_aug_from_metal = self._get_augmented_label_matrix_MeTaL(L, higher_order=True)
		
		# transfer clique data from self.c_data_MeTaL [dict of dicts] into self.c_data [dict of class objs]
		self.c_data: Dict[int, _CliqueData] = {}
		#for id in self.metal_obj.c_data.keys():
		for id in self.c_data_MeTaL.keys():
			#print(id, self.metal_obj.c_data[id]["start_index"], self.metal_obj.c_data[id]["end_index"], self.metal_obj.c_data[id]["max_cliques"])
			self.c_data[id] =  _CliqueData(
				start_index = self.c_data_MeTaL[id]["start_index"], 
				end_index = self.c_data_MeTaL[id]["end_index"], 
				max_cliques = self.c_data_MeTaL[id]["max_cliques"])
		return L_aug_from_metal

