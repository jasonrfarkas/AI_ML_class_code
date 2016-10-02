## Machine learning Project
import pandas as pd
import numpy as np
import os
import sys
import random

import ipdb, pdb
from PdbSublimeTextSupport import preloop, precmd
pdb.Pdb.preloop = preloop
pdb.Pdb.precmd = precmd

try:
    from ipdb.__main__ import Pdb as ipdb_Pdb
except ImportError:
    pass
else:
    ipdb_Pdb.preloop = preloop
    ipdb_Pdb.precmd = precmd

"""
notes 

k centroids # set to 3, 
each centroid has a value in each dimention
If I have a dataframe of all datapoints and their given group
I would also need a table of centroid and their means for each dimention

to update the mean of each dimention I would need to just grab all that contain the group and average their stats

to update the group of each point I would need to take each datapoint and compute the distance to each group somehow, grab the min one gorup number 
and set it to that. check equality to the previous version, if it equal stop, else repeat

(minimize) IV = the sum of the distance between each cluster and its datapoints
(maximize) EV = 1 '\ ' len(df) 
something like caclulating the difference between every datapoint provided that they are from different clusters
 

"""


class K_Means(object):

	def __init__(self, data, number_of_centroids=3, iterations=5, starting_centroids=None):
		self.original_data = data.copy()
		self.real_cluster_numer = data[7]
		self.original_data.drop(labels=7, axis=1, inplace=True)
		self.current_iteration = 0
		self.number_of_centroids = number_of_centroids if not starting_centroids else len(starting_centroids)
		self.starting_centroids = starting_centroids
		self.iterations_holder=[]
		self.k_means_alg()

	def k_means_alg(self):
		for i in range(self.iterations):
			self.k_means_iteration()
		## SELECT ITTERATION WITH BEST IV/EV
		ipdb.set_trace()
		pass

	def update_centroids(self, old_data, centroids):
		data=old_data.copy()
		new_centroids = data.groupby(["centroid"]).mean()
		data["centroid"] = data.apply(lambda x: self.get_reassigned_group(x, new_centroids), axis=1)
		return data, new_centroids

	def get_reassigned_group(self, row, centroids):
		group_number = (((centroids.sub(row, axis=1))**2).sum(axis=1)**.5).argmin()
		return group_number

	def setup_iteration(self):
		data = self.original_data.copy()
		data["centroid"] = -1
		self.itteration_info = {}
		if self.starting_centroids:
			centroids = self.starting_centroids
		else:
			centroids = pd.concat([pd.Series(data.apply(lambda x: random.uniform(x.min(),\
				x.max()), axis=0), name=i) for i in range(self.number_of_centroids)], axis=1)
			centroids = centroids.transpose().drop(labels="centroid", axis=1)
		self.itteration_info["starting_centroids"] = centroids.copy()
		data["centroid"] = data.apply(lambda x: self.get_reassigned_group(x, centroids), axis=1) 
		return data, centroids

	def k_means_iteration(self):
		data, centroids = self.setup_iteration()
		updated_data, updated_centroids = self.update_centroids(data, centroids)
		ipdb.set_trace()
		while not centroids.equals(updated_centroids):
			data, centroids = updated_data, updated_centroids
			updated_data, updated_centroids = self.update_centroids(data, centroids)
		self.itteration_info["final_centroids"] = centroids.copy()
		self.itteration_info["IV, EV, IV/EV"] = (0,0,0)
		self.iterations_holder.append(self.itteration_info)
		#starting centroids, final centroids, IV, EV, and IV/EV 

original_data = pd.read_table(os.path.join(os.getcwd(), "original_data.txt"), sep="\t", header=None)
k_means_alg = K_Means(original_data)




































