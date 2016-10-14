## Machine learning Project
import pandas as pd
import numpy as np
import os
import sys
import random
# import matplotlib
# import matplotlib.pyplot as plt
# matplotlib.style.use('ggplot')

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
 meaning sum the distances between elements from different clusters and divide the result by number of elements

"""



class K_Means(object):

	def __init__(self, data, number_of_centroids=3, iterations=5, starting_centroids=None):
		self.original_data = data.copy()
		self.real_cluster_number = data[7]
		self.original_data.drop(labels=7, axis=1, inplace=True)
		self.current_iteration = 0
		self.number_of_centroids = number_of_centroids if starting_centroids is None else len(starting_centroids)
		self.starting_centroids = starting_centroids
		self.iterations_holder=[]
		self.iterations = iterations
		self.k_means_alg()

	def k_means_alg(self):
		self.highest_iv_ev = -1
		self.best_iteration = {}
		for i in range(self.iterations):
			self.k_means_iteration()
		return self.best_iteration

	def get_cluster_info(self):
		return self.best_iteration

	def update_centroids(self, old_data, centroids):
		data=old_data.copy()
		new_centroids = centroids.copy()
		new_centroids.update(data.groupby(["centroid"]).mean())
		## I NEED TO RETAIN THE OLD CENTROIDS AND FILL IN THE NEW ONES WHEN THEY MAY BE MISSING, SINCE SOME CENTROIDS MAY NOT BE APPLIED
		data["centroid"] = data.apply(lambda x: self.get_reassigned_group(x, new_centroids), axis=1)
		return data, new_centroids

	def get_reassigned_group(self, row, centroids):
		group_number = (((centroids.sub(row, axis=1))**2).sum(axis=1)**.5).argmin()
		return group_number

	def setup_iteration(self):
		data = self.original_data.copy()
		data["centroid"] = -1
		self.itteration_info = {}
		if self.starting_centroids is not None:
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
		while not centroids.equals(updated_centroids):
			data, centroids = updated_data, updated_centroids
			updated_data, updated_centroids = self.update_centroids(data, centroids)
		self.itteration_info["final_centroids"] = centroids.copy()
		self.itteration_info["final_data"] = data
		ev = self.calculate_ev(data)
		iv = self.calculate_iv(data, centroids)
		iv_ev = iv/ev
		self.itteration_info["IV"] = iv
		self.itteration_info["EV"] = ev
		self.itteration_info["IV/EV"] = iv_ev
		self.iterations_holder.append(self.itteration_info)
		if iv_ev > self.highest_iv_ev:
			self.highest_iv_ev = iv_ev
			self.best_iteration = self.itteration_info

	def calculate_ev(self, data):
		# bob = pd.DataFrame({'a':{0:"a", 1:"b", 2:"c"}, 'b':{0:"aa", 1:"bb", 2 :"c"},'c':{0:"z", 1:"z", 2:"z"}})
		df = data.copy()
		df["c"] = True
		df  = pd.merge(df.reset_index(),df.reset_index(), how="outer", on=["c"])
		def ordered_indexes(x,y):
			first = x if x > y else y
			second = x if first != x else y
			return str(first) + "_" + str(second)

		df["ordered_indexes"] = np.vectorize(ordered_indexes)(df['index_x'], df['index_y'])
		df.drop_duplicates(subset="ordered_indexes", inplace=True)
		## Then I need to ensure that they don't share the same group
		df = df.loc[df["centroid_x"]!=df["centroid_y"]]
		df.drop(labels=["centroid_x", "centroid_y", "c", 'index_x', "index_y", "ordered_indexes"], axis=1, inplace=True)
		df_one_columns = [c for c in df.columns if "_x" in c]
		df_two_columns = [c for c in df.columns if "_y" in c]
		columns = [c for c in self.original_data.columns]
		df_one = df[df_one_columns]
		df_one.rename(inplace=True, columns=dict(zip(df_one_columns,columns)))
		df_two = df[df_two_columns]
		df_two.rename(inplace=True, columns=dict(zip(df_two_columns,columns)))
		# (((centroids.sub(row, axis=1))**2).sum(axis=1)**.5).argmin()
		ev = ((((df_one - df_two)**2).sum(axis=1)**.5).sum(axis=0))/len(self.original_data)
		return ev

	
	def calculate_iv(self, data, centroids):
		distances = data.apply(lambda x: self.get_centroids_distance(x, centroids), axis=1)
		iv = distances.sum(axis=0)
		return iv

	def get_centroids_distance(self, row, centroids):
		distance = ((centroids.ix[row["centroid"]].sub(row))**2).sum()**.5
		return distance



original_data = pd.read_table(os.path.join(os.getcwd(), "original_data.txt"), sep="\t", header=None)
# ax = original_data.plot.scatter(x=0, y=0, color='Blue', label='Group 0');
# original_data.plot.scatter(x=1, y=1, color='Green', label='Group 1', ax=ax);
# original_data.plot.scatter(x=2, y=2, color='Red', label='Group 2', ax=ax);
# original_data.plot.scatter(x=3, y=3, color='Orange', label='Group 3', ax=ax);
# original_data.plot.scatter(x=4, y=4, color='Purple', label='Group 4', ax=ax);
# original_data.plot.scatter(x=5, y=5, color='Black', label='Group 5', ax=ax);
# original_data.plot.scatter(x=6, y=6, color='White', label='Group 6', ax=ax);
# # ipdb.set_trace()

k_means_alg = K_Means(original_data)
cluster_info = k_means_alg.get_cluster_info()
breakdown = cluster_info["final_data"]

centroid_set_dict = {0: {0: 19, 1: 11, 2: 15.5},
 1: {0: 16.5, 1: 13.5, 2: 15},
 2: {0: 0.88, 1: 0.82, 2: 0.91},
 3: {0: 5.6, 1: 5.3, 2: 6.5},
 4: {0: 3.8, 1: 2.8, 2: 3.2},
 5: {0: 2, 1: 3.5, 2: 5},
 6: {0: 6.0, 1: 4.7, 2: 5.3}}
 
k_means_2 = K_Means(original_data, starting_centroids=pd.DataFrame.from_dict(centroid_set_dict))
cluster_info_2 = k_means_2.get_cluster_info()
ipdb.set_trace()

# print ("hi")




























