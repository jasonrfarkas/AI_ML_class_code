## Machine learning Project
import pandas as pd

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
		self.original_data = data
		self.current_iteration = 0
		self.number_of_centroids = number_of_centroids if not starting_centroids else: len(starting_centroids)
		self.starting_centroids = starting_centroids
		self.iterations_holder=[]
		self.k_means_iteration()

	def update_centroids(data, centroids):
		new_centroids = data.groupby(["centroid"]).mean()
		data["centroid"] = self.get_distance(data, centroids)
		return data, new_centroids



	def k_means_iteration(self):
		data = self.original_data.copy()
		data["centroid"] = -1
		itteration_info = {}
		if self.starting_centroids:
			centroids = self.starting_centroids
		else:
			mins = ## GET MIN OF EACH COLUMN
			maxs = ## GET MIN OF EACH COLUMN
			self.number_of_centroids 
			centroids = pd.dataframe()# RANDOMLY GENERATE values between the min and max of each column, each time for the number of k
		itteration_info["starting_info"] = 
		updated_data, updated_centroids = self.update_centroids(data, centroids)
		while not (data.equals(updated_data):
			data, centroids = updated_data, updated_centroids
		self.iterations_holder.append(itteration_info)
		#starting centroids, final centroids, IV, EV, and IV/EV 







































