"""
	This file contains two classes and some initial running code outside of that. 
	BayNode: A node meant to work as part of a baysian network. It keeps track of
			events that inolve itself, it keep track of these events in referance
			to its parents. Using this information to calculate the probability
			that an event occurs (a calculation function must be called). After
			calculation the probability of a conditional event can be retrieved
			rapidly. Additionally, the probability that an event occured without
			conditions can also be retrieved, but requires calculation. When
			creating a node you must pass it a name, a range of values, and all
			of its parents. 
	BayNetwork: Contains a list of nodes and somefunctions that call a function
				on the correct node to return a desired value, and functions that
				will call a function on each node. This class can read in a dataset
				where the first line is the name of each node, provided these nodes
				were created in a python function and passed to this class before
				If the dataset has missing data the network can still calculate
				probabilities by doign expectationMaximazation. EM will generally
				assign random distributions to events it does not know, and then
				use that distribution to calculate a model, and then use that
				to calculate a better distribution until the likelyhood of the
				previous model is very similar to the just calculated one.
"""

import copy
import random
import math
import sys

class BayNetwork(object):
	"""
		This class has the following methods:
		__init__
		printNetwork
		printConditionalProbNetwork
		recursive jointProb
		probValueGivenNetwork
		probDistrubtionGivenNetwork(self, node, subeventConditionallist):
		randomProbDistrubtionGivenNetwork(self, node, subeventConditionallist):
		convertFileIntoDataSet(self, fileName):
		readOriginalValues(self):
		networkResetProbabilities(self):
		emKickoff(self):
		expectation(self):
		maximization(self, dataSet):
		logLikelyhood(self):
		emIteration(self):
		expectationMaximazation(self,

	"""

	def __init__(self, baynodes = {}):
		self.originalNodes = copy.deepcopy(baynodes) # nodes are of the form {'Name': BayNode}
		self.nodes = baynodes  # nodes are of the form {'Name': BayNode}
		self.previousNodes = copy.deepcopy(baynodes)
		self.kickoffValues = []
		
	def printNetwork(self):
		for node in self.nodes.values():
			node.printNode()

	def printConditionalProbNetwork(self, fancyPrintDict= {}, tableListDict=None):
		for node in self.nodes.values():
			if tableListDict:
				node.printConditionalProbabilityTable(fancyPrintDict, tableListDict[node.name])
			else:
				node.printConditionalProbabilityTable(fancyPrintDict)

	# def jointProb(self, subeventConditionallist):
	# 	# subeventConditionallist is of the form [ (NodeName, value),(NodeName, value)]
	# 	"""
	# 	the probability that each node in the network is given the values specified in subeventConditionallist
	# 	is equal to the product of each p(node|parent) of each node in the network
	# 	getSubEventConditionalProbability return the value of p(node|[parent]) and takes care of any values 
	# 	that we don't know
	# 	"""

	# 	### I MAY NEED TO SUM OVER THE PARENTS/ EVERY OTHER NODE IN THE NETWORK HERE
	# 	total = 1.0
	# 	for nodeName, nodeValue in subeventConditionallist:
	# 		node = self.nodes[nodeName]
	# 		cp = node.getSubEventConditionalProbability(nodeValue, subeventConditionallist)
	# 		if cp != 0.0:
	# 			total *= cp
	# 		else:
	# 			print "NOT ENOUGH DATA FOR ACCURATE JOINT PROBABILITY ESTIMATE, ESTIMATING NULL EVENT AT 0.00000000000000000000001"
	# 			node.printConditionalProbabilityTable()
	# 			total *= 0.00000000000000000000001

	# 	return total

	def recursiveJointProb(self, jointNodeListc, subeventConditionallistc=[]):
		jointNodeList= copy.deepcopy(jointNodeListc)
		subeventConditionallist = copy.deepcopy(subeventConditionallistc)
		specifiedParentList = set()
		nodeNamesInSubCondList = [n[0] for n in subeventConditionallist]
		for nodeName, nodeValue in jointNodeList:
			node = self.nodes[nodeName]
			for parentName in node.parents.keys(): # self.parents[name] = node
				specified = False
				for nodeNameValuePair in jointNodeList:
					if parentName in nodeNameValuePair: # if parent is specified in jointNodeList
						specifiedParentList.add(copy.deepcopy(nodeNameValuePair))
						specified = True
						break
				if specified == False: # if it isn't specified by the jointNodeList, it still may be in the subeventConditionallist
					for nodeNameValuePair in subeventConditionallist:
						if parentName in nodeNameValuePair: # if parent is specified in subEventConditionalList
							specifiedParentList.add(copy.deepcopy(nodeNameValuePair))
							specified = True
							break
				if specified == False: # if it still is unspecified, it really is unspecified, so add it to that list
					# Since some parent value isn't specified, I need to return the sum of all potential values of that parent. 
					# This means I must recursively call this function, but the new call will be make sure that the parent is specified
					# To specify it I need to add this value to the subeventConditionallist, as I don't want to add this to the joint calculation,
					# only to the list that lets me know what value it is when calculating it's join or the child given this parent
					total = 0.0
					parentNode = self.nodes[parentName]
					for pvalue in parentNode.val_range: # figure out the probability of what we want assuming the parent is the pvalue and add this to the total, as this is one way to satisify our original joint probility question and by adding all the probabilities of all the ways it can occur we have the probability it can occur
						newsubeventConditionallist = copy.deepcopy(subeventConditionallist)
						newsubeventConditionallist.append((parentName,pvalue))
						total += self.recursiveJointProb(jointNodeList, newsubeventConditionallist)
					return total
		# I am sure now that no parent is left unspecified. There is nothing to sum over. 
		# I now need to make sure that everything specified in the jointNodeList is specified the same way in the subeventConditionalList
		for nodeName, nodeValue in jointNodeList:
			if nodeName not in nodeNamesInSubCondList:
				subeventConditionallist.append((nodeName, nodeValue))
		# Now everything I know is specified in the subeventConditionallist and everything I know about the parents also is specified in the specifiedParentList 
		# All there is left to do is multiple each p(child|parents) and multiple that by the joint probability of the parents
		total = 1.0 
		for nodeName, nodeValue in jointNodeList:
			if (nodeName, nodeValue) not in specifiedParentList: # if 
				node = self.nodes[nodeName]
				# print "CALCULATING SUBEVENT PROB of"
				cp = node.getSubEventConditionalProbability(nodeValue, subeventConditionallist) # this should deep copy with itself if needed, but it really shouldn't be needed as everything should be specified
				if cp != 0.0:
					total *= cp
				else:
					print "NOT ENOUGH DATA FOR ACCURATE RECURSIVE JOINT PROBABILITY ESTIMATE, ESTIMATING NULL EVENT AT 0.00000000000000000000001"
					node.printConditionalProbabilityTable()
					total *= 0.00000000000000000000001
		# multiple the total by the join probability of the parents
		# if there is only one parent then you can just get the subeventConditionalProbability of that parent
		if len(specifiedParentList) <= 1:
			for nodeName, nodeValue in specifiedParentList:
				cp = self.nodes[nodeName].getSubEventConditionalProbability(nodeValue, subeventConditionallist)
				total *= cp
		else:
			total *= self.recursiveJointProb(specifiedParentList, subeventConditionallistc)
		return total

		

	def probValueGivenNetwork(self, nodeName, value, subeventConditionallist):
		"""
		the event lacks a value for A, so
		add the value of A to a copy of the event
		return jointProb(copy) / jointProb(event)
		"""
		currentNodeValuetuple = (nodeName, value)
		scl = copy.deepcopy(subeventConditionallist)
		scl.append(currentNodeValuetuple)
		jp = self.recursiveJointProb(scl)
		cl = copy.deepcopy(subeventConditionallist)
		jpscl = self.recursiveJointProb(cl)
		return jp/jpscl

	def probDistrubtionGivenNetwork(self, node, subeventConditionallist):
		# returns a tuple of the form ( (value, %), (value, %))
		"""
		tupleList = []
		for each value in range
			t = (value, probValueGivenNetwork(value,event))
			tupleList.append(t)
		return tuple(tupleList)

		"""
		nodeName = node.name
		tupleList = []
		for value in node.val_range:
			t = (value, self.probValueGivenNetwork(nodeName,value,subeventConditionallist))
			tupleList.append(t)
		return tuple(tupleList)

	def randomProbDistrubtionGivenNetwork(self, node, subeventConditionallist):
		""" This returns a probibility distribution that isn't cause by data"""
		nodeName = node.name
		tupleList = []
		randomPercentageList =[]
		randomNumberList =[]
		total = 0.0
		"""
			Giving it the right starting points needs to be added
		"""
		for value in node.val_range:
			rand = random.uniform(1, 10000)
			randomNumberList.append(rand)
			total += rand
		for pvalue in randomNumberList:
			percentage = pvalue/total
			pvalue = percentage
			randomPercentageList.append(pvalue)
		random.shuffle(randomPercentageList)
		for i, value in enumerate(node.val_range):
			tupleList.append( (value, randomPercentageList[i]) )
		return tuple(tupleList)
	
	def convertFileIntoDataSet(self, fileName):
		self.nodePositions =[] # holds the actual node coorisponding to the index of a position
		self.nodePositionsDict ={}
		infile = open(fileName)
		with infile as inf:
			datapoints = []
			dirtyDateSet = [line.strip().split('\t') for line in inf]
			for line in dirtyDateSet:
				datapoints.append([point.strip() for point in line])
			counter = 0
			self.headerLine = datapoints[0]
			for i, nodeName in enumerate(self.headerLine):
			 	if nodeName in self.nodes:
			 		self.nodePositions.append(self.nodes[nodeName])
			 		self.nodePositionsDict[nodeName]= i  # Save the index of the node for the name
			 		counter+=1
			 	else:
			 		print ("NETWORK ISNT CONFIGURED CORRECTLY FOR THIS FILE: ABSENT NODE in network")
			if len(self.nodes) != counter:
				print ("NETWORK ISNT CONFIGURED CORRECTLY FOR THIS FILE: File is missing a node ")
				return -1
			self.originalDataSet = datapoints[1:] # Save the actual data for expectation maximization

	def readOriginalValues(self):
		"""
			This function is not required or used for the project. 
			It serves to read in a full dataset
		"""
		for dataLine in self.originalDataSet:  
			for i, datapoint in enumerate(dataLine):  
				parent_event_list = []  # This will need to turn it into a tuple with tuple(parent_event_list) so it can be a dictionary key
				for parentName in self.nodePositions[i].getParents():  # go through the parents of the BayNodes at our datapoint
					#  Data for a node must be in the form of is a tuple of ((("parent_name", parent_value),("parent_name", parent_value)), self_value)
					#  Now I get the index of the node in the list and get the data at that point
					parental_event= dataline[self.nodePositionsDict[parentName]]  # the event of the parent
					parent_event_list.append( (parentName, parental_event) ) #  Store the parental_event with the name of the parent and place it in the tuple list
				conditional_event = ( tuple(parent_event_list), datapoint)  # A tuple of parent events combined with the child event
				self.nodePositions[i].addData(conditional_event)  # Have the node coorisponding to the current column keep track of the event
		self.networkResetProbabilities()
	
	def networkResetProbabilities(self):
		"""
			This function make sure probabilities are correct by looking at the events the network has captured
		"""
		for node in self.nodes.values():
			node.fullResetProbabilities()

	def emKickoff(self):
		"""
			## convertFileIntoDataSet(filename) must be called before this function works

			This function reads through self.originalDataSet which is a list that contains events (represented as lists)
			It then parses through that data to add the data correctly in the form of conditional events to all the nodes in network 
			Each event is assumed to be complete and not missing any data
	
			This function returns a list of events (represented as lists) with the probability that each event occured
			in the form of [ ( ( ('name', value),('name', value) ) , %)]
		"""
		parentDataList= []
		newDataSet = []
		# This function looks at each line of data {aka an event} in the original originalDataSet
		for event in self.originalDataSet:
			# Make a dictionary that maps nodeName -> realNodeData it recievied in this event
			for i, realNodeData in enumerate(event): 
			  nodeName = self.nodePositions[i].name
			  if realNodeData != '-': # or if it is not in the set of values that mean the data is missing
			  	parentDataList.append((nodeName, realNodeData)) # [nodeName] = realNodeData
			  # otherwise nothing is know about the data so don't place it into the disctionary, since that will mess up function that try to look
			  # for probabities of various events as that event will be the event that the node = '-' as opposed to summing over all its values
			eventList =[] # this holds lists that represent different events which occur with different probabilities they occured
			# it will hold one event provided that all the data is there, but it will hold multiple events if the data is missing and it uses a probability
			# distribution to fill in the data
			""" 
				If there is no missing data the event occured with a probability of one, 
				If there is missing data, I break the events down into multiple events each having a < 1 probability of occuring, that sums to one. 
				The way it is stored is that each eventList contains tuples of 
			"""
			# Go through each piece of data and create a new dataset that includes the probability of each event and doesn't have any '-' spaces or missing data
			for i, realNodeData in enumerate(event):
				node = self.nodePositions[i]
				nodeName = node.name
				if realNodeData == '-':
					temp_event_lists =[]
					distribution = self.randomProbDistrubtionGivenNetwork(node, parentDataList) # returns a tuple of the form ( (value, %), (value, %))
					for (distrubtedValue, percentage) in distribution: # MAY NEED TO DO ENUMERATE or something
						if not eventList: # if eventList is empty the next chunk of code won't run so
							completeEvent = []
							completeEvent.append( ( (nodeName, distrubtedValue), percentage) )
							temp_event_lists.append(completeEvent)
						else:
							for incompleteEvent in eventList:
								completeEvent = copy.deepcopy(incompleteEvent)
								completeEvent.append( ( (nodeName, distrubtedValue), percentage) )
								temp_event_lists.append(completeEvent)
					eventList = temp_event_lists
				else:
					# just append a tuple of (realNodeData, 1) to each value in the eventList
					if not eventList:
						eventList.append([( (nodeName, realNodeData), 1)]  )
					else:
						for incompleteEvent in eventList:
							incompleteEvent.append( ( (nodeName, realNodeData), 1) )
			 # [ ( ( ('name', value),('name', value) ) , %)]
			for event in eventList:
				tl = []
				totalp = 1.0
				for (partial_event, percentage) in event:
					totalp *= percentage
					tl.append(partial_event)
				newDataSet.append( (tl, totalp) ) 
		return newDataSet


	def expectation(self):
		"""
			This function returns a list of events (represented as lists) with the probability that each event occured
			in the form of [ ( ( ('name', value),('name', value) ) , %)]
		"""
		newDataSet =[]
		# This function looks at each line of data {aka an event} in the original originalDataSet
		# parentDate includes info about children as well
		for i, event in enumerate(self.originalDataSet):
			parentDataList= []
			# Make a dictionary that maps nodeName -> realNodeData it recievied in this event
			for i, realNodeData in enumerate(event): 
			  node = self.nodePositions[i]
			  nodeName = node.name
			  if realNodeData != '-': # or if it is not in the set of values that mean the data is missing
			  	parentDataList.append((nodeName, realNodeData)) # [nodeName] = realNodeData
			  # otherwise nothing is know about the data so don't place it into the disctionary, since that will mess up function that try to look
			  # for probabities of various events as that event will be the event that the node = '-' as opposed to summing over all its values
			eventList =[] # this holds lists that represent different events which occur with different probabilities they occured
			""" 
				If there is no missing data the event occured with a probability of one, 
				If there is missing data, I break the events down into multiple events each having a < 1 probability of occuring, that sums to one. 
				The way it is stored is that each eventList contains tuples of 
			"""
			# Go through each piece of data and create a new dataset that includes the probability of each event and doesn't have any '-' spaces or missing data
			for i, realNodeData in enumerate(event):
				node = self.nodePositions[i]
				nodeName = node.name
				if realNodeData == '-':
					# DEAL WITH COPYING Eventlist and appending to each for each value in the distribution
					# probDistrubtionGivenNetwork(self, node, subeventConditionallist)
					temp_event_lists =[]
					distribution = self.probDistrubtionGivenNetwork(node, parentDataList) # returns a tuple of the form ( (value, %), (value, %))
					for (distrubtedValue, percentage) in distribution: # MAY NEED TO DO ENUMERATE or something
						if not eventList: # if eventList is empty the next chunk of code won't run so
								completeEvent = []
								completeEvent.append( ( (nodeName, distrubtedValue), percentage) )
								temp_event_lists.append(completeEvent)
						else:
							for incompleteEvent in eventList:
								completeEvent = copy.deepcopy(incompleteEvent)
								completeEvent.append( ( (nodeName, distrubtedValue), percentage) )
								temp_event_lists.append(completeEvent)
					eventList = temp_event_lists
				else:
					# just append a tuple of (realNodeData, 1) to each value in the eventList
					if not eventList:
						eventList.append([( (nodeName, realNodeData), 1)]  )
					else:
						for incompleteEvent in eventList:
							incompleteEvent.append( ( (nodeName, realNodeData), 1) )
			 # [ ( ( ('name', value),('name', value) ) , %)]
			eventsWithProb = []
			for event in eventList:
				tl = []
				totalp = 1.0
				for (partial_event, percentage) in event:
					totalp *= percentage
					tl.append(partial_event)
				newDataSet.append((tl, totalp))
		return newDataSet
					

	def maximization(self, dataSet):
		""" 
			This function reads through a list that contains events (represented as list) with the probability that they occured
			It then parses through that data to add the data correctly in the form of conditional events to all the nodes in network 
			Each event is assumed to be complete and not missing any data
		"""

		for event in dataSet:
			for i, node in enumerate(self.nodePositions): # go through every node in the header, which should be every node in the dataset/network
				modifiedEvent = copy.deepcopy(event[0])
				valueOfCurrentNode = modifiedEvent.pop(i)[1]
				### add the data to the relivant node in the form of a proper event, with the percetange that the event occured
				e = node.createEvent(modifiedEvent, valueOfCurrentNode) 
				node.addData(e, event[1])


	def logLikelyhood(self):
		totalLikelyhood = 0.0
		multtotalLikelyhood= 1.0
		for event in self.originalDataSet:
			"""
				clean line, 
				joint prob of clean line
				log of joint prob
				add result to total
			"""			
			eventlist = []  # eventList is of the form [ (NodeName, value),(NodeName, value)]
			for i, realNodeData in enumerate(event):
				if realNodeData != '-':
					node = self.nodePositions[i]
					nodeName = node.name
					eventlist.append( (nodeName, realNodeData) )
			# cacluating joint prob for event:"
			jp = self.recursiveJointProb(eventlist) # jp = self.jointProb(eventlist)
			if jp != 0.0:
				multtotalLikelyhood =  multtotalLikelyhood * jp
				totalLikelyhood+= math.log(jp)
		return totalLikelyhood

	def emIteration(self):
		# This returns the log likelyhood of the model generated
		# self.printNetwork()
		expectation  = self.expectation()
		self.previousNodes = copy.deepcopy(self.nodes)
		self.nodes = copy.deepcopy(self.originalNodes)
		self.nodePositions =[] # holds the actual node coorisponding to the index of a position
		for i, nodeName in enumerate(self.headerLine): # the first line of the file
		 	self.nodePositions.append(self.nodes[nodeName])
		#  "calculating model"
		self.maximization(expectation)
		self.networkResetProbabilities()
		return self.logLikelyhood()
			  
		
	def expectationMaximazation(self, kickoff = 1, fancyPrintDict ={}, printListDict = None):
		# Set a convergance threashold
		converganceThreashold = 0.001
		if(kickoff >=1 ):
			print "kicking off"
			expectation  = self.emKickoff()
			print "node copying"
			self.nodes = copy.deepcopy(self.originalNodes)
			self.nodePositions =[] # holds the actual node coorisponding to the index of a position
			for i, nodeName in enumerate(self.headerLine): # the first line of the file
			 	self.nodePositions.append(self.nodes[nodeName])
			print "maximization"	 	
			self.maximization(expectation)
			print "reseting network prob"
			self.networkResetProbabilities()
		else:
			print "ASSUMING PROBABILITIES HAVE BEEN GIVEN TO NETWORK"
		print "STARTING CONDTIONAL PROBABILTIES VALUES:"
		self.printConditionalProbNetwork(fancyPrintDict, printListDict)
		print " getting likelyhood"
		previousLikelyhood = self.logLikelyhood()

		print " running itteration"
		currentLikelyhood = self.emIteration()
		dif = currentLikelyhood-previousLikelyhood
		iterationLikehoodTable = "\nTable Mapping iterations to likelyhood\niteration \t likelyhood"
		iterationCounter = 1
		print "checking likelyhood comparison"
		while(dif > converganceThreashold ):
			iterationLikehoodTable +=  "\n%d\t%f" %(iterationCounter,previousLikelyhood)
			iterationCounter +=1 
			print ("Running another itteration as currentLikelyhood-previousLikelyhood = ", dif)
			previousLikelyhood = copy.deepcopy(currentLikelyhood) 
			currentLikelyhood = self.emIteration()
			dif = currentLikelyhood-previousLikelyhood
		print "Not running another itteration as currentLikelyhood-previousLikelyhood = %f" %(dif)
		# revert back to the previous itteration as that may be better then the current one
		# and it is know that they are not significantly different
		self.nodes = copy.deepcopy(self.previousNodes )
		iterationLikehoodTable +=  "\n%d\t%f" %(iterationCounter,previousLikelyhood)
		iterationLikehoodTable += "\n"
		return previousLikelyhood


				

class BayNode(object):
	"""
		This function serves to represent a node of a baysian network
		Methods:
			Setup Methods:
				1. __init__(self, name, val_range = [], parents = []):
					- constructor that takes in option arguments for the range of values and parents
				2. addParent(self, node, name):
					- This function should be used to add parents to the node. If parents are added
					previous events and probabilities tracked will not be handled correctly 
				3. fullResetProbabilities(self):
					- This updates all the probabilities within the node to reflect the new events it 
					  has seen. It must be called after new data is added in order to return correct probabilities

			Print Methods:
				1. printNode(self):
					- Prints the nodeName and the way in which the events and thier probabilities are stored
				2. printConditionalProbabilityTable(self, fancyPrintDict ={}):
					- Prints a friendly version of conditional probabilities stored

			Helper Methods:
				1. createEvent(self, parentnameEventTuple_holderMain, currentNodeValue):

			Get Methods:
				1. getParents(self):
					- Not used in project. 
				2. getConditionalProb(self, event):
					- Returns the probability of an event, the event is conditional upon parents
					- Events must specify every parent value
				3. getSubEventConditionalProbability(self, thisNodeValue, subeventConditionallist):
					- Recursive function that returns the probability of a potential conditional event
					- Events do not need to specify every parent value
				4. getProbViaConditional(self, event):
					- Returns the probability that a non conditional event occured by mariginalizing 
					  relivant conditional probabilities 
	"""

	# parents are in the form {'Name': BayNode}
	def __init__(self, name, val_range = [], parents = []):
		self.name = name
		self.val_range = val_range
		self.parents = {}
		self.parentList =[]
		if parents:
			for parentNode in parents:
				self.addParent(parentNode, parentNode.name)
		self.value_counts = {} # is a dictionary mapping a tuple of ((("parent_name", parent_value),("parent_name", parent_value)), self_value) to the number of times it occurs
		self.event_probibility = {} # similar to the above, but the value is a probability not a count
		self.intelligentGuess = False
		self.guessBasedOnAllData = True
		self.parentList = list(self.parents.keys()) # this list is to ensure the order of an event is also referanced correctly in the right tuple order

	def printNode(self):
		print "\nNode name: " + self.name
		for (prob, event) in enumerate(self.event_probibility):
			print "event: " 
			print event 
			print "prob : " 
			print self.event_probibility[event]

	def addParent(self, node, name):
		self.parents[name] = node
		self.parentList.append(name)
	
	def getParents(self):
			return self.parents

	def addData(self, data, percentage= 1.0):
		# data is a tuple of ((("parent_name", parent_value),("parent_name", parent_value)), self_value)
		self.value_counts[data] = self.value_counts.get(data, 0.0) + percentage

	# event is of the form ((("parent_name", parent_value),("parent_name", parent_value)), self_value)
	def getConditionalProb(self, event):
		return self.event_probibility[event]

	# for node in network reset the probabilities
	def fullResetProbabilities(self):
		self.event_probibility = {}
		masterEventList =[]
		# Generate a list for all possible parental configurations
		if self.parentList:
			for parentName in self.parentList:
				parentNode = self.parents[parentName]
				if not masterEventList:
					for pvalue in parentNode.val_range:
						masterEventList.append([(parentName, pvalue)])
				else:
					newMasterList = []
					for pvalue in parent:
						copyList= [copy.deepcopy(sublist) for sublist in  masterEventList]
						for eventList in copyList:
							eventlist.append( (parentName,pvalue) )
						newMasterList.append(copyList)
					masterEventList = newMasterList
		else:
			masterEventList =[()]
		# For each value of this node | that configuration set the probability of that full event
		for eventlist in masterEventList:
			total = 0.0
			fullEventsList = []
			for cvalue in self.val_range:
				fullEvent = self.createEvent(eventlist, cvalue) 
				try:
					total += self.value_counts[fullEvent]
				except Exception:
					total += 0.0
				fullEventsList.append(fullEvent)
			for fullEvent in fullEventsList:
				try:
					self.event_probibility[fullEvent] = self.value_counts[fullEvent]/total
				except Exception as inst:
					self.event_probibility[fullEvent] = 0.0

	
	def printConditionalProbabilityTable(self, fancyPrintDict ={}, tablePrintList = None):
		# fancyPrintDict is in the form {(NodeName,Nodevalue): RealValue }
		# tablePrintList is a list of what cols,rows shoudld print in what order where
		# [ selfValueList, ParentList1, ParentList2] # this may have a too many to unpack problem with more than 1 parent
		t= ""
		header= ""
		if tablePrintList:
			rows = tablePrintList[0]
			cols = tablePrintList[1:]
			r = ""
			for row in rows:
				r = row+":\t"
				if header == "":
					header += self.name +"\t"
					for column in cols:
						e = (column,row)
						r +=  "%f\t"%(self.event_probibility[e])
						for pName, pvalue in column:
							header+= pName + "=" + pvalue + "\t"
					header += "\n"
				else:
					for column in cols:
						e = (column,row)
						r +=  "%f\t"%(self.event_probibility[e])
				r += "\n"
				t += r
		tabl= header+t
		s= ""
		for event in self.event_probibility:
			e="\nP("
			condition=""
			for (parentName, parentValue) in event[0]:
				condition+= parentName + "="
				if (parentName, parentValue) in fancyPrintDict.keys():
					condition += fancyPrintDict[(parentName, parentValue)]
				else:
					condition += parentValue
			e+= self.name + "="
			if (self.name, event[1]) in fancyPrintDict.keys():
				e+= fancyPrintDict[(self.name, event[1])]
			else:
				e+= event[1]
			if condition != "":
				e+= " | "
			e += condition + " )" + ("=%f" %self.event_probibility[event]) 
			s+= e
		print s
		print "\nTableView:"
		print tabl


	def getSubEventConditionalProbability(self, thisNodeValue, subeventConditionallist):  
		"""
		This function only can give you the probability of a node given its parents and will ignore the value of its children
		
			The next section recursively sums ( p(a| knownvalues, someparent) p(someparent)) such that 
			the someparent is irrelivant in knowing the absolute p(a | knownvalues)

			The method to doing so iteratively, would be to generate a list of relivant dictonary lookup values 
				for the parental event which is given. An event is represented by a list of p(parents= their values).
				if we know every parent's value, we can return self.getConditionalProb(event) 

				if we do not know a parents value,
					for each possible value
						we must call self.getConditionalProb(event), assuming that value is x
						multiply the result by the probability that the parent would have a value x
						and add the result of the multiplication to a counter that represents the total probability of the desired event

				# Doing so recursively is easy to write because we don't need to know what other values we do or do not know, since we can 
				# get the probability of the desired event by just recalling this function, 
				# doing so itteratively requires you to have all the dictionary lookups before you start to look them up

				To get all the lookups before looking them up we can keep a list of dictionary lookups
				each dictionary lookup is a sublist of its own, containing tuples of the parentValue, p(parent=value)
				if we are given the value to start then that probability is 1. 

				for each parent node check if we know its values, 
					if we do then for each possible sublist in our master list append (parent, parentValue, 1) to that list
					# this is equivellent to having the dictonary lookup specified to the cases where we know the parentValue
					if we don't know the value, 
						copy our master list
						for each value the parent can be, 
							get the probability the parent is that value, 
							then for each lookup in our original list futher specify that lookup by appending to 
							the sublist that represents that lookup a tuple of (parent, potential value, the probability the parent is that value)
							then make our original list = to our master list, # this way the next time we don't know a lookup we can branch from all our current loopup branches
				for each sublist 
					create a dictionary lookup by using the first two values of every tuple in our sublist
					store the value of the lookup via self.getConditionalProb(event)
					find the probability that the event occured by calculating the product of the 3rd value in each tuple of a sublist
					multiple the probability we got from the lookup by the probability the event occured, 
					add that result to a counter that starts at 0.0  to represent the probability of any one of those dictionary lookups occuring, 
					since if any one of them occurs our original event would have occured,  
				return the value of the counter as the probability this subevent occured

			"""
		# subeventConditionallist will turn into a tuple within createEvent, it is of the form [ (NodeName, value),(NodeName, value)]
		specifiedNodeNames = [t[0] for t in subeventConditionallist ]
		unspecifiedparent = None
		for parentName in self.parentList:
			if parentName not in specifiedNodeNames:
				unspecifiedparent = parentName
				break;
		# unspecifiedlist = [parentName for parentName in self.parentList if parentName not in [l[0] for l in subeventConditionallist ]]  # a list of parent nodes that we must sum over
		if not unspecifiedparent: # no parent is unspecified, everything is specified and we can get the probability of the value given the event
			event = self.createEvent(subeventConditionallist, thisNodeValue)
			return self.getConditionalProb(event)
		else: # there is a hidden variable that we must sum over
			totalProb = 0.0
			parentName = unspecifiedparent
			""" 
				Assuming there is only one unspecified parentValue this function will sum ( p(event|parentValue = potential value) * p(value) )
				if there is still another unspecified value, the call to self.getSubEventConditionalProbability will realized that and handle it occurdingly
			"""
			sumProbability= 0.0
			for parentValue in self.parents[parentName].val_range:
				newConditionalList = copy.deepcopy(subeventConditionallist)
				newConditionalList.append((parentName,parentValue )) # not sure if it should be in the form of a list or a tuple
				scp = self.getSubEventConditionalProbability(thisNodeValue, newConditionalList)
				probabilityOfParent = self.parents[parentName].getProbViaConditional(parentValue) # this is the objective probability of the parents 
				wieghtedSumProb= (scp*probabilityOfParent)
				sumProbability += wieghtedSumProb
			totalProb += sumProbability
			return totalProb


	def createEvent(self, parentnameEventTuple_holderMain, currentNodeValue):
		"""
		parentnameEventTuple_holder must contain values for every referance in self.parentList, 
		but self.parentList does not need to contain a referance for every value contained in parentnameEventTuple_holder 
		this enable passing the entire state of the network to this function and this function will only search for the conditional 
		event of this node on the parents, yet still pass the parents the information they require regarding their parents
		"""
		parentnameEventTuple_holder = copy.deepcopy(parentnameEventTuple_holderMain)
		tempTuple_holder = []
		for parentName in self.parentList:
			popped = False 
			for i, parentnameEventTuple in enumerate(parentnameEventTuple_holder):
				if parentName == parentnameEventTuple[0]: # if the parent names match
					tempTuple_holder.append(tuple((parentnameEventTuple_holder[i][0], parentnameEventTuple_holder[i][1]) ))
					# remove the item from the pass holder, and place into the holder 
					popped = True
					break;
			if popped == False:
				print("ERROR WHILE CREATING EVENT, Missing tuple for " + parentName)
		return ( tuple(tempTuple_holder), currentNodeValue) # tuple(tuple(tempTuple_holder), currentNodeValue)
	

	def getProbViaConditional(self, event):
		totalProb = 0.0 # self.value counts doesn't always have all the possible events, make sure it does
		for e in (relivant_events for relivant_events in self.event_probibility if relivant_events[1] == event):
			condprob = self.getConditionalProb(e)
			for (parent_name, parent_value) in e[0]:
				condprob *= self.parents[parent_name].getProbViaConditional(parent_value)
			totalProb += condprob
		return totalProb

###### End of BayNode Class	

def multiEM(file, outFile, kickoff = 1, runnTimes = 10):
	"""
		This function creates the Baysian network specified by the project guildlines. 
		It reads in a file and will perform the EM ALGORITHM multiple times as specified
		by runnTimes. It will only save the best resulting network and its likelyhood. 
		Additionally this function has the ability to use the conditional probabilities
		specified within the project guildines as a starting point or to use randomly 
		generated values, this is decided by the value of kickoff. It will print the 
		conditional probability for the best version.

	"""
	
	sys.stdout = open(outFile, "w")
	infile = file
	print outFile
	
	gender1 = BayNode('Gender', ('0', '1'))
	wieght1 = BayNode('Weight', ('0', '1'), [gender1])
	hieght1  = BayNode('Height', ('0', '1'), [gender1])

	if kickoff < 1:	
		runnTimes = 1
		gprobDict = {((), '0'): 0.7, ((), '1'):0.3}
		# P(gender=M)=0.7; 
		wprobDict = {((('Gender', '0'),), '1'): .2, ((('Gender', '1'),), '1'): .6, 
					((('Gender', '1'),), '0'): .4, ((('Gender', '0'),), '0'):.8 }
		# P(weight=greater_than_130|gender=M)=0.8;
		# P(weight=greater_than_130|gender=F)=0.4; 
		hprobDict = {((('Gender', '0'),), '1'): .3, ((('Gender', '1'),), '1'): .7,
		 			((('Gender', '1'),), '0'): .3, ((('Gender', '0'),), '0'):.7 }
		# P(height= greater_than_55|gender=M)=0.7; 
		# P(height= greater_than_55|gender=F)=0.3;
		gender1.event_probibility = gprobDict
		wieght1.event_probibility = wprobDict
		hieght1.event_probibility = hprobDict 

	fancyPrintDict= {('Gender', '0'):'M',('Gender', '1'):'F',
				('Weight', '0'):'greater_than_130',('Weight', '1'):'less_than_130',
				('Height', '0'):'greater_than_55',('Height', '1'):'less_than_55',}

	tableListDict= { "Gender":[('0', '1'), (tuple()), ],
					"Height": [('0', '1'), (('Gender','0'),),(('Gender','1'),)],
					"Weight" :[('0', '1'), (('Gender','0'),),(('Gender','1'),)]  }
				
	net1 = BayNetwork(baynodes = {gender1.name :gender1, wieght1.name: wieght1, hieght1.name: hieght1})
	
	net1.convertFileIntoDataSet(infile)

	a = net1.expectationMaximazation(kickoff, fancyPrintDict, tableListDict)
	a = copy.deepcopy(a)
	
	bestNet = net1

	for i in range(runnTimes):
		print "\n \t ", i, "= nth time RUNNING EM ALGORITHM \n"
		gender2 = BayNode('Gender', ('0', '1'))
		wieght2 = BayNode('Weight', ('0', '1'), [gender2])
		hieght2  = BayNode('Height', ('0', '1'), [gender2])
		if kickoff < 1:
			gender2.event_probibility = gprobDict
			wieght2.event_probibility = wprobDict
			hieght2.event_probibility = hprobDict 

		net2 = BayNetwork(baynodes = {gender2.name :gender2, wieght2.name: wieght2, hieght2.name: hieght2})
		net2.convertFileIntoDataSet(infile)
		b = net2.expectationMaximazation(kickoff, fancyPrintDict, tableListDict)
		b = copy.deepcopy(b)

		if b > a:
			bestNet = net2
			a = b
			print " Found a better answer"
		else:
			print "Not a better answer"
		print "Best Answer has a likelyhood of"
		print a

	print "\n\n Printing conditionalProbability for best iteration of EM:"
	bestNet.printConditionalProbNetwork(fancyPrintDict, tableListDict)
	print bestNet.logLikelyhood()

"""
	This code here runs the multiEM function on multiple files and uses thier names to generate output files. 
"""
files=["hw2dataset_10.txt", "hw2dataset_30.txt","hw2dataset_50.txt","hw2dataset_70.txt","hw2dataset_100.txt"]
# files = ["testCase.txt"]
for fileName in files:
	outName = "OUTPUTFILE_statingValues_" + fileName	
	multiEM(fileName, outName, 0)  # without kickoff
	outName = "OUTPUTFILE_randomStart_" + fileName
	multiEM(fileName, outName, 1)  # with kickoff

