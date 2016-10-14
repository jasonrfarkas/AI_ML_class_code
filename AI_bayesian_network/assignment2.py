import copy
import random
import math
import sys

#class RootBayNode(object):
class BayNetwork(object):

	def __init__(self, baynodes = {}): # , previousBaynodes = {}):
		self.originalNodes = copy.deepcopy(baynodes) # nodes are of the form {'Name': BayNode}
		self.nodes = baynodes  # nodes are of the form {'Name': BayNode}
		# if not previousBaynodes:
		# 	previousBaynodes = baynodes
		self.previousNodes = copy.deepcopy(baynodes)
		self.kickoffValues = []
		
	def printNetwork(self):
		for node in self.nodes.values():
			node.printNode()

	def printConditionalProbNetwork(self, fancyPrintDict= {}):
		for node in self.nodes.values():
			node.printConditionalProbabilityTable(fancyPrintDict)

	def jointProb(self, subeventConditionallist):
		# subeventConditionallist is of the form [ (NodeName, value),(NodeName, value)]
		"""
		the probability that each node in the network is given the values specified in subeventConditionallist
		is equal to the product of each p(node|parent) of each node in the network
		getSubEventConditionalProbability return the value of p(node|[parent]) and takes care of any values 
		that we don't know
		"""
		total = 1.0
		for nodeName, nodeValue in subeventConditionallist:
			node = self.nodes[nodeName]
			cp = node.getSubEventConditionalProbability(nodeValue, subeventConditionallist)
			if cp != 0.0:
				total *= cp
			else:
				print "NOT ENOUGH DATA FOR ACCURATE JOINT PROBABILITY ESTIMATE, ESTIMATING NULL EVENT AT 0.00000000000000000000001"
				node.printConditionalProbabilityTable()
				# print "node.name %s is %s" %(node.name, nodeValue)
				# print subeventConditionallist
				total *= 0.00000000000000000000001

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
		jp = self.jointProb(scl)
		jpscl = self.jointProb(subeventConditionallist)
		# print ("jp = %f and jpscl= %f " %(jp,jpscl ))
		# print ("the event is " + nodeName + " = %s | " %value)
		# print subeventConditionallist
		return jp/jpscl
		#jp = self.jointProb()

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
		nodekickoffValues = [cptuple for cptuple in self.kickoffValues if cptuple[0][0] == node.name]
		# kickoffValues is of the form  (  (node = value | otherNodes = otherValue), probability) 
		# (  (nodeName, value) , ( (othernodeName, value) ,(othernodeName, value)  ) , probability) 
		if nodekickoffValues: # if there are no kickoff values for this node
			blank = 0
		# 	for value in node.val_range:
		# 		t = (value, probValueGivenNetwork(nodeName,value,subeventConditionallist))
		# 		tupleList.append(t)
		else:
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

	# maybe change it to accept a list of potential values, and a list of probabilities 
	def addToKickoffValues(self, node, value, otherNodeValueTuples, probability ):
		self.kickoffValues.append( (node, value) ,otherNodeValueTuples, probability )	
			

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
				# print nodeName
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
		for dataLine in self.originalDataSet:  
			for i, datapoint in enumerate(dataLine):  
				parent_event_list = []  # This will need to turn it into a tuple with tuple(parent_event_list) so it can be a dictionary key
				for parent in self.nodePositions[i].getParents():  # go through the parents of the BayNodes at our datapoint
					#  Parent is the name of the parent, not the node itself
					#  Data for a node must be in the form of is a tuple of ((("parent_name", parent_value),("parent_name", parent_value)), self_value)
					#  Now I get the index of the node in the list and get the data at that point
					parental_event= dataline[self.nodePositionsDict[parent]]  # the event of the parent
					parent_event_list.append( (parent, parental_event) ) #  Store the parental_event with the name of the parent and place it in the tuple list
				conditional_event = ( tuple(parent_event_list), datapoint)  # A tuple of parent events combined with the child event
				self.nodePositions[i].addData(conditional_event)  # Have the node coorisponding to the current column keep track of the event
	
	def networkResetProbabilities(self):
		for node in self.nodes.values():
			node.fullResetProbabilities()

	def emKickoff(self):
		"""
			This function reads through self.originalDataSet which is a list that contains events (represented as lists)
			It then parses through that data to add the data correctly in the form of conditional events to all the nodes in network 
			Each event is assumed to be complete and not missing any data
		"""
		## convertFileIntoDataSet(filename) must be called before this function works
		"""
			This function returns a list of events (represented as lists) with the probability that each event occured
			in the form of [ ( ( ('name', value),('name', value) ) , %)]
		"""
		parentDataList= []
		newDataSet = []
		# This function looks at each line of data {aka an event} in the original originalDataSet
		for event in self.originalDataSet:
			# print event
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
				
#= baynodes  
				if realNodeData == '-':
					# DEAL WITH COPYING Eventlist and appending to each for each value in the distribution
					# probDistrubtionGivenNetwork(self, node, subeventConditionallist)
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
					# print eventList
				else:
					# just append a tuple of (realNodeData, 1) to each value in the eventList
					if not eventList:
						eventList.append([( (nodeName, realNodeData), 1)]  )
					else:
						for incompleteEvent in eventList:
							incompleteEvent.append( ( (nodeName, realNodeData), 1) )
			 # [ ( ( ('name', value),('name', value) ) , %)]
			for event in eventList:
				# print event
				tl = []
				totalp = 1.0
				for (partial_event, percentage) in event:
					totalp *= percentage
					tl.append(partial_event)
				newDataSet.append( (tl, totalp) ) 
		# print "\nnew data set "
		# for event in newDataSet:
		# 	print event
		return newDataSet


	def expectation(self):
		"""
			This function returns a list of events (represented as lists) with the probability that each event occured
			in the form of [ ( ( ('name', value),('name', value) ) , %)]
		"""
		newDataSet =[]
		# This function looks at each line of data {aka an event} in the original originalDataSet
		for i, event in enumerate(self.originalDataSet):
			parentDataList= []
			#print ("\t Calculating probability for event %d" %i)
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
		# print "\nnew data set "
		# for event in newDataSet:
		# 	print event
		return newDataSet

			# data is a tuple of ((("parent_name", parent_value),("parent_name", parent_value)), self_value)

					

	def maximization(self, dataSet):
		""" 
			This function reads through a list that contains events (represented as list) with the probability that they occured
			It then parses through that data to add the data correctly in the form of conditional events to all the nodes in network 
			Each event is assumed to be complete and not missing any data
		"""
		#addData(self, data, percentage= 1.0):
		# for node in self.nodePositions:
		# 	print "\n\n in maximization node name is "
		# 	print node.name
		# 	print "node value counts beforehand are "
		# 	for (val, key) in enumerate(node.value_counts):
		# 		print val
		# 		print key

		# print "\n\n"

		for event in dataSet:
			# print "\t ONE EVENT IS "
			# print event
			for i, node in enumerate(self.nodePositions): # go through every node in the header, which should be every node in the dataset/network
				# print event
				# print ("i = %d" %(i) )
				modifiedEvent = copy.deepcopy(event[0])
				# print modifiedEvent
				valueOfCurrentNode = modifiedEvent.pop(i)[1]
				### add the data to the relivant node in the form of a proper event, with the percetange that the event occured
				# print "event[1] in maximization is"
				# print event[1]
				# print "value of current node is "
				# print valueOfCurrentNode
				e = node.createEvent(modifiedEvent, valueOfCurrentNode) 
				node.addData(e, event[1])
			
		# for node in self.nodePositions:
		# 	print "\n\n in maximization node name is "
		# 	print node.name
		# 	print "node value counts AFTERWARDS are "
		# 	for (val, key) in enumerate(node.value_counts):
		# 		print val
		# 		print key

		# print "\n\n"

	def logLikelyhood(self):
		totalLikelyhood = 0.0
		multtotalLikelyhood= 1.0
		addTotalLikelyhood = 0.0
		for event in self.originalDataSet:
			# print "event in likelyhood is : "
			# print event
			"""
				clean line, 
				joint prob of clean line
				log of joint prob
				add result to total
			"""
			# jointProb(subeventConditionallist) 
			# subeventConditionallist is of the form [ (NodeName, value),(NodeName, value)]
			eventlist = []
			for i, realNodeData in enumerate(event):
				if realNodeData != '-':
					node = self.nodePositions[i]
					nodeName = node.name
					eventlist.append( (nodeName, realNodeData) )
			# print "\n cacluating joint prob for event: \n"
			jp = self.jointProb(eventlist)

			if jp != 0.0:
				# print jp
				multtotalLikelyhood =  multtotalLikelyhood * jp
				totalLikelyhood+= math.log(jp)
				addTotalLikelyhood += jp
		# print multtotalLikelyhood
		# print totalLikelyhood
		return totalLikelyhood
			#

	def emIteration(self):
		# This returns the log likelyhood of the model generated
		# self.printNetwork()
		#print "generating expectation "
		expectation  = self.expectation()
		#print "\n deepcopying nodes\n"
		self.previousNodes = copy.deepcopy(self.nodes)
		self.nodes = copy.deepcopy(self.originalNodes)
		
		#print "\nresetting node positions\n"
		self.nodePositions =[] # holds the actual node coorisponding to the index of a position
		for i, nodeName in enumerate(self.headerLine): # the first line of the file
		 	self.nodePositions.append(self.nodes[nodeName])
		
		#print "calculating model"
		self.maximization(expectation)

		#print "\n resting probabilities \n "
		self.networkResetProbabilities()

		#print "\n calculating likelyhood\n"	 	
		return self.logLikelyhood()
			  
		
	def expectationMaximazation(self, kickoff = 1, fancyPrintDict ={}):
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
			""
			
			print "maximization"	 	
			self.maximization(expectation)

			print "reseting network prob"
			self.networkResetProbabilities()
		else:
			print "ASSUMING PROBABILITIES HAVE BEEN GIVEN TO NETWORK"

		print "STARTING CONDTIONAL PROBABILTIES VALUES:"
		self.printConditionalProbNetwork(fancyPrintDict)
		
		print " getting likelyhood"
		previousLikelyhood = self.logLikelyhood()

		print " running itteration"
		currentLikelyhood = self.emIteration()

		dif = currentLikelyhood-previousLikelyhood
		iterationLikehoodTable = "\nTable Mapping iterations to likelyhood\niteration \t likelyhood"
		# iterationLikehoodTable +=  "\n%d\t%f" %(0,previousLikelyhood)
		iterationCounter = 1
		print "checking likelyhood comparison"
		while(dif > converganceThreashold ):
			iterationLikehoodTable +=  "\n%d\t%f" %(iterationCounter,previousLikelyhood)
			iterationCounter +=1 
			print ("Running another itteration as currentLikelyhood-previousLikelyhood = ", dif)
			# if(dif < converganceThreashold):
			# 	break;
			# print "\n"
			# print previousLikelyhood
			# print currentLikelyhood
			previousLikelyhood = currentLikelyhood
			currentLikelyhood = self.emIteration()
			dif = currentLikelyhood-previousLikelyhood
		print "Not running another itteration as currentLikelyhood-previousLikelyhood = %f" %(dif)
		
		# revert back to the previous itteration as that may be better then the current one
		# and it is know that they are not significantly different
		self.nodes = copy.deepcopy(self.previousNodes )
		iterationLikehoodTable += "\n"
		print iterationLikehoodTable
		return currentLikelyhood


				

class BayNode(object):
	"""
		This function serves to represent a node of a baysian network
		Methods:
			1. __init__
			2. addParent
				- Adds a node to be the parent of this node marking this node as the child of that node within that node, and that node as the parent within this one 
			3. getParents
				- returns the list of parents
			4. addData
				- adds an event that contains all the values of the each parent in addition to the value of this node when the parents are that value
			5. probAGivenB
				- this returns the value of parent given the child is a 
			6. getConditionalProb
			7. resetProbabilities - sets all proability based on events
			8. setConditionalProb - sets on probability
			9. getSubEventConditionalProbability
			10. createEvent 
			11. getProb
			12. guessValue
			13. guessProbAGivenB 	
	"""

	# def probabilityDistribution(self, nodeValueDict): # nodeValueDict[parentNode]
	# 	# This returns a tuple like ((value, %),(value, %))		
	# 	tupleList = []
	# 	for value in self.val_range:
	# 		probability = probabilityThisNodIs(value, nodeValueDict)
	# 		tupleList.append( (value, probability) )
	# 	return tuple(tupleList)


	# def probabilityThisNodIs(self, value, nodeValueDict):
	# 	for node, value in enumerate(nodeValueDict):
	# 		distribution=node.probabilityDistribution()
	# 		total= 1.0
	# 		if node in self.parents.values():
	# 			semitotal = 1.0

	# 		elif node in self.children.values():


	# def probabilityDistributionvtwo(self, nodeValueDict):
	# 	if nodeValueDict[self] == '-':
	# 		# This returns a tuple like ((value, %),(value, %))
	# 		tupleList = []
	# 		for value in self.val_range:
	# 			probability = 1.0 # probabilityThisNodIs(value, nodeValueDict)

	# 			for node, pvalue in enumerate(nodeValueDict):
	# 				distribution=node.probabilityDistributionvtwo()
	# 				total= 1.0
	# 				if node in self.parents.values():
	# 					semitotal = 1.0
	# 					for semiDistribution in distribution:


	# 				elif node in self.children.values():


	# 			thisGivenParentsSum = 1.0
	# 			listOfPotentialParentEvents =[]
	# 			for parentNode in self.parents.values():
	# 				newListOfPotentialParentEvents = []
	# 				distribution=parentNode.probabilityDistributionvtwo()
	# 				for semiDistribution in distribution:
	# 					for ppe in listOfPotentialParentEvents:
	# 						allButCurrentParent = copy.deepcopy(listOfPotentialParentEvents)
	# 						# add this parent event to the stack, for each possible outcome of this parent
	# 						allButCurrentParent.append(semiDistribution)
	# 						newListOfPotentialParentEvents.append(allButCurrentParent)
	# 				listOfPotentialParentEvents = newListOfPotentialParentEvents
	# 			for eachPotentialParentEvent in listOfPotentialParentEvents:
	# 				#make an event for the probability of this node given its parents, assuming its parents are certain values
	# 				probThisNodeGivenAllParentNodes =  getConditionalProb ( event(eachPotentialParentEvent, value)



	# 			# sum of (for each possible value of each parent node:
	# 			# 															prob(this given parents)
	# 			# 															* the parents are that


	# 			probability*= probThisNodeGivenAllParentNodes, 

	# 			tupleList.append( (value, probability) )
	# 		return tuple(tupleList)
	# 	else:
	# 		return (nodeValueDict[self], 1.0 )# Since we know the value of this node, there  is a probability of 1 this is the value


	# parents are in the form {'Name': BayNode}
	def __init__(self, name, val_range = [], parents = []):
		self.name = name
		self.children = {}
		self.val_range = val_range
		self.parents = {}
		self.parentList =[]
		if parents:
			for parentNode in parents:
				self.addParent(parentNode, parentNode.name)
		# self.parents = parents
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
			#print self.getConditionalProb(event)
		#print s

	def addParent(self, node, name):
		#self.parents.append(node, name)
		self.parents[name] = node
		node.children[self.name] = self
		self.parentList.append(name)

	# for node in network reset the probabilities
	# make sure reset calles setconditionalProb for every possible event
	def fullResetProbabilities(self):
		self.event_probibility = {}

		masterEventList =[]
		# Generate a list for all possible parental configurations
		if self.parentList:
			for parentName in self.parentList:
				parentNode = self.parents[parentName]
				if not masterEventList:
					# print "printing value counts of parent " + parentNode.name
					# print parentNode.value_counts
					for pvalue in parentNode.val_range:
						masterEventList.append([(parentName, pvalue)])
				else:
					newMasterList = []
					for pvalue in parent:
						copyList= [copy.deepcopy(sublist) for sublist in  masterEventList]
						# print "copyList " +self.name + " pvalue " + pvalue
						# print copyList
						for eventList in copyList:
							eventlist.append( (parentName,pvalue) )
						newMasterList.append(copyList)
					masterEventList = newMasterList
		else:
			masterEventList =[()]
		# For each value of this node | that configuration set the probability of that full event
		# print "master event list"
		# print masterEventList
		for eventlist in masterEventList:
			total = 0.0
			fullEventsList = []
			for cvalue in self.val_range:
				# print self.name + (" is creating an event from fullResetProbabilities with %s" %cvalue)
				# print "Event list is"
				# print eventlist
				# print "\t\t for master event list"
				# print masterEventList
				fullEvent = self.createEvent(eventlist, cvalue) 
				try:
					total += self.value_counts[fullEvent] # THIS MAY NEED A TRY SINCE THE EVENT MAY n
				except Exception:
					total += 0.0
				fullEventsList.append(fullEvent)
			for fullEvent in fullEventsList:
				# print self.name + (" is creating an event from fullResetProbabilities with %s" %fullEvent[1])
				# print "Event list is"
				# print eventlist
				# print self.name
				# print fullEvent
				try:
					self.event_probibility[fullEvent] = self.value_counts[fullEvent]/total
				except Exception as inst:
					# print type(inst)     # the exception instance
					# print inst.args      # arguments stored in .args
					# print inst           # __str__ allows args to be printed directly
					# print "NO VALUE COUNT FOR:    -ESTIMATING AT 0.0"
					# print fullEvent
					# print "total is "
					# print total
					self.event_probibility[fullEvent] = 0.0

		# print "\n\n"
		# print "print valid keys for " + self.name + " \n self = "
		# print self
		# print self.event_probibility.keys()
		# print "\n\n"
	
	def printConditionalProbabilityTable(self, fancyPrintDict ={}):
		# fancyPrintDict is in the form {(NodeName,Nodevalue): RealValue }
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

	def getParents():
		return self.parents

	# data is a tuple of ((("parent_name", parent_value),("parent_name", parent_value)), self_value)
	def addData(self, data, percentage= 1.0):
		# print self.name
		# print ("adding data:")
		# print data
		# print "\t" + self.name + " adding data for "
		# print data
		self.value_counts[data] = self.value_counts.get(data, 0.0) + percentage
		
	# Prob A|B = p(A)p(B|A)/p(B)
						# The event is of the form (("parent_name", parent_value)), child_value)) 
	def probAGivenB(self, parentValue, child, childValue):
		if child.parents[self.name] != self:
			print ("INVALD CALL OF METHOD, Child is not a child of parent")
			return None
		else:
			probA = getProb(parentValue)  # getProbViaConditional
			probBGivenA = child.getConditionalProb((self.name, parentValue), childValue)
			probB = child.getProb(B)      # getProbViaConditional
			return probA*probBGivenA/probB


	# def guessProbAGivenB(self, parentValue, child, childValue):
	# 	if child.parents[self.name] != self:
	# 		print ("INVALD CALL OF METHOD, Child is not a child of parent")
	# 		return None
	# 	else:
	# 		tupleList = []
	# 		guess prob of child, 
	# 		for each guess of child 
	# 		for value in self.val_range:
	# 		percentage = get probAGivenB 
	# 		add value, percentage into tupleList
	# 		return tupleList

	#  def probAGivenBvtwo(self, a, b):
		# p(a|b) = p(b|a)*p(a) / sum( p(b|Ai)*p(Ai) )



	# event is of the form ((("parent_name", parent_value),("parent_name", parent_value)), self_value)
	def getConditionalProb(self, event):
		# print "getConditionalProb for " + self.name
		# print "self = "
		# print self
		# print "event:"
		# print event
		# print "print valid keys for " + self.name
		# print self.event_probibility.keys()
		# print self.event_probibility
		return self.event_probibility[event]
		# # print "getting Cprobability for:"
		# # print event
		# total= 0.0;
		# for value in self.val_range:
		# 	total += self.value_counts[event[0],value]
		# # print "total is %d" %total
		# # print "number of times event occured is %d" %self.value_counts[event]
		# # print "cprob is %f" %(self.value_counts[event]/total)
		# return self.value_counts[event]/total



	def resetProbabilities(self):
		# to fix this and make everyProbability is stored correct 
		# go through each value this can be, for each value it parents can have as a permutation
		# self.value_counts = {} # is a dictionary mapping a tuple of ((("parent_name", parent_value),("parent_name", parent_value)), self_value) to the number of times it occurs
		for event in self.value_counts:
			self.setConditionalProb(event)

	def setConditionalProb(self, event):
		total= 0.0;
		for value in self.val_range:
			total += self.value_counts[event[0],value]
		self.event_probibility[event]=self.value_counts[event]/total

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
				# print parentnameEventTuple
				if parentName == parentnameEventTuple[0]: # if the parent names match
					# t = parentnameEventTuple_holder.pop(i)
					# print "print t "
					# print t
					# print "printing tuple t"
					# print tuple(t)
					tempTuple_holder.append(tuple((parentnameEventTuple_holder[i][0], parentnameEventTuple_holder[i][1]) ))
					# remove the item from the pass holder, and place into the holder 
					popped = True
					break;
			if popped == False:
				print("ERROR WHILE CREATING EVENT, Missing tuple for " + parentName)
		# print "printing tuple holder"
		# print tempTuple_holder
		# print "printing tuple tupleHolder"
		# print tuple(tempTuple_holder)
		return ( tuple(tempTuple_holder), currentNodeValue) # tuple(tuple(tempTuple_holder), currentNodeValue)
	

	# event is of the form <value>
	def getProb(self, event):
		# print "getting probability for:"
		# print event
		totalProb = 0.0 # self.value counts doesn't always have all the possible events, make sure it does
		for e in (relivant_events for relivant_events in self.value_counts if relivant_events[1] == event):
			condprob = self.getConditionalProb(e)
			for (parent_name, parent_value) in e[0]:
				condprob *= self.parents[parent_name].getProb(parent_value)
			totalProb += condprob
		return totalProb
		#for parent_name, parent in self.parents.iteritems():

	def getProbViaConditional(self, event):
		# print "getting probability for:"
		# print event

		totalProb = 0.0 # self.value counts doesn't always have all the possible events, make sure it does
		for e in (relivant_events for relivant_events in self.event_probibility if relivant_events[1] == event):
			condprob = self.getConditionalProb(e)
			for (parent_name, parent_value) in e[0]:
				condprob *= self.parents[parent_name].getProbViaConditional(parent_value)
			totalProb += condprob
		return totalProb

	# Truthfully for cases with grandparens I need another function that passes a dictionary of what I know about the event
	# def getRecursiveKnownProb


	# def guessValue(self, nodeValueDict): # returns ((value, %),(value, %))
	# 	tupleList = []
	# 	if self.intelligentGuess == False:
	# 		randomPercentageList =[]
	# 		total = 0.0
	# 		for value in self.val_range:
	# 			rand = random.uniform(1, 10000)
	# 			randomPercentageList.append(rand)
	# 			total += rand
	# 		for pvalue in randomPercentageList:
	# 			percentage = pvalue/total
	# 			value = percentage
	# 		random.shuffle(randomPercentageList)
	# 		for i, value in enumerate(self.val_range):
	# 			tupleList.append((value,randomPercentageList[1]))
	# 	else:
	# 		#  nodeValueDict # relates a nodeobject to its value in this line
	# 		if not self.parents:  # if this is a root node
	# 			for value in self.val_range:
	# 				if not self.guessBasedOnAllData:
	# 					tupleList.append((value,getProb(value)))
	# 				else: # I am meant to guess a based on b's value
	# 					probA= self.getProb(value)
	# 					childrenGivenParentsProb = 1.0
	# 					for childName, childNode in enumerate(self.children):
	# 						probBGivenA= childNode.getConditionalProb()
	# 					print "NOT SURE HOW I WOULD CALCULATE a Gender guess given the values of Wieght and Hieght"
	# 		else:
	# 			# THE FOLLOWING CODE WILL NOT WORK for multiple parents
	# 			# For multiple parents i need to break up thier guesses to create different value entries				
	# 			for parentName, parentNode in enumerate(self.parents):
	# 				parentValueDistribution = ()
	# 				if (nodeValueDict[parentNode]) == '-':
	# 					parentValueDistribution = parentNode.guessValue(nodeValueDict)
	# 				else:
	# 					parentValueDistribution = ((nodeValueDict[parentNode], 1))
	# 				###	
	# 				for value in self.val_range:
	# 					totalPercentage = 0.0	
	# 					for (parentValue, percentageTrue) in parentValueDistribution:
	# 							passTuple = ( ((parentName, parentValue)), value)
	# 							unwieghtedCP= self.getConditionalProb( passTuple)
	# 							weightedConditionalProb = unwieghtedCP * percentageTrue
	# 							totalPercentage += weightedConditionalProb
	# 					tupleList.append( (value,totalPercentage) )
	# 	return tuple(tupleList)





def multiEM(file, outFile, kickoff = 1):
	
	sys.stdout = open(outFile, "w")
	infile = file
	
	gender1 = BayNode('Gender', ('0', '1'))
	wieght1 = BayNode('Weight', ('0', '1'), [gender1])
	hieght1  = BayNode('Height', ('0', '1'), [gender1])

	if kickoff < 1:	
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
				
	net1 = BayNetwork(baynodes = {gender1.name :gender1, wieght1.name: wieght1, hieght1.name: hieght1})
	
	net1.convertFileIntoDataSet(infile)

	a = net1.expectationMaximazation(kickoff, fancyPrintDict)
	
	bestNet = net1

	for i in range(10):
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
		b = net2.expectationMaximazation(kickoff, fancyPrintDict)

		if b > a:
			bestNet = net2
			a = b
			print " Found a better answer"
		else:
			print "Not a better answer"
		print "Best Answer has a likelyhood of"
		print a

	print "\n\n Printing conditionalProbability for best iteration of EM:"
	bestNet.printConditionalProbNetwork(fancyPrintDict)


# The starting points of the learning
# The final conditional probability tables for each learning
# Plots of the likelihood vs number of iterations to demonstrate the convergence of your algorithm.

# Create all the nodes in the network,
# gender = BayNode('Gender', ('0', '1'))
# wieght = BayNode('Weight', ('0', '1'), [gender])
# hieght  = BayNode('Height', ('0', '1'), [gender])
# add them all to the network
# nodes are of the form {'Name': BayNode}
# net = BayNetwork(baynodes = {gender.name :gender, wieght.name: wieght, hieght.name: hieght})
# infile = "hw2dataset_30.txt"
# net.convertFileIntoDataSet(infile)
files=["hw2dataset_10.txt", "hw2dataset_30.txt","hw2dataset_50.txt","hw2dataset_70.txt","hw2dataset_100.txt"]
# files=["hw2dataset_10.txt",]
for fileName in files:
	outName = "OUTPUTFILE_statingValues_" + fileName
	# without kickoff
	multiEM(fileName, outName, 0)
	# with kickoff
	outName = "OUTPUTFILE_randomStart_" + fileName
	multiEM(fileName, outName, 1)

# net.expectationMaximazation()
# print "kickoff"
# k = net.emKickoff()
# print "maxmazation"
# net.maximization(k)
# print "reset prob"
# net.networkResetProbabilities()
# net.printNetwork()
# print net.logLikelyhood()

# probDictweight= {((('Gender', '0'),), '1'): .1, ((('Gender', '1'),), '1'): .5, ((('Gender', '1'),), '0'): .5, ((('Gender', '0'),), '0'):.9 }
# net.nodes['Weight'].event_probibility = probDictweight

# probDictgender = {((), '1'): 0.4, ((), '0'):0.6}
# net.nodes['Gender'].event_probibility = probDictgender

# print "after modifying probability negitively"
# print net.logLikelyhood()

# probDictgender = {((), '1'): 0.1, ((), '0'):0.9}
# net.nodes['Gender'].event_probibility = probDictgender

# print "after modifying probability negitively"
# print net.logLikelyhood()

# print "printing jp 1"
# totaljp= 0.0
# for event in net.originalDataSet:
# 	eventlist = []
# 	for i, realNodeData in enumerate(event):
# 		if realNodeData != '-':
# 			node = net.nodePositions[i]
# 			nodeName = node.name
# 			eventlist.append( (nodeName, realNodeData) )
# 	# print "\n cacluating joint prob for event: \n"
# 	jp = net.jointProb(eventlist)
# 	totaljp+=jp
# 	# print jp
# print totaljp
# net.emIteration()
# print "printing jp 2"
# totaljp= 0.0
# for event in net.originalDataSet:
# 	eventlist = []
# 	for i, realNodeData in enumerate(event):
# 		if realNodeData != '-':
# 			node = net.nodePositions[i]
# 			nodeName = node.name
# 			eventlist.append( (nodeName, realNodeData) )
# 	# print "\n cacluating joint prob for event: \n"
# 	jp = net.jointProb(eventlist)
# 	totaljp+=jp
# 	# print jp
# print totaljp
# net.emIteration()
# print "printing jp 3"
# totaljp= 0.0
# for event in net.originalDataSet:
# 	eventlist = []
# 	for i, realNodeData in enumerate(event):
# 		if realNodeData != '-':
# 			node = net.nodePositions[i]
# 			nodeName = node.name
# 			eventlist.append( (nodeName, realNodeData) )
# 	# print "\n cacluating joint prob for event: \n"
# 	jp = net.jointProb(eventlist)
# 	totaljp+=jp
# 	# print jp
# print totaljp
# l = net.emIteration()

# print "after em "
# print l # net.logLikelyhood()

# # print "expectation"
# # e = net.expectation()
# print "em iteration"
# l = net.emIteration()
# print "likelyhood is "
# print l

# net.expectationMaximazation()





"""
	This function returns a list of events (represented as lists) with the probability that each event occured
	in the form of [ ( ( ('name', value),('name', value) ) , %)]
"""

# print "\n\n\n printing events at the end"
# for event in e:
# 	print event



# infile = "hw2dataset_10.txt"
# infile = open(infile)
# gender = BayNode('gender')
# gender.val_range = ('0', '1')
# with infile as inf:
# 	datapoints = (line.strip().split('\t') for line in inf)
# 	for point in datapoints:
# 		# print point
# 		# print point[0]
# 		# data is a tuple of ((("parent_name", parent_value),("parent_name", parent_value)), self_value)
# 		data = ( () , point[0] ) # there are no parents so it is an empty tuple
# 		gender.addData(data)
# 	gender.resetProbabilities()

# print "\n\n\n\n"
# event1 = '0'
# probE1 = gender.getProb(event1)
# print ("probability that gender = %s is %f" %(event1, probE1) )
# event2 = '1'
# probE2 = gender.getProb(event2)
# print ("probability that gender = %s is %f" %(event2, probE2) )
# total = probE1+ probE2
# print ("\n\nthis sums to %f \n\n" %(total) )





# print gender.value_counts
# getting Cprobability for:
# ((), '0')
# total is 185
# number of times event occured is 119
# cprob is 0.643243
# probability that gender = 0 is 0.643243
# getting probability for:

# getting Cprobability for:
# ((), '1')
# total is 185
# number of times event occured is 66
# cprob is 0.356757
# probability that gender = 1 is 0.356757


# this sums to 1.000000 


## check if probabilties via em are non sensicle, 
