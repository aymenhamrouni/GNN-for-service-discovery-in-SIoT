'''
Create an Onwership device graph between the users.
  - From the social network between the onwers. Do the following.
  	- When the onwer is friend than create an edge of "0.75" between the devices.
  	- When the onwer is friend of friend than create edge of "0.5" between the devices.
'''

import networkx as nx
import pandas as pd
import ast
import collections
import math
import itertools
import random

# Get the nodes location in X and Y for each node and put it in dictionary

# Create a dictionary for nodes and the location
device_onwers = {}

from time import sleep

# function to check if the onwer is exist in the dictionary
def checkKey(dict, key): 
      
    if key in dict.keys(): 
        return True
    else: 
        return False

def createSFOOR(SelectedDevices, df):

	# Convert the device ids to int
	SelectedDevices = [ x for x in SelectedDevices ]
	
	df_random = pd.read_excel(open("data/random_devices.xlsx",'rb'), sheet_name='Sheet1')

	# 1- Loop through all the onwers in the dataset; exculde the public devices.
	#    Create a dictionary of owners and devices they own.
	for device in SelectedDevices:
		# Get the device location
		if device.isalpha() == True:
			continue
		else:
			if int(device) < 19999:
				onwerid = df.id_user.loc[df['id_device'] == int(device)].item()
			else:
				onwerid = df_random.id_user.loc[df_random['id_device'] == int(device)].item()

			# Check if the onwer is already there than append to the same key other wise create a new key (onwer)
			if checkKey(device_onwers, str(onwerid)) == True:
				device_onwers[str(onwerid)].append(int(device))
			elif checkKey(device_onwers, str(onwerid)) == False:
				device_onwers[str(onwerid)] = [int(device)]

	# 2- Check all pairs of owners to determine if they have relationship or not.
	#    Create an empty graph with all devices
	G_owners = nx.Graph()

	owners_list = []
	edge_weight_onwed = 1.0

	# Loop through the dictionary of Key -> Onwer ID and the Values -> Devices IDs
	for onwer, listofdevice in device_onwers.items():
		comb_devices = list(itertools.combinations(listofdevice, 2))

		# creating an edges between the devices owned by one owner with weight = 1
		for i in comb_devices:
			node1, node2 = i

			if edge_weight_onwed > 0.0:
				G_owners.add_edge(str(node1), str(node2), weight=edge_weight_onwed)
		owners_list.append(onwer)

	onwer_combinations = list(itertools.combinations(owners_list, 2))

	# 3- If owner1 has an edge with onwer2 than create an edge of weight 0.75 between all the devices they own.

	# Load the generated social graph from file
	# This socail network is created in "generateRandomSN.py" using Watts-Strogatz algorithm
	#SN = nx.Graph()

	# Generate a Graph (Social Network) with n is number of nodes. 
	# How many onwers in the dataset beside the public onwed devices
	#nn = 4000
	#p : float is the probability of adding a new edge for each edge
	# regularity (p=0) and disorder (p=1), also small world network is p=0.5
	#pp= 0.5
	#k Each node is connected to k nearest neighbors in ring topology,
	# This can be the friends for each nodes. We asume 10.
	#kk=6
	#SN = nx.watts_strogatz_graph(nn, kk, pp, seed=None)

	SN = nx.read_edgelist('data/SNOnwers_Edgelist_Final_NoPublic', delimiter=',', nodetype=int, encoding="utf-8")

	# Loop thourgh all the edges that represnets friendships between onwers

	# check for friends of friends and beyond and create an edge for it
	# By using the shortest path between two different nodes and make the result is (1/shortest path distance)

	for i in onwer_combinations:
		onwer_one = i[0]
		onwer_two = i[1]
		if onwer_one != onwer_two:
			# to make sure that the relation is not between public device and private owner
			if int(onwer_one) > 0 and int(onwer_two) > 0:
				try:
					distance_owner = nx.shortest_path_length(SN,source=int(onwer_one),target=int(onwer_two))
				except:
					continue
				
				
				# Friendship distance thershold. 3 friends of friends
				if distance_owner > 2:
					continue
				else:
					# the edge weight is the 1 divided by the distance number of nodes 
					friends_weight = 1 / (distance_owner+1)

					# Add edges between the devices of owner 1 and owner 2
					for device1 in device_onwers[onwer_one]:
						for device2 in device_onwers[onwer_two]:
							if friends_weight > 0.0:
								G_owners.add_edge(str(device1), str(device2), weight=friends_weight)

	#fh=open("relations/SFOOR_Edgelist_notComplete.csv",'wb')
	#nx.write_weighted_edgelist(G_owners, fh, delimiter=',')
	#adjacency_matrix = nx.to_numpy_matrix(G_owners)
	
	edge_list = nx.generate_edgelist(G_owners)

	noEdges = G_owners.number_of_edges()

	# Edge list dataframe
	edge_list_dataframe = nx.to_pandas_edgelist(G_owners)
	edge_list_dataframe[["source", "target", "weight"]]

	edge_list_dataframe['source'] = edge_list_dataframe.source.astype(str)
	edge_list_dataframe['target'] = edge_list_dataframe.target.astype(str)

	return edge_list, noEdges, edge_list_dataframe


# Getting all the devices in the same community using this function
def getAll(mydata, key):
    return [item[0] for item in mydata if item[1] == key]
