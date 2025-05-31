import ast
import networkx as nx
import collections
import math

import pandas as pd

#def colocationCreate(threshold, device_locations, window_number):
def colocationCreate(threshold, device_locations):

	# Create a dictionary for nodes and the location
	#device_locations = {}

	#id_devices = dataset["id_device"].tolist()

	# 1- Get the maxiumum value of rows in "id_device" column where it is the number of devices and iterate through it
	#for device in id_devices:
		
		# Get the device location
	#	location = dataset.device_locations.loc[dataset['id_device'] == device].item()

		# Convert the string to a list using ast
	#	location = ast.literal_eval(location)

		# If the device is dynamic get the first device position
	#	if type(location[0]) == list:
	#		x = location[0][0]
	#		y = location[0][1]
		
		# If the device is static
	#	elif type(location[0]) == float:
	#		x = location[0]
	#		y = location[1]

		# 2- For each device as key and the x and y location as it values
	#	device_locations[str(device)] = [float(x), float(y)]
	
	# Devices IDS
	id_devices = device_locations.keys()

	# 3- Create an empty graph
	G = nx.Graph()
	# Add all the nodes to the graph
	G.add_nodes_from(id_devices)

	device_locations_list = list(device_locations.items())

	no_devices = len(device_locations_list)

	count_loc = 0
	count_edges = 0

	# 4- Loop through all nodes and create
	for node1, location_node1 in device_locations_list:

		x1, y1 = location_node1
		
		# Check from the next node and for rest of the nodes and create an distance edge
		for node2Location in device_locations_list[count_loc+1:]:

			node2 = node2Location[0]
			x2, y2 = node2Location[1]

			distance = math.sqrt( ((x2-x1)**2) + ((y2-y1)**2) )
			norm_distance = (distance)/1.41 #1.41 is the max distance
	
			# Calcuate the distance between two devices using "Distance = sqrt((x_2-x_1)^2+(y_2-y_1)^2))"
			edge_weight = 1 - norm_distance

			if edge_weight >= threshold:
				
				# Make sure the edges between nodes not negitve 
				if edge_weight <= 0:
					continue
				else:
					if edge_weight > 0.0:
						if node1.isnumeric() and node2.isnumeric():
							G.add_edge(int(node1), int(node2), weight=edge_weight)
						elif node1.isnumeric() and node2.isalpha():
							G.add_edge(int(node1), str(node2), weight=edge_weight)
						elif node1.isalpha() and node2.isnumeric():
							G.add_edge(str(node1), int(node2), weight=edge_weight)
						elif node1.isalpha() and node2.isalpha():
							G.add_edge(str(node1), str(node2), weight=edge_weight)

				
				count_edges = count_edges + 1
				
		count_loc = count_loc + 1

	# Creating the Adjacency Matrix
	#labels = nx.draw_networkx_labels(G,pos=nx.spring_layout(G))
	#adjacency_matrix = nx.to_numpy_matrix(G)
	
	edge_list = nx.generate_edgelist(G)
	
	# Edge list dataframe
	edge_list_dataframe = nx.to_pandas_edgelist(G)
	edge_list_dataframe[["source", "target", "weight"]]

	edge_list_dataframe['source'] = edge_list_dataframe.source.astype(str)
	edge_list_dataframe['target'] = edge_list_dataframe.target.astype(str)

	# For edgelist save which window
	#file_name = 'results/CLOR/CLOR_Edgelist_W'
	#file_name += str(window_number)
	# Save the CLOR to CSV File
	#nx.write_edgelist(G, file_name, delimiter=',')

	return edge_list, count_edges, edge_list_dataframe


'''
# Remove all edges below 0.8
df_reduce = pd.read_csv("dataset/new_relations/CLOR_Edgelist.csv", names=["from", "to", "weight"])
new_relations_reduced = df_reduce.drop(df_reduce[df_reduce.weight < 0.8].index)

export_csv = new_relations_reduced.to_csv('dataset/new_relations/CLOR_Edgelist_reduced.csv', index = None)
'''