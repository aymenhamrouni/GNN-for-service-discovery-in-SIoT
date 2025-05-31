# Python Packages required for the Project
import pandas as pd
import numpy as np
import random
from tqdm import tqdm, tqdm_notebook
import decimal
import networkx as nx

import matplotlib.pyplot as plt

from statistics import mean

import findNodes

from random import sample

import getAllDevicesPositionTimes
import getDevicesWithinWindow

import createColocationDevicesRelationsDynamic
import createSocialFriendshipOnwershipRelations

import communityDetectionLouvain

import sys

# Dataset where the devices infomration such as devices locations and ownership data.
dataset = pd.read_excel(open("data/small_area_data_no_public.xlsx",'rb'), sheet_name='Sheet1')

# INITIAL PHASE (Preprocessing Stage)
# -- Collect all the devices data within a speific space and at time zero.
id_devices = dataset["id_device"].tolist()

# Pass the dataset and a list of all the devices you want to get the times and locations of it 
device_locations_time = getAllDevicesPositionTimes.getAllDevicesPositionTimes(dataset, id_devices)

# Select devices based in the desired window (device_locations_time, window_start, window_end)
# 0.0007 = 1 mintue
# 0.034723 = 1 hour

windowSizePeriod = 0.034723
windowSizeStart = 21
windowSizeEnd = 22

# Create a list where, x the start time, and y is the end time. Then slice it based on the 
# jump value (windowSizePeriod)
def drange(x, y, jump):
  while x < y:
    yield float(x)
    x += decimal.Decimal(jump)

def diff(list1, list2):
  #return list(set(list1).symmetric_difference(set(list2)))  # or return list(set(list1) ^ set(list2))
  #return list(set(list1) - set(list2))
  #return np.setdiff1d(list1, list2)
  return [i for i in list1 if i not in list2]

windows = list(drange(windowSizeStart, windowSizeEnd, windowSizePeriod))
print("How many loop needed to run the code:", len(windows))
k = 0

# -------------------------------------------- MAIN PART --------------------------------------
# Loop through the windows for Snapshots Community Detection

# Dataframes for results
df_snaps_CLOR = pd.DataFrame(columns = ['Window Number', 'Relation', 'Window Start','Window End',
'Number of Communities', 'Largest Community', 'Avarage Community Size', 'Time to process', 
'Number of Nodes', 'Number of Edges', 'Coverage', 'Performance', 'Modularity', 'Density', 'Number of Devices', 
'NoofNewDevices', 'NoofExitDevices']) 

df_snaps_SFOR = pd.DataFrame(columns = ['Window Number', 'Relation','Window Start','Window End',
'Number of Communities', 'Largest Community', 'Avarage Community Size', 'Time to process', 
'Number of Nodes', 'Number of Edges', 'Coverage', 'Performance', 'Modularity', 'Density', 'Number of Devices', 
'NoofNewDevices', 'NoofExitDevices']) 

result = []
#len(windows)-1
while k < 1:
  # Testing
  #if k == 5:
  #  df_snaps_CLOR.to_excel("results/snapshots_test.xlsx", index=None, header=True)
  #  exit()

  result.clear()

  window_start = windows[k]
  window_end = windows[k+1]

  # window_devices_ids it returns the devices ids to the user
  window_devices_locations, window_devices_ids, selected_devices_ids_locations = getDevicesWithinWindow.getDevicesWithinWindow(device_locations_time, window_start, window_end)

  NoofDevices = len(window_devices_ids)
  newDevicesIDS = list(selected_devices_ids_locations.keys())

  if k == 0:
    NoofNewDevices = 0
    NoofExitDevices = 0
  else:
    newDevices = findNodes.newNodes(oldDevicesIDS, newDevicesIDS)
    exitDevices = findNodes.removeNodes(oldDevicesIDS, newDevicesIDS)

    NoofNewDevices = len(newDevices)
    NoofExitDevices = len(exitDevices)
  
  for values in selected_devices_ids_locations.items():
   plt.plot([values[1][0]], [values[1][1]], marker='o', markersize=1, color="red")
  plt.show()
  

  # convert the id of devices into integer
  window_devices_ids = [int(i) for i in window_devices_ids] 

  if k == 0 or k == 1 or k == 9:
    # Select the devices in the dataframe that is mentioned ------ NO NEED!
    #selected_devices = dataset[dataset['id_device'].isin(window_devices_ids)]

    # 1- Create the colocation adjencey matrix, Specify the thershold to drop the edges below 
    thresholdValue = 0.95
    
    # Bulid the relations matrix for CLOR
    print("Creating a CLOR relation with thershold of %s in window number %s"% (thresholdValue, k))
    colocationEdgelist, edgesCount, EdgeDataframeCLOR = createColocationDevicesRelationsDynamic.colocationCreate(thresholdValue, selected_devices_ids_locations)
    
    # Bulid the relations of Social Friendship Onwers
    print("Creating a SFOR relation in window number", k)
    SFOREdgelist, edgesCount, EdgeDataframeSFOR = createSocialFriendshipOnwershipRelations.createSFOOR(selected_devices_ids_locations, dataset)
    # Bulid the Social Friendship and Onwership relation SFOR for the selected community FIX THIS
    #createSocialFriendshipOnwershipRelations.createSFOOR(closeCLORComLabel, smallCLORComTuples, dataset)
    print("Starting plot")
    print(EdgeDataframeSFOR)
    G_SFOR = nx.read_edgelist(SFOREdgelist)
    random_nodes = sample(list(G_SFOR.nodes), 500)
    G_SFOR.remove_nodes_from(random_nodes)
    nx.draw(G_SFOR, with_labels = True)
    plt.show()

    # THIS ADDITIONAL BUT IMPORTANT TO MAKE SURE THE NUMBER OF DEVICES NOT INCREASING Remove the additional devices that are not in the selected devices
    G_CLOR = nx.read_edgelist(colocationEdgelist)
    window_devices_ids = list(map(str, window_devices_ids))
    # REMEMBER: Order of the diff function for the passing lists are important 
    NotInTheWindow = diff(list(G_CLOR.nodes),window_devices_ids)
    print(NotInTheWindow)
    #exit()
    G_CLOR.remove_nodes_from(NotInTheWindow)
    oldDevicesIDS = list(G_CLOR.nodes)
    # -----------------

    # Detect the communities for CLOR
    print("Detect the communities a CLOR")
    LargestcommunitySize, noOfCommunities, timeProcess, avgCommunitiesSize, medianCommunity, NoNodes, NoEdges, Coverage, Performance, Modularity, Density = communityDetectionLouvain.commmunityDetection_basic(G_CLOR)
    
    print("basic",LargestcommunitySize,noOfCommunities,timeProcess,avgCommunitiesSize,medianCommunity,NoNodes,NoEdges,Coverage,Performance,Modularity,Density)
    
    LargestcommunitySize, noOfCommunities, timeProcess, avgCommunitiesSize, medianCommunity, NoNodes, NoEdges, Coverage, Performance, Modularity, Density = communityDetectionLouvain.commmunityDetection_GNN(G_CLOR,0,k)
    print("GNN",LargestcommunitySize,noOfCommunities,timeProcess,avgCommunitiesSize,medianCommunity,NoNodes,NoEdges,Coverage,Performance,Modularity,Density)

    result.extend((k + 1, "CLOR", window_start, window_end, noOfCommunities, LargestcommunitySize, avgCommunitiesSize, timeProcess, NoNodes, NoEdges, Coverage, Performance, Modularity, Density, NoofDevices, NoofNewDevices, NoofExitDevices))
    df_snaps_CLOR.loc[len(df_snaps_CLOR), :] = result
    print("How many communities:",noOfCommunities)
    print("CLOR no. comm", result[4])
    print("CLOR avg. comm", result[6])
    result.clear()

    print("Detect the communities a SFOR")

    # THIS ADDITIONAL BUT IMPORTANT TO MAKE SURE THE NUMBER OF DEVICES NOT INCREASING Remove the additional devices that are not in the selected devices
    G_SFOR = nx.read_edgelist(SFOREdgelist)
    # REMEMBER: Order of the diff function for the passing lists are important 
    NotInTheWindow = diff(list(G_SFOR.nodes),window_devices_ids)
    G_SFOR.remove_nodes_from(NotInTheWindow)
    # -----------------

    LargestcommunitySize, noOfCommunities, timeProcess, avgCommunitiesSize, medianCommunity, NoNodes, NoEdges, Coverage, Performance, Modularity, Density = communityDetectionLouvain.commmunityDetection_basic(G_SFOR)
    print("basic",LargestcommunitySize,noOfCommunities,timeProcess,avgCommunitiesSize,medianCommunity,NoNodes,NoEdges,Coverage,Performance,Modularity,Density)
    LargestcommunitySize, noOfCommunities, timeProcess, avgCommunitiesSize, medianCommunity, NoNodes, NoEdges, Coverage, Performance, Modularity, Density = communityDetectionLouvain.commmunityDetection_GNN(G_SFOR,0,k)
    print("GNN",LargestcommunitySize,noOfCommunities,timeProcess,avgCommunitiesSize,medianCommunity,NoNodes,NoEdges,Coverage,Performance,Modularity,Density)

    
    
    
    
    
    result.extend((k + 1, "SFOR", window_start, window_end, noOfCommunities, LargestcommunitySize, avgCommunitiesSize, timeProcess, NoNodes, NoEdges, Coverage, Performance, Modularity, Density, NoofDevices, NoofNewDevices, NoofExitDevices))
    df_snaps_CLOR.loc[len(df_snaps_CLOR), :] = result
    print("SFOR no. comm", result[4])
    print("SFOR avg. comm", result[6])
    result.clear()
    
    result.append(window_start)
    result.append(window_end)
    result.append(noOfCommunities)
    result.append(LargestcommunitySize)
    result.append(avgCommunitiesSize)
    result.append(timeProcess)
    result.append(NoNodes)
    result.append(NoEdges)
    result.append(Coverage)
    result.append(Performance)
    
    # Old Devices IDS

    # Add the results to the dataframe
  k = k + 1
  if k == 4:
    df_snaps_CLOR.to_excel("results/snapshots_testing_1.xlsx", index=None, header=True)
    exit()

df_snaps_CLOR.to_excel("results/snapshots_final_2.xlsx", index=None, header=True)

# ---------------------- DYNAMIC COMMUNITIES PART ------------------------------

# -- Build the relations (CLOR, SFOR) between the selected devices at time zero.

# -- Find the communities in each relations (CLOR, SFOR) at time zero.