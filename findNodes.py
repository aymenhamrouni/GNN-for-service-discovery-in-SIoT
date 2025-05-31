def newNodes(list1, list2):
    # Additional values in second list (oldDeviceIDS)
    return list(set(list2).difference(list1))

def removeNodes(list1, list2):
    # Missing values in second list (newDevicesIDS)
    return list(set(list1).difference(list2))