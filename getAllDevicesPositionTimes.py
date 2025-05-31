import ast
from collections import defaultdict

def getAllDevicesPositionTimes(dataset, id_devices):
    device_locations = defaultdict(list)

    for device in id_devices:
        # Get the device location
        location = dataset.device_locations.loc[dataset['id_device'] == device].item()

        # Convert the string to a list using ast
        location = ast.literal_eval(location)

        # If the device is dynamic get the first device position
        if type(location[0]) == list:

            for i in range(0,len(location)):
                
                x = location[i][0]
                y = location[i][1]
                timestart = (float(location[i][2])/(3600*24))
                timeend = (float(location[i][3])/(3600*24))
                
                # Check if the key already added or not
                if str(device) in device_locations.keys():
                    device_locations[str(device)].append(float(x))
                    device_locations[str(device)].append(float(y))
                    device_locations[str(device)].append(float(timestart))
                    device_locations[str(device)].append(float(timeend))
                else:
                    device_locations[str(device)] = [float(x), float(y), float(timestart), float(timeend)]

        # If the device is static
        elif type(location[0]) == float:
            x = location[0]
            y = location[1]
            device_locations[str(device)] = [float(x), float(y)]
    
    return device_locations