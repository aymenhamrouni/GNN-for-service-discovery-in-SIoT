def getDevicesWithinWindow(device_locations_time, window_start, window_end):
    selected_devices = []
    selected_devices_ids = []
    selected_devices_ids_locations = {}

    # To get the values of speific device
    #print(device_locations_time['5290'])

    for i in device_locations_time:
        #starttimes = []
        #endtimes = []

        dataDevices = device_locations_time[str(i)]

        # if the device is static and not having timestamp
        if len(dataDevices) == 2:
            x_position = dataDevices[0]
            y_position = dataDevices[1]

            if x_position > 0.6 and x_position < 0.8 and y_position > 0.0 and y_position < 0.6:
                selected_devices.append([x_position, y_position])
                selected_devices_ids.append(i)

                # Device id as key and the x and y location as it values
                selected_devices_ids_locations[str(i)] = [float(x_position), float(y_position)]

        else:
            
            dataLength = len(dataDevices)
            j = 0

            while j < dataLength:

                x_position = dataDevices[j]
                y_position = dataDevices[j + 1]

                device_start_time = j + 3
                device_end_time = j + 2

                
                if not window_end <= dataDevices[device_start_time]: 
                    #print("Working")
                    #if window_start < dataDevices[device_start_time] and window_end > dataDevices[device_end_time]:
                    #    selected_devices.append([x_position, y_position])
                    #    selected_devices_ids.append(i)

                    if window_start <= dataDevices[device_end_time] and window_start <= dataDevices[device_start_time] and window_end >= dataDevices[device_start_time] and window_end <= dataDevices[device_end_time]:
                        
                        if x_position > 0.6 and x_position < 0.8 and y_position > 0.0 and y_position < 0.6:
                            selected_devices.append([x_position, y_position])
                            selected_devices_ids.append(i)

                            # Device id as key and the x and y location as it values
                            selected_devices_ids_locations[str(i)] = [float(x_position), float(y_position)]

                    if window_start > dataDevices[device_end_time] and window_end < dataDevices[device_end_time]:
                        
                        if x_position > 0.6 and x_position < 0.8 and y_position > 0.0 and y_position < 0.6:
                            selected_devices.append([x_position, y_position])
                            selected_devices_ids.append(i)

                            # Device id as key and the x and y location as it values
                            selected_devices_ids_locations[str(i)] = [float(x_position), float(y_position)]

                j = j + 4
            
            # The 4th,8th... etc. element in the list is the start time for the device
            #starttimes = dataDevices[3::4]
            # The 3rd,6th... etc. element in the list is the end time for the device
            #endtimes = dataDevices[2::4] 
            
            # Return how many elements are above and below the start and end
            '''
            try:
                aboveStart = [ n for n,i in enumerate(starttimes) if i >= start ][0]
                belowEnd = [ n for n,i in enumerate(endtimes) if i <= end ][0]
            except IndexError:
                aboveStart = 0
                belowEnd = 0

            if aboveStart > 0 and belowEnd > 0:
                selected_devices.append(i)
                
                print(i)

                print(" ")
                print(aboveStart)
            '''

    print("How many selected_devices", len(selected_devices))
    print("*"*50)

    return selected_devices, selected_devices_ids, selected_devices_ids_locations