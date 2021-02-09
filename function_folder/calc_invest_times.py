###calc_invest_times.py#########
def check_orientation(df, index_loc, extra_coords):

    import numpy as np

	## main function for testing orientation of the head ##
	## this is also a relatively new addition, based on some coordinate math ##
	## essentially we are calculating whether nose is closest to the center point of the subject chamber ##
	## to do this, we are calculating the distances to all three head body parts ##

    orientation = 'not_oriented'
    x_center = extra_coords['x_center']
    y_center = extra_coords['y_center']

    dist_to_x = np.sqrt(((x_center[0] - df['nose']['x'].loc[index_loc])**2) + ((x_center[1] - df['nose']['y'].loc[index_loc])**2))
    dist_to_y = np.sqrt(((y_center[0] - df['nose']['x'].loc[index_loc])**2) + ((y_center[1] - df['nose']['y'].loc[index_loc])**2))
	
    if dist_to_x > dist_to_y:        
	    distance_to_nose = np.sqrt(((y_center[0] - df['nose']['x'].loc[index_loc])**2) + ((y_center[1] - df['nose']['y'].loc[index_loc])**2))
	    distance_to_l_ear = np.sqrt(((y_center[0] - df['left ear']['x'].loc[index_loc])**2) + ((y_center[1] - df['left ear']['y'].loc[index_loc])**2))
	    distance_to_r_ear = np.sqrt(((y_center[0] - df['right ear']['x'].loc[index_loc])**2) + ((y_center[1] - df['right ear']['y'].loc[index_loc])**2))
    elif dist_to_x < dist_to_y:
	    distance_to_nose = np.sqrt(((x_center[0] - df['nose']['x'].loc[index_loc])**2) + ((x_center[1] - df['nose']['y'].loc[index_loc])**2))
	    distance_to_l_ear = np.sqrt(((x_center[0] - df['left ear']['x'].loc[index_loc])**2) + ((x_center[1] - df['left ear']['y'].loc[index_loc])**2))
	    distance_to_r_ear = np.sqrt(((x_center[0] - df['right ear']['x'].loc[index_loc])**2) + ((x_center[1] - df['right ear']['y'].loc[index_loc])**2))
	
	## and changing the orientation variable to 'oriented' if the distance to nose is indeed the lowest value ##
	
    if distance_to_nose == np.min([distance_to_nose, distance_to_l_ear, distance_to_r_ear]):
	    orientation = 'oriented'
		
    return orientation




### export labelled frames function that allows you to take a video and create labelled frames with the output head vector

#calc_invest_times.py
### funtion for calculation investigation times on only one video
#should be able to replace the the big calc function from the main app with this


def calculate_investigation_times_single(df, possible_places, extra_coords):
	
	import numpy as np
	import pandas as pd
	import matplotlib.pyplot as plt
	from shapely.geometry import Point
	from shapely.geometry.polygon import Polygon
	import cv2
	from tqdm import tqdm
	
	bodyparts = np.unique(df.columns.get_level_values(0))[1:]

	arr = np.zeros(shape = (len(df), len(bodyparts), len(possible_places)))

	### now we should check the coordinates of each bodypart in each frame
	print('Calculating Investigation Times: ')
	for row in tqdm(range(len(df))):
		for j in range(len(bodyparts)):
			arr[row][j] = check_coords(df[bodyparts[j]][['x', 'y']].loc[row].values, possible_places)
			
	print('Array Constructed!')

	### set which patterns mean x vs y investigation, only for the first three bodyparts (nose and ears, cuz we don't care about tail base yet)
	x_inv = np.array([[1., 0., 1., 0., 0.]])
	y_inv = np.array([[0., 1., 0., 0., 1.]])

	### now we want to check each frame in our array, and create a frame_val array that holds info about where the mouse's head was detected
	z = -1
	frame_val = np.zeros(shape = len(arr), dtype = 'object')
	for frame in tqdm(range(len(arr))):
		z = z + 1
		comparison_x = arr[frame][0:1] == x_inv
		comparison_y = arr[frame][0:1] == y_inv

		if comparison_x.all() == True:
			if check_orientation_single(df, z, extra_coords) == 'oriented':
				frame_val[z] = 'X Investigation'
			elif check_orientation_single(df, z, extra_coords) == 'not_oriented':
				frame_val[z] = 'X Close'
		elif comparison_y.all() == True:
			if check_orientation_single(df, z, extra_coords) == 'oriented':
				frame_val[z] = 'Y Investigation'
			elif check_orientation_single(df, z, extra_coords) == 'not_oriented':
				frame_val[z] = 'Y Close'
		else:
			frame_val[z] = 'Somewhere else'
		
	print('Investigation Times Calculated!!')

	return frame_val


def calculate_investigation_times_multi(df_in, output_dir, bodypart = 'nose'):

	import os
	import numpy as np
	import pandas as pd
	import matplotlib.pyplot as plt
	from shapely.geometry import Point
	from shapely.geometry.polygon import Polygon
	import cv2
	from tqdm import tqdm
	from function_folder.utils import check_coords

	arr = np.zeros(shape = (len(df_in), len(bodyparts), len(possible_places)))

        ### now we should check the coordinates of each bodypart in each frame
        ### Below starts to look a lot like the single version
	print('Loading in bodypart coordinates for each frame')
	for row in tqdm(range(len(df_in))):
		for j in range(len(bodyparts)):
			arr[row][j] = check_coords(df_in[bodyparts[j]][['x', 'y']].loc[row].values, possible_places)

        ### set which patterns mean x vs y investigation, only for the first three bodyparts (nose and ears, cuz we don't care about tail base yet)
		if bodypart == 'nose':
			x_inv = np.array([[1., 0., 1., 0., 0.]])
			y_inv = np.array([[0., 1., 0., 0., 1.]])

        if bodypart == 'nose_and_ears':
            x_inv = np.array([[1., 0., 1., 0., 0.], [1., 0., 1., 0., 0.], [1., 0., 1., 0., 0.]])
            y_inv = np.array([[0., 1., 0., 0., 1.], [1., 0., 1., 0., 0.], [1., 0., 1., 0., 0.]])       


        print('Now comparing bodypart coordinates against annotated zones')
        ### now we want to check each frame in our array, and create a frame_val array that holds info about where the mouse's head was detected
        z = -1
        frame_val = np.zeros(shape = len(arr), dtype = 'object')
        for frame in tqdm(range(len(arr))):
            z = z + 1
            comparison_x = arr[frame][0:1] == x_inv
            comparison_y = arr[frame][0:1] == y_inv

            if comparison_x.all() == True:
                if check_orientation(df_in, z, extra_coords) == 'oriented':
                    frame_val[z] = 'X Investigation'
                elif check_orientation(df_in, z, extra_coords) == 'not_oriented':
                    frame_val[z] = 'X Close'
            elif comparison_y.all() == True:
                if check_orientation(df_in, z, extra_coords) == 'oriented':
                    frame_val[z] = 'Y Investigation'
                elif check_orientation(df_in, z, extra_coords) == 'not_oriented':
                    frame_val[z] = 'Y Close'
            else:
                frame_val[z] = 'Somewhere else'
                
        print(np.unique(frame_val, return_counts = True))
        

        x_invest = list(frame_val).count('X Investigation') 
        x_close = list(frame_val).count('X Close')  
        y_invest = list(frame_val).count('Y Investigation') 
        y_close = list(frame_val).count('Y Close')  
        somewhere_else = list(frame_val).count('Somewhere else')  
                
        final_frame_counts = [somewhere_else, x_invest, y_invest, x_close, y_close]

        final_dict[df_times['VideoName'][i]] = final_frame_counts


        global output_df
        output_df = pd.DataFrame(final_dict, index = ['Somewhere else','X Investigation', 'Y Investigation', 'X Close', 'Y Close']).T
        
        output_df['type'] = ["all"] * len(output_df)
        ###stop cutting
        output_df.reset_index(inplace = True)
        output_df.set_index(['index', 'type'], inplace = True)

        print('Just finished Video ' + str(i + 1) + ' of ' + str(len(df_times)))
        #makes a new folder and saves frame_vals
        frame_filepath = os.path.join(output_dir, "all" + "_" + df_times['VideoName'][i]) + "_frame_val.csv")
        pd.DataFrame(frame_val).to_csv(frame_filepath)

