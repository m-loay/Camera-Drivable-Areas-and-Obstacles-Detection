'''
Created on Nov 18, 2019

@author: mloay
'''
from m6bk import *
#################################################################################################################

'''
1 - Drivable Space Estimation Using Semantic Segmentation Output
'''

'''
1.1 Estimating the x, y, and z coordinates of every pixel in the image
'''
def xy_from_depth(depth, k):
    """
    Computes the x, and y coordinates of every pixel in the image using the depth map and the calibration matrix.

    Arguments:
    depth -- tensor of dimension (H, W), contains a depth value (in meters) for every pixel in the image.
    k -- tensor of dimension (3x3), the intrinsic camera matrix

    Returns:
    x -- tensor of dimension (H, W) containing the x coordinates of every pixel in the camera coordinate frame.
    y -- tensor of dimension (H, W) containing the y coordinates of every pixel in the camera coordinate frame.
    """
    ### START CODE HERE ###

    # Get the shape of the depth tensor
    H,W = depth.shape

    # Grab required parameters from the K matrix
    f   = k[0,0]
    c_u = k[0,2]
    c_v = k[1,2]
    
    # Generate a grid of coordinates corresponding to the shape of the depth map
    xa = np.arange(1, W+1, 1)
    ya = np.arange(1, H+1, 1)
    u, v = np.meshgrid(xa, ya)
    
    # Compute x and y coordinates
    x = (u - c_u)*depth/f
    y = (v - c_v)*depth/f

    ### END CODE HERE ###
    
    return x, y

'''
1.2 - Estimating The Ground Plane Using RANSAC
'''
def ransac_plane_fit(xyz_data):
    """
    Computes plane coefficients a,b,c,d of the plane in the form ax+by+cz+d = 0
    using ransac for outlier rejection.

    Arguments:
    xyz_data -- tensor of dimension (3, N), contains all data points from which random sampling will proceed.
    num_itr -- 
    distance_threshold -- Distance threshold from plane for a point to be considered an inlier.

    Returns:
    p -- tensor of dimension (1, 4) containing the plane parameters a,b,c,d
    """
    
    ### START CODE HERE ### 
    
    # Set thresholds:
    num_itr = 1000  # RANSAC maximum number of iterations
    min_num_inliers = 200000 # RANSAC minimum number of inliers
    distance_threshold = 0.00001  # Maximum distance from point to plane for point to be considered inlier
    N = xyz_data.shape[1]
    inlineC_temp = []
    CCC_temp = np.zeros((3,4))
    
    for i in range(num_itr):
        # Step 1: Choose a minimum of 3 points from xyz_data at random.
        ac = np.random.choice(N,6)
        CCC = (np.vstack((xyz_data.T[ac[0]],xyz_data.T[ac[1]],xyz_data.T[ac[2]],xyz_data.T[ac[3]],xyz_data.T[ac[4]],xyz_data.T[ac[5]]))).T        
        
        # Step 2: Compute plane model
        p = compute_plane(CCC)
        
        # Step 3: Find number of inliers
        xx=[]
        yy=[]
        zz=[]
        
        xx = np.array(xyz_data[0,0:]).T
        yy = np.array(xyz_data[1,0:]).T
        zz = np.array(xyz_data[2,0:]).T
        dis = dist_to_plane(p,xx,yy,zz)
        
        # Step 4: Check if the current number of inliers is greater than all previous iterations
        # and keep the inlier set with the largest number of points.
        inlinec = np.zeros(dis.shape)
        inlinec[dis < distance_threshold] = 1
        if sum(inlinec) > sum(inlineC_temp):
            CCC_temp = CCC
        
        # Step 5: Check if stopping criterion is satisfied and break.         
        if sum(inlinec) >= min_num_inliers:
            return p
        
    # Step 6: Recompute the model parameters using largest inlier set.         
    p = compute_plane(CCC_temp)
    ### END CODE HERE ###
    output_plane = p
    
    return output_plane 


'''
2 - Lane Estimation Using The Semantic Segmentation Output
'''

'''
2.1 Estimating Lane Boundary Proposals
'''
def estimate_lane_lines(segmentation_output):
    """
    Estimates lines belonging to lane boundaries. Multiple lines could correspond to a single lane.

    Arguments:
    segmentation_output -- tensor of dimension (H,W), containing semantic segmentation neural network output
    minLineLength -- Scalar, the minimum line length
    maxLineGap -- Scalar, dimension (Nx1), containing the z coordinates of the points

    Returns:
    lines -- tensor of dimension (N, 4) containing lines in the form of [x_1, y_1, x_2, y_2], where [x_1,y_1] and [x_2,y_2] are
    the coordinates of two points on the line in the (u,v) image coordinate frame.
    """
    ### START CODE HERE ###
    # Step 1: Create an image with pixels belonging to lane boundary categories from the output of semantic segmentation
    rmask = np.zeros(segmentation_output.shape)
    rmask[segmentation_output == 6] = 1
    rmask[segmentation_output == 8] = 1

    # Step 2: Perform Edge Detection using cv2.Canny()
    blur = cv2.Sobel(rmask,cv2.CV_64F,1,0,ksize=5)

    # Step 3: Perform Line estimation using cv2.HoughLinesP()
    blur = np.uint8(blur)
    lines = cv2.HoughLinesP(blur, rho=1, theta=np.pi / 180, threshold=400, minLineLength=100, maxLineGap=10)
    lines = lines.reshape((lines.shape[0],lines.shape[2]))
    
    # Note: Make sure dimensions of returned lines is (N x 4)
    ### END CODE HERE ###

    return lines

'''
2.2 - Merging and Filtering Lane Lines
'''
def merge_lane_lines(lines,image_size,road_mask):
    """
    Merges lane lines to output a single line per lane, using the slope and intercept as similarity measures.
    Also, filters horizontal lane lines based on a minimum slope threshold.

    Arguments:
    lines -- tensor of dimension (N, 4) containing lines in the form of [x_1, y_1, x_2, y_2],
    the coordinates of two points on the line.

    Returns:
    merged_lines -- tensor of dimension (N, 4) containing lines in the form of [x_1, y_1, x_2, y_2],
    the coordinates of two points on the line.
    """
    
    ### START CODE HERE ###
    
    # Step 0: Define thresholds
    slope_similarity_threshold = 0.1
    intercept_similarity_threshold = 40
    min_slope_threshold = 0.3
    
    # Step 1: Get slope and intercept of lines
    slopes, intercepts = get_slope_intecept(lines)
    
    # Step 2: Determine lines with slope less than horizontal slope threshold.
    slope_list = []
    intercept_list = []
    final_list_slope = []
    final_list_intercept = []
    
    # Step 3: Iterate over all remaining slopes and intercepts and cluster lines that are close to each other using a slope and intercept threshold.
    for slope, intercept in zip(slopes, intercepts):
        if len(slope_list) == 0:
            slope_list.append(slope)
            intercept_list.append(intercept)
        else:
            for i in range(len(slope_list)):
                if np.absolute(slope - slope_list[i]) >= slope_similarity_threshold and np.absolute(intercept - intercept_list[i]) >= intercept_similarity_threshold:
                    final_list_slope.append(slope)
                    final_list_intercept.append(intercept)
                
                else:
                    slope_list.append(slope)
                    intercept_list.append(slope)
        

    # Step 4: Merge all lines in clusters using mean averaging
    final_list_slope.append(slope_list[0]) 
    final_list_intercept.append(intercept_list[0])
    new_lines = []
    max_y = image_size 
    min_y = np.min(np.argwhere(road_mask == 1)[:, 0])
    
    for slope, intercept, in zip(final_list_slope, final_list_intercept):
        x1 = (min_y - intercept) / slope
        x2 = (max_y - intercept) / slope
        new_lines.append([x1, min_y, x2, max_y])

    return np.array(new_lines)
    
    # Note: Make sure dimensions of returned lines is (N x 4)
    ### END CODE HERE ###
    return new_lines

'''
3 - Computing Minimum Distance To Impact Using The Output of 2D Object Detection
'''

'''
3.1 - Filtering Out Unreliable Detections
'''
def filter_detections_by_segmentation(detections, segmentation_output):
    """
    Filter 2D detection output based on a semantic segmentation map.

    Arguments:
    detections -- tensor of dimension (N, 5) containing detections in the form of [Class, x_min, y_min, x_max, y_max, score].
    
    segmentation_output -- tensor of dimension (HxW) containing pixel category labels.
    
    Returns:
    filtered_detections -- tensor of dimension (N, 5) containing detections in the form of [Class, x_min, y_min, x_max, y_max, score].

    """
    ### START CODE HERE ###
    
    # Set ratio threshold:
    ratio_threshold = 0.3  # If 1/3 of the total pixels belong to the target category, the detection is correct.
    filtered_detections = []
    for detection in detections:
        
        # Step 1: Compute number of pixels belonging to the category for every detection.
        image = segmentation_output[int(np.asfarray(detection[2])): int(np.asfarray(detection[4])),
                             int(np.asfarray(detection[1])): int(np.asfarray(detection[3]))]
        image = np.array(image)
        pcount = np.zeros(image.shape)
        pcount[image == 10] = 1
        s = sum(sum(pcount))
        a = image.shape[0]*image.shape[1]
        # Step 2: Devide the computed number of pixels by the area of the bounding box (total number of pixels).
        r = s/a
            
        # Step 3: If the ratio is greater than a threshold keep the detection. Else, remove the detection from the list of detections.
        if r >= ratio_threshold:
            filtered_detections.append(detection)
    ### END CODE HERE ###
    #filtered_detections = np.array(filtered_detections)
    return filtered_detections

'''
3.2 - Estimating Minimum Distance To Impact
'''
def find_min_distance_to_detection(detections, x, y, z):
    """
    Filter 2D detection output based on a semantic segmentation map.

    Arguments:
    detections -- tensor of dimension (N, 5) containing detections in the form of [Class, x_min, y_min, x_max, y_max, score].
    
    x -- tensor of dimension (H, W) containing the x coordinates of every pixel in the camera coordinate frame.
    y -- tensor of dimension (H, W) containing the y coordinates of every pixel in the camera coordinate frame.
    z -- tensor of dimensions (H,W) containing the z coordinates of every pixel in the camera coordinate frame.
    Returns:
    min_distances -- tensor of dimension (N, 1) containing distance to impact with every object in the scene.

    """
    ### START CODE HERE ###
    min_distances = []
    for detection in detections:
        # Step 1: Compute distance of every pixel in the detection bounds
        dx = x[int(np.asfarray(detection[2])): int(np.asfarray(detection[4])),
                     int(np.asfarray(detection[1])): int(np.asfarray(detection[3]))]
        dy = y[int(np.asfarray(detection[2])): int(np.asfarray(detection[4])),
                     int(np.asfarray(detection[1])): int(np.asfarray(detection[3]))]
        dz = z[int(np.asfarray(detection[2])): int(np.asfarray(detection[4])),
                     int(np.asfarray(detection[1])): int(np.asfarray(detection[3]))]
        
        dis = np.sqrt(dx**2 + dy**2 + dz**2)        
        # Step 2: Find minimum distance
        min_distances.append(dis.min())

    ### END CODE HERE ###
    return min_distances

#################################################################################################################