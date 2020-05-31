'''
Created on Aug 5, 2019

@author: modys
'''
from m6bk import *
from helperfunctions import *

'''
0 - Loading and Visualizing the Data
'''
test_res = False
dataset_handler = DatasetHandler()
dataset_handler.current_frame
image = dataset_handler.image
print("Show Original Image")
plt.imshow(image)
# plt.show() 

k = dataset_handler.k
print("Calibration Matrix")
print(k)

print("Show depth Image")
depth = dataset_handler.depth
plt.imshow(depth, cmap='jet')
# plt.show()

print("Show Segmentation Output")
segmentation = dataset_handler.segmentation
plt.imshow(segmentation)
# plt.show()

print("Show Coloured Segmentation Output")
colored_segmentation = dataset_handler.vis_segmentation(segmentation)
plt.imshow(colored_segmentation)
# plt.show()

'''
1 - Drivable Space Estimation Using Semantic Segmentation Output
'''
'''
1.1 - Estimating the x, y, and z coordinates of every pixel in the image
'''
dataset_handler.set_frame(0)
k = dataset_handler.k
z = dataset_handler.depth
x, y = xy_from_depth(z, k)

test_res =  (x[800,800] == 0.720475)
test_res &= (y[800,800] == 1.436475)
test_res &= (z[800,800] == 2.864)
test_res &= (x[500,500] == -9.5742765625)
test_res &= (y[500,500] == 1.4464734375)
test_res &= (z[500,500] == 44.083)

if(test_res == True):
    print("Test 1 passed")
else:
    print("Test 1 Failed")    
    
'''
1.2 - Estimating The Ground Plane Using RANSAC
'''
# Get road mask by choosing pixels in segmentation output with value 7
test_res = False
test_tol = 0.01
road_mask = np.zeros(segmentation.shape)
road_mask[segmentation == 7] = 1

# Show road mask
plt.imshow(road_mask)
# plt.show()

# Get x,y, and z coordinates of pixels in road mask
x_ground = x[road_mask == 1]
y_ground = y[road_mask == 1]
z_ground = dataset_handler.depth[road_mask == 1]
xyz_ground = np.stack((x_ground, y_ground, z_ground))
p_final = ransac_plane_fit(xyz_ground)
p = [0.01791606, -0.99981332,0.00723433, 1.40281479]

test_res  = (p[0]+test_tol > p_final[0] and p[0]-test_tol < p_final[0] )
test_res &= (p[1]+test_tol > p_final[1] and p[1]-test_tol < p_final[1] )
test_res &= (p[2]+test_tol > p_final[2] and p[2]-test_tol < p_final[2] )
test_res &= (p[3]+test_tol > p_final[3] and p[3]-test_tol < p_final[3] )

if(test_res == True):
    print("Test 2 passed")
else:
    print("Test 2 Failed") 
    
'''
2 - Lane Estimation Using The Semantic Segmentation Output
'''
'''
2.1 Estimating Lane Boundary Proposals
'''

lane_lines = estimate_lane_lines(segmentation)
# print(lane_lines.shape)
plt.imshow(dataset_handler.vis_lanes(lane_lines))
# plt.show()

'''
2.2 - Merging and Filtering Lane Lines
'''
lane_lines = estimate_lane_lines(segmentation)
merged_lane_lines = merge_lane_lines(lane_lines,dataset_handler.image.shape[0],road_mask)
plt.imshow(dataset_handler.vis_lanes(merged_lane_lines))
# plt.show()

max_y = dataset_handler.image.shape[0]
min_y = np.min(np.argwhere(road_mask == 1)[:, 0])

extrapolated_lanes = extrapolate_lines(merged_lane_lines, max_y, min_y)
final_lanes = find_closest_lines(extrapolated_lanes, dataset_handler.lane_midpoint)
plt.imshow(dataset_handler.vis_lanes(final_lanes))
# plt.show()

'''
3 - Computing Minimum Distance To Impact Using The Output of 2D Object Detection
'''

detections = dataset_handler.object_detection
plt.imshow(dataset_handler.vis_object_detection(detections))
detection = detections[2]
image = segmentation[int(np.asfarray(detection[2])): int(np.asfarray(detection[4])),
                     int(np.asfarray(detection[1])): int(np.asfarray(detection[3]))]
image = np.array(image)
pcount = np.zeros(image.shape)
pcount[image == 10] = 1
plt.imshow(pcount)
# plt.show()
plt.imshow(segmentation)
# plt.show()

'''
3.1 - Filtering Out Unreliable Detections
'''
filtered_detections = filter_detections_by_segmentation(detections, segmentation)
plt.imshow(dataset_handler.vis_object_detection(filtered_detections))
# plt.show()

'''
3.2 - Estimating Minimum Distance To Impact
'''
test_res = False
f_detections = np.array(filtered_detections)
min_distances = find_min_distance_to_detection(f_detections, x, y, z)
test_res = (min_distances[0] == 8.511952208248577)
if(test_res == True):
    print("Test 3 passed")
else:
    print("Test 3 Failed") 