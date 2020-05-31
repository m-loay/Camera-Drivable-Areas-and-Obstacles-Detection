'''
Created on Aug 5, 2019

@author: modys
'''
from m6bk import *
from helperfunctions import *

dataset_handler = DatasetHandler()
dataset_handler.set_frame(1)
segmentation = dataset_handler.segmentation
detections = dataset_handler.object_detection
z = dataset_handler.depth

# Part 1
k = dataset_handler.k
x, y = xy_from_depth(z, k)
road_mask = np.zeros(segmentation.shape)
road_mask[segmentation == 7] = 1
x_ground = x[road_mask == 1]
y_ground = y[road_mask == 1]
z_ground = dataset_handler.depth[road_mask == 1]
xyz_ground = np.stack((x_ground, y_ground, z_ground))
p_final = ransac_plane_fit(xyz_ground)

# Part II
lane_lines = estimate_lane_lines(segmentation)
merged_lane_lines = merge_lane_lines(lane_lines,dataset_handler.image.shape[0],road_mask)
max_y = dataset_handler.image.shape[0]
min_y = np.min(np.argwhere(road_mask == 1)[:, 0])

extrapolated_lanes = extrapolate_lines(merged_lane_lines, max_y, min_y)
final_lanes = find_closest_lines(extrapolated_lanes, dataset_handler.lane_midpoint)

# Part III
filtered_detections = filter_detections_by_segmentation(detections, segmentation)
min_distances = find_min_distance_to_detection(filtered_detections, x, y, z)

# Print Submission Info

final_lane_printed = [list(np.round(lane)) for lane in final_lanes]
print('plane:') 
print(list(np.round(p_final, 2)))
print('\n lanes:')
print(final_lane_printed)
print('\n min_distance')
print(list(np.round(min_distances, 2)))


# Original Image
plt.imshow(dataset_handler.image)

# Part I
dist = np.abs(dist_to_plane(p_final, x, y, z))

ground_mask = np.zeros(dist.shape)

ground_mask[dist < 0.1] = 1
ground_mask[dist > 0.1] = 0

plt.imshow(ground_mask)
plt.show()

# Part II
plt.imshow(dataset_handler.vis_lanes(final_lanes))
plt.show()

# Part III
font = {'family': 'serif','color': 'red','weight': 'normal','size': 12}

im_out = dataset_handler.vis_object_detection(filtered_detections)

for detection, min_distance in zip(filtered_detections, min_distances):
    bounding_box = np.asfarray(detection[1:5])
    plt.text(bounding_box[0], bounding_box[1] - 20, 'Distance to Impact:' + str(np.round(min_distance, 2)) + ' m', fontdict=font)
    
plt.imshow(im_out)
plt.show()    