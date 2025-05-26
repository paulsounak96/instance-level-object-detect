import numpy as np
import cv2
from utils import *


def create_database(image, descriptor_method = 'sift'):
  """
  Creates database for query lookup, given an image, i.e. returns
  corresponding keypoints and descriptors.
  """
  gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

  if descriptor_method == 'sift':
    desc_type = cv2.SIFT_create()
  elif descriptor_method == 'surf':
    desc_type = cv2.SURF_create()
  elif descriptor_method == 'orb':
    desc_type = cv2.ORB_create()
  else:
    raise ValueError("Unknown descriptor entered. Please enter sift, surf, or orb.")

  return desc_type.detectAndCompute(gray, None)



def database_lookup(lookup_keypoint, lookup_descriptor, train_keypoints,\
                     train_descriptors, train_shape, ratio_test_coeff = 0.8):
  """
  Enables lookup of SIFT descriptors with created database, and returns parameters
  (i.e. coordinates of bottom left and top right corners) of bounding rectangle.
  """
  bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck = False)

  # Perform the matching between the SIFT descriptors of the training image and the test image
  matches = bf.knnMatch(lookup_descriptor, train_descriptors, 2)

  for m, n in matches:
    if m.distance < ratio_test_coeff*n.distance:
      origin = np.array(lookup_keypoint[0].pt)
      point_c1 = origin + np.array([0, train_shape[0]]) - np.array(train_keypoints[m.trainIdx].pt)
      point_c2 = origin + np.array([train_shape[1], 0]) - np.array(train_keypoints[m.trainIdx].pt)
      angle = lookup_keypoint[0].angle - train_keypoints[m.trainIdx].angle
      scale = lookup_keypoint[0].size/train_keypoints[m.trainIdx].size

      good =[tuple(origin), train_keypoints[m.trainIdx].pt,\
              rotate_and_scale(origin, point_c1, angle = angle, scale = scale),\
                           rotate_and_scale(origin, point_c2, angle = angle, scale = scale)]
    else:
      good = []
  
  return good



def hough_voting(matches_list, vote_threshold = 5, cluster_size = 25):
  """
  Runs a Hough Transformation clustering algorithm and groups elements of matches_list
  into clusters, on basis of their 4d distance.
  """
  final_instances = []
  for i in range(len(matches_list)):
    current_instances = []
    current_instances.append(matches_list[i])
    fixed = matches_list[i]

    if np.sum([fixed in _ for _ in final_instances]) > 0:
      continue

    for j in range(len(matches_list)):
      if j == i:
        continue
      compare = matches_list[j]
      if np.linalg.norm(np.array(fixed[2]) - np.array(compare[2])) < cluster_size and\
                 np.linalg.norm(np.array(fixed[3]) - np.array(compare[3])) < cluster_size:
        current_instances.append(matches_list[j])

    if len(current_instances) >= vote_threshold:
      final_instances.append(current_instances)

  return final_instances



def preliminary_boundingbox_corners(test_image, training_image, vote_threshold=5, cluster_size=25):
  """
  Deploys Hough Transformation based clustering to find correct clusters corresponding
  to a valid instance detection. Returns list of corners for all the detections (possibly
  duplicates) of the object of interest.
  """
  train_keypoints, train_descriptors = create_database(training_image, descriptor_method = 'sift')
  test_keypoints, test_descriptors = create_database(test_image, descriptor_method = 'sift')

  # Create a Brute Force Matcher object.
  bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck = False)
  database_results = []

  for i in range(len(test_keypoints)):
    info = database_lookup(test_keypoints[i:i+1], test_descriptors[i:i+1],\
                            train_keypoints, train_descriptors, training_image.shape, 0.8)
    if len(info) > 0:
      database_results.append(info)

  hough_clusters = hough_voting(database_results,\
                                 vote_threshold=vote_threshold, cluster_size=cluster_size)
  warped_corner_list = []

  for cluster in hough_clusters:
    pts_train = np.float32([point[1] for point in cluster])
    pts_test = np.float32([point[0] for point in cluster])
    affine_matrix, _ = cv2.estimateAffinePartial2D(pts_train, pts_test,\
                                                    method = cv2.RANSAC, ransacReprojThreshold = 3)

    r, c = training_image.shape[:2]
    corners = np.array([[0, 0], [0, r], [c, r], [c, 0]])
    warped_corners = np.round(np.hstack([corners,\
                                          np.array([1,1,1,1])[:, np.newaxis]]) @ affine_matrix.T).astype(int)

    warped_corner_list.append(warped_corners)

  return warped_corner_list



def duplicate_box_remove(warped_corner_list, duplicate_thres):
  """
  Removes duplicate boxes in case hough_voting resulted in multiple detections
  of the same stop sign.
  """
  final_warped_corner_list = []
  while len(warped_corner_list) > 0:
    fixed_cluster = []
    fixed_corner = warped_corner_list[0]

    for compare_corner in warped_corner_list:
      if np.linalg.norm(np.array(fixed_corner) - np.array(compare_corner), ord = 'fro') < duplicate_thres:
        fixed_cluster.append(compare_corner)

    for corner in fixed_cluster:
      remove_array_from_list(warped_corner_list, corner)

    final_warped_corner_list.append(np.mean(fixed_cluster, axis=0).astype(int))
  return final_warped_corner_list



def instance_level_object_detector(test_path, train_path, vote_threshold=5, sensitivity=10):
  """
  Main function which:
  1. Reads the reference and test images from the given paths
  2. Detects all instances (possibly duplicates) of reference object in the test image,
      by calling `preliminary_boundingbox_corners`
  3. Removes the duplicates, by calling `duplicate_box_remove`
  4. Prints the number of instances along with the coordinates of the corresponding
      bounding boxes, as well as diplay the test image and the bounding boxes enclosing
      the detected instances, by calling `plot_and_print_results`
  """
  training_image = cv2.imread(train_path)
  training_image = cv2.cvtColor(training_image, cv2.COLOR_BGR2RGB)
  test_image = cv2.imread(test_path)
  test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
  
  cluster_size = sensitivity * np.amin(test_image.shape[:2]) / 100.0
  warped_corner_list = preliminary_boundingbox_corners(test_image, training_image,\
                                                       vote_threshold=vote_threshold,\
                                                          cluster_size=cluster_size)

  final_warped_corner_list = duplicate_box_remove(warped_corner_list, 4*cluster_size)
  plot_and_print_results(test_image, final_warped_corner_list)