import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import math


def remove_array_from_list(lst, arr):
  """
  Removes first instance of a numpy array encountered in a list.
  If array not encountered, this function does not do anything.
  """
  ind, size = 0, len(lst)
  while ind != size and not np.array_equal(lst[ind], arr):
    ind += 1
  if ind != size:
    lst.pop(ind)



def newline(p1, p2, segment = True, linestyle = '-', color='cyan', linewidth=7):
  """
  Plots a new line or line segment from points p1 to p2.
  This helper function is a modification of the one from Problem set 2.
  """
  ax = plt.gca()
  xmin, xmax = ax.get_xbound()

  if (p2[0] == p1[0]):
      xmin = xmax = p1[0]
      ymin, ymax = ax.get_ybound()
  else:
      ymax = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmax-p1[0])
      ymin = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmin-p1[0])
    
  if segment:
      l = mlines.Line2D([p1[0], p2[0]], [p1[1], p2[1]], linestyle=linestyle,\
                         color=color, linewidth=linewidth)
  else:
      l = mlines.Line2D([xmin, xmax], [ymin, ymax], linestyle=linestyle,\
                         color=color, linewidth=linewidth)

  ax.add_line(l)
  return l



def rotate_and_scale(origin, point, angle = 0, scale = 1):
    """
    Rotate a point clockwise by a given angle (in degrees) around a given origin,
    and scale it by a given factor.
    """
    theta = math.radians(angle)
    ox, oy = origin[0], origin[1]
    px, py = point[0], point[1]

    qx = ox + np.cos(-theta)*(px-ox) - np.sin(-theta)*(py-oy)
    qy = oy + np.sin(-theta)*(px-ox) + np.cos(-theta)*(py-oy)
    return (ox + scale*(qx-ox), oy + scale*(qy-oy))



def plot_and_print_results(image, box_corner_list):
  """
  Plots image along with the bounding boxes, accounting for the spill-over as well.
  Also prints the number of instances detected, as well as the vertices of the
  bounding boxes (in anti-clockwise order).
  """
  xmin = np.amin([0, np.amin(box_corner_list, axis = (0,1))[0]])
  xmax = np.amax([image.shape[1], np.amax(box_corner_list, axis = (0,1))[0]])
  ymin = np.amin([0, np.amin(box_corner_list, axis = (0,1))[1]])
  ymax = np.amax([image.shape[0], np.amax(box_corner_list, axis = (0,1))[1]])

  plt.xlim([xmin, xmax])
  plt.ylim([ymax, ymin])
  i = 1

  print("Number of instances found: %d\n" % (len(box_corner_list)))
  for detection in box_corner_list:
    print("Bounding box number %d:  %s" % (i, str(tuple([tuple(point) for point in detection.tolist()]))))
    i = i+1
    newline(detection[0], detection[1])
    newline(detection[1], detection[2])
    newline(detection[2], detection[3])
    newline(detection[3], detection[0])

  plt.imshow(image)
  plt.show()