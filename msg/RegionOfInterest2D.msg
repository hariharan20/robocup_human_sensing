# Message type inspired in https://github.com/ros4hri/hri_msgs/blob/master/msg/NormalizedRegionOfInterest2D.msg
# This contains the top-leftmost coordinates and size of a region of interest (typically in an image)
# The (x, y) tuple stores the top-leftmost coordinates of the ROI, while (w, h) represents the
# width and heightof the ROI.
# c is a confidence level (between 0. and 1.) associated to that ROI
Header header # Header timestamp should be acquisition time of the original image
string[] ids
int32[] x
int32[] y
int32[] w
int32[] h
float32[] c
