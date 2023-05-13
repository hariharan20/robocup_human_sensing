# Message type inspired in https://github.com/ros4hri/hri_msgs/blob/master/msg/BodyPosture.msg
# Describes the general body posture in a symbolic manner.
Header header

#General option
uint8 UNKNOWN = 0

#Posture options
uint8 STANDING = 1
uint8 SITTING = 2

#Orientation options
uint8 FRONT = 1 # facing the camera
uint8 BACK = 2 # giving the back
uint8 LEFT = 3 # showing the left side
uint8 RIGHT = 4 # showing the right side

string[] ids
uint8[] orientation # one of the above constants
float32[] orientation_confidence # confidence level (between 0. and 1.)
uint8[] posture # one of the above constants
float32[] posture_confidence # confidence level (between 0. and 1.)
