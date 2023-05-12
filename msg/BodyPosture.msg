# Message type inspired in https://github.com/ros4hri/hri_msgs/blob/master/msg/BodyPosture.msg
# Describes the general body posture in a symbolic manner.
Header header

uint8 STANDING = 1
uint8 SITTING = 2
uint8 CROUCHING = 3
uint8 LAYING = 4

uint8 OTHER = 0

uint8 posture # one of the above constants
float32 c # c is a confidence level (between 0. and 1.)
