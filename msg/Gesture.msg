# Message type inspired in https://github.com/ros4hri/hri_msgs/blob/master/msg/Gesture.msg
# Describes body language/attitude/gesture detected from a body.
Header header

# Additional gestures might be added in the future, please open
# issues/pull requests to suggest new ones.

uint8 UNKNOWN = 0

uint8 LEFT_ARM_UP = 1
uint8 LEFT_HAND_FRONT = 2
uint8 LEFT_ARM_SIDEWAYS = 3
uint8 LEFT_FOREARM_SIDEWAYS = 4
uint8 RIGHT_ARM_UP = 5
uint8 RIGHT_HAND_FRONT = 6
uint8 RIGHT_ARM_SIDEWAYS = 7
uint8 RIGHT_FOREARM_SIDEWAYS = 8
uint8 BOTH_ARMS_UP = 9
uint8 BOTH_HANDS_FRONT = 10

uint8 ARMS_CROSSED = 11
uint8 HANDS_ON_FACE = 12

string[] ids
uint8[] gesture # one of the above constants
float32[] gesture_confidence # confidence level (between 0. and 1.)
