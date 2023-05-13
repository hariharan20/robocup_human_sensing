# Message type inspired in https://github.com/ros4hri/hri_msgs/blob/master/msg/SoftBiometrics.msg
# This message describes soft biometrics (age and gender)
Header header

uint8 UNDEFINED=0
uint8 FEMALE=1
uint8 MALE=2
uint8 OTHER=3

string[] ids
uint8[] age
float32[] age_confidence
uint8[] gender # one of UNDEFINED, FEMALE, MALE, OTHER
float32[] gender_confidence
