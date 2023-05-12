# Message type inspired in https://github.com/ros4hri/hri_msgs/blob/master/msg/IdsList.msg
# This message encodes a list of ROS4HRI IDs (eg face ids, body ids, person
# ids...).
# It is for instance published on /humans/faces/tracked to access the list of
# currently detected faces.
Header header
string[] ids
