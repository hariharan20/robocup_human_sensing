#! /usr/bin/env python3
import rospy
from person_detection.msg import RegionOfInterest2D 
from darknet_ros_msgs.msg import BoundingBoxes

rospy.init_node('human_boxes')
pub = rospy.Publisher('h_boxes' , RegionOfInterest2D  ,queue_size = 10)

def callback(data):
	msg = RegionOfInterest2D()
	for i in range(len(data.bounding_boxes)):
		if(data.bounding_boxes[i].Class == 'person'):
			msg.ids += str(i) 
			msg.x.append( data.bounding_boxes[i].xmin)
			msg.y.append( data.bounding_boxes[i].ymin)
			msg.w.append( data.bounding_boxes[i].xmax - data.bounding_boxes[i].xmin)
			msg.h.append( data.bounding_boxes[i].ymax - data.bounding_boxes[i].ymin)
	rospy.loginfo(msg)
	pub.publish(msg)


rospy.Subscriber('/darknet_ros/bounding_boxes' , BoundingBoxes , callback) 
rospy.spin()
