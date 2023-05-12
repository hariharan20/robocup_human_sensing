#! /usr/bin/python3

#required packages
import rospy #tf
import message_filters #to sync the messages
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
from tf.transformations import euler_from_quaternion
import sys
from math import * #to avoid prefix math.
import numpy as np #to use matrix
from numpy import linalg as LA
import time
import cv2
import yaml
from cv_bridge import CvBridge
import joblib
from robocup_human_sensing.msg import IdsList, NormalizedRegionOfInterest2D, BodyPosture, Gesture

##########################################################################################
#GENERAL PURPUSES VARIABLES
pub_hz=0.01 #main loop frequency
bridge = CvBridge() #initializing cvbrigde
#GLOBAL CONFIG FILE DIRECTORY
config_direct=rospy.get_param("/hs_gesture_estimation/config_direct") 
a_yaml_file = open(config_direct+"global_config.yaml")
parsed_yaml_file = yaml.load(a_yaml_file, Loader=yaml.FullLoader)
##SKELETON EXTRACTION USING OPENPOSE
gesture_classifier_model=rospy.get_param("/hs_gesture_estimation/gesture_classifier_model") 
model_rf = joblib.load(gesture_classifier_model)   
selected_joints=parsed_yaml_file.get("openpose_config").get("skeleton_extract_param")
n_joints=len(selected_joints)
n_features=parsed_yaml_file.get("openpose_config").get("n_features")
joints_min=parsed_yaml_file.get("openpose_config").get("joints_min")
performance_default="normal" #OpenPose performance, can be "normal" or "high", "normal" as initial condition
##OPENPOSE INITIALIZATION 
openpose_python=parsed_yaml_file.get("directories_config").get("open_pose_python") 
openpose_models=parsed_yaml_file.get("directories_config").get("open_pose_models") 
try:
    sys.path.append(openpose_python);
    from openpose import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e
params = dict()
params["model_folder"] = openpose_models
if performance_default=="normal":
    net_resolution= parsed_yaml_file.get("openpose_config").get("openpose_normal_performance") # has to be numbers multiple of 16, the default is "-1x368", and the fastest performance is "-1x160"
else: #high performance
    net_resolution= parsed_yaml_file.get("openpose_config").get("openpose_high_performance")  #High performance is "-1x480"
params["net_resolution"] = net_resolution 
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()
datum = op.Datum()
#ROS PUBLISHER SET UP
pub_img = rospy.Publisher('openpose_output', Image,queue_size=1)
msg_img = Image()       
pub_gesture = rospy.Publisher('people_gestures', Gesture,queue_size=1)
msg_gesture= Gesture()
pub_posture = rospy.Publisher('people_postures', BodyPosture,queue_size=1)
msg_posture= BodyPosture()
#RGBD CAMERA INTRINSIC,DISTORTION PARAMETERS
intr_param=parsed_yaml_file.get("camera_config").get("intr_param") #camera intrinsic parameters
#CAMERA_BASED DETECTION
camera_topics=parsed_yaml_file.get("camera_config").get("camera_topic_names") # a list with the names of the topics to be subscribed
#########################################################################################################################

class human_class:
    def __init__(self): #It is done only the first iteration
        #Variables to store temporally the info of human detectection at each time instant
        self.n_human=1 # considering 1 human detected initially
        self.ids=["id1"] #list with string corresponding to the IDs of the people detected
        self.roi=np.zeros([self.n_human,4]) #the top-leftmost and bottom-rightmost coordinates of a Region of Interest (ROI) normalized from 0 to 1
        self.gesture=np.zeros([self.n_human,2]) #Gesture label with the corresponding confidence level from 0 to 1
        self.posture=np.zeros([self.n_human,2]) #Body posture label with the corresponding confidence level from 0 to 1
        self.performance=performance_default #performance for openpose     
    
    def rgbd_callback(self,rgb, depth):
        if self.n_human>0:
            roi=self.roi
            ids=self.ids
            n_human=self.n_human
            gesture=np.zeros([n_human,2])
            posture=np.zeros([n_human,2])
            ##################################################################################33
            #Color image
            color_image = bridge.imgmsg_to_cv2(rgb,"bgr8")
            #Depth image
            depth_image = bridge.imgmsg_to_cv2(depth,"32FC1")
            depth_array= np.array(depth_image, dtype=np.float32)/1000
            image_size = color_image.shape
            for i in range(0,n_human):
                ##############################################
                #JUST FOR TESTING USE THE FULL IMAGE
                roi[i,0]=0
                roi[i,1]=0
                roi[i,2]=image_size[0]
                roi[i,3]=image_size[1]
                ##############################################
                color_crop = color_image.crop((roi[i,0], roi[i,1], roi[i,2], roi[i,3]))
                depth_crop = depth_array.crop((roi[i,0], roi[i,1], roi[i,2], roi[i,3]))
                [gesture[i,:],posture[i,:],image_show]=self.processing(color_crop,depth_crop)
            #Publish last OPENPOSE output as an image
            scaling=0.5
            openpose_image=cv2.resize(image_show,(int(image_show.shape[1]*scaling),int(image_show.shape[0]*scaling))) #resizing it 
            msg_img.header.stamp = rospy.Time.now()
            msg_img.height = openpose_image.shape[0]
            msg_img.width = openpose_image.shape[1]
            msg_img.encoding = "bgr8"
            msg_img.is_bigendian = False
            msg_img.step = 3 * openpose_image.shape[1]
            msg_img.data = np.array(openpose_image).tobytes()
            pub_img.publish(msg_img)
            #Publish gesture message
            
            #Publish body posture message

            #######################################################################################
    
    def roi_ids_callback(self,roi,ids):
        n_human=len(ids)
        roi=np.zeros([n_human,4]) 
        if n_human==0:
            self.performance="low"
            opWrapper.stop() # to stop openpose
        else: # there is at least a human detected
            for i in range(0,n_human):
                roi[i,0]=roi.xmin
                roi[i,1]=roi.ymin
                roi[i,2]=roi.xmax
                roi[i,3]=roi.ymax
        self.roi=roi
        self.n_human=len(ids)
        self.ids=ids
    
    def processing(self,color_image,depth_array):
        performance_past=self.performance                  
        self.performance=performance_default
        
        if self.performance=="high" and performance_past!=self.performance:
            params = dict()
            params["model_folder"] = openpose_models
            net_resolution= parsed_yaml_file.get("openpose_config").get("openpose_high_performance")
            params["net_resolution"] = net_resolution 
            opWrapper.stop()
            opWrapper.configure(params)
            opWrapper.start()

        elif self.performance=="normal" and performance_past!=self.performance:
            params = dict()
            params["model_folder"] = openpose_models
            net_resolution= parsed_yaml_file.get("openpose_config").get("openpose_normal_performance")
            params["net_resolution"] = net_resolution 
            opWrapper.stop()
            opWrapper.configure(params)
            opWrapper.start()

        #######################################################################################    
        datum.cvInputData = color_image
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))
        #Keypoints extraction using OpenPose
        keypoints=datum.poseKeypoints
        if keypoints is None: #if there is no human skeleton detected
            #print('No skeleton extracted')
            gesture=np.zeros([1,2]) 
            posture=np.zeros([1,2]) 
        else: 
            #Feature extraction
            [gesture,posture]=self.gesture_estimation(keypoints,depth_array)             
        image_show=datum.cvOutputData #for visualization             
        return gesture,posture,image_show
        #######################################################################################
        
        
    def gesture_estimation(self,poseKeypoints,depth_array):
        gesture=np.zeros([1,2])
        posture=np.zeros([1,2])
        features=np.zeros([1,n_features]) 
        index=0 #in case of multiple skeleton detection, condiser only the first skeleton in poseKeypoints
        #Using only the important joints
        joints_x_init=poseKeypoints[index,0:n_joints,0]
        joints_y_init=poseKeypoints[index,0:n_joints,1]   
        joints_z_init=[0]*n_joints   
        for k in range(0,n_joints):
            joints_z_init[k]=depth_array[int(joints_y_init[k]),int(joints_x_init[k])]
        #Normalization and scaling
        #Translation
        J_sum_x=0
        J_sum_y=0
        J_sum_z=0
        for k in range(0,n_joints):   
           J_sum_x=joints_x_init[k]+J_sum_x
           J_sum_x=joints_y_init[k]+J_sum_y
           J_sum_z=joints_z_init[k]+J_sum_z
        J_mean_x=J_sum_x/(n_joints) 
        J_mean_y=J_sum_y/(n_joints)
        J_mean_z=J_sum_z/(n_joints)
        joints_x_trans=joints_x_init-J_mean_x 
        joints_y_trans=joints_y_init-J_mean_y
        joints_z_trans=joints_z_init-J_mean_z
        #Normalization   
        J_sum2=0
        valid=0
        for k in range(0,n_joints):  
            J_sum2=joints_x_trans[k]**2+joints_y_trans[k]**2+joints_z_trans[k]**2+J_sum2  
            if joints_x_trans[k]!=0 and joints_y_trans[k]!=0 and joints_z_trans[k]!=0:
                valid=valid+1
        Js=sqrt(J_sum2/(n_joints))
        #only continue if there are enough joints detected on the skeleton
        if Js!=0 and valid>=joints_min: 
            joints_x = joints_x_trans/Js      
            joints_y = joints_y_trans/Js
            joints_z = joints_z_trans/Js              
            #Distances from each joint to the neck joint
            dist=[0]*n_joints
            for k in range(0,n_joints):
               dist[k]=sqrt((joints_x[k]-joints_x[1])**2+(joints_y[k]-joints_y[1])**2+(joints_z[k]-joints_z[1])**2)         
            #Vectors between joints
            v1_2=[joints_x[1]-joints_x[2], joints_y[1]-joints_y[2], joints_z[1]-joints_z[2]]  
            v2_3=[joints_x[2]-joints_x[3], joints_y[2]-joints_y[3], joints_z[2]-joints_z[3]]  
            v3_4=[joints_x[3]-joints_x[4], joints_y[3]-joints_y[4], joints_z[3]-joints_z[4]]  
            v1_5=[joints_x[1]-joints_x[5], joints_y[1]-joints_y[5], joints_z[1]-joints_z[5]]  
            v5_6=[joints_x[5]-joints_x[6], joints_y[5]-joints_y[6], joints_z[5]-joints_z[6]]  
            v6_7=[joints_x[6]-joints_x[7], joints_y[6]-joints_y[7], joints_z[6]-joints_z[7]]  
            v1_0=[joints_x[1]-joints_x[0], joints_y[1]-joints_y[0], joints_z[1]-joints_z[0]]  
            v0_15=[joints_x[0]-joints_x[15], joints_y[0]-joints_y[15], joints_z[0]-joints_z[15]]  
            v15_17=[joints_x[15]-joints_x[17], joints_y[15]-joints_y[17], joints_z[15]-joints_z[17]]  
            v0_16=[joints_x[0]-joints_x[16], joints_y[0]-joints_y[16], joints_z[0]-joints_z[16]]
            v16_18=[joints_x[16]-joints_x[18], joints_y[16]-joints_y[18], joints_z[16]-joints_z[18]]  
            v1_8=[joints_x[1]-joints_x[8], joints_y[1]-joints_y[8], joints_z[1]-joints_z[8]]
            v8_9=[joints_x[8]-joints_x[9], joints_y[8]-joints_y[9], joints_z[8]-joints_z[9]]  
            v9_10=[joints_x[9]-joints_x[10], joints_y[9]-joints_y[10], joints_z[9]-joints_z[10]]  
            v10_11=[joints_x[10]-joints_x[11], joints_y[10]-joints_y[11], joints_z[10]-joints_z[11]]  
            v8_12=[joints_x[8]-joints_x[12], joints_y[8]-joints_y[12], joints_z[8]-joints_z[12]]  
            v12_13=[joints_x[12]-joints_x[13], joints_y[12]-joints_y[13], joints_z[12]-joints_z[13]]  
            v13_14=[joints_x[13]-joints_x[14], joints_y[13]-joints_y[14], joints_z[13]-joints_z[14]]    
            #Angles between joints  
            angles=[0]*(n_joints-2) #17 angles
            angles[0] = atan2(LA.norm(np.cross(v15_17,v0_15)),np.dot(v15_17,v0_15))
            angles[1] = atan2(LA.norm(np.cross(v0_15,v1_0)),np.dot(v0_15,v1_0))
            angles[2] = atan2(LA.norm(np.cross(v16_18,v0_16)),np.dot(v16_18,v0_16))
            angles[3] = atan2(LA.norm(np.cross(v0_16,v1_0)),np.dot(v0_16,v1_0))
            angles[4] = atan2(LA.norm(np.cross(v1_0,v1_2)),np.dot(v1_0,v1_2))
            angles[5] = atan2(LA.norm(np.cross(v1_2,v2_3)),np.dot(v1_2,v2_3))
            angles[6] = atan2(LA.norm(np.cross(v2_3,v3_4)),np.dot(v2_3,v3_4))
            angles[7] = atan2(LA.norm(np.cross(v1_0,v1_5)),np.dot(v1_0,v1_5))
            angles[8] = atan2(LA.norm(np.cross(v1_5,v5_6)),np.dot(v1_5,v5_6))
            angles[9] = atan2(LA.norm(np.cross(v5_6,v6_7)),np.dot(v5_6,v6_7))
            angles[10] = atan2(LA.norm(np.cross(v1_2,v1_8)),np.dot(v1_2,v1_8))
            angles[11] = atan2(LA.norm(np.cross(v1_8,v8_9)),np.dot(v1_8,v8_9))
            angles[12] = atan2(LA.norm(np.cross(v8_9,v9_10)),np.dot(v8_9,v9_10))
            angles[13] = atan2(LA.norm(np.cross(v9_10,v10_11)),np.dot(v9_10,v10_11))
            angles[14] = atan2(LA.norm(np.cross(v1_8,v8_12)),np.dot(v1_8,v8_12))
            angles[15] = atan2(LA.norm(np.cross(v8_12,v12_13)),np.dot(v8_12,v12_13))
            angles[16] = atan2(LA.norm(np.cross(v12_13,v13_14)),np.dot(v12_13,v13_14))           
            #HUMAN FEATURES CALCULATION
            features=dist+angles  
            #HUMAN GESTURE RECOGNITION
            X=np.array(features).transpose()
            gesture[0]=model_rf.predict([X])
            prob_max=0
            prob=model_rf.predict_proba([X])
            for ii in range(0,prob.shape[1]): #depends of the number of gestures to classified
                if prob[0,ii]>=prob_max:
                    prob_max=prob[0,ii]
            gesture[1]=prob_max 
        else:
            gesture[0]=0 # No gesture
            gesture[1]=1 # 1 confidence
        ########################################################################
        #BODY POSTURE RECOGNITION
        #Orientation inference using nose, ears, eyes keypoints
        orientation=0 # "facing the robot" by default
        if poseKeypoints[index,0,0]!=0 and poseKeypoints[index,15,0]!=0 and poseKeypoints[index,16,0]!=0 : 
            orientation=0 #"facing the robot" 
        elif poseKeypoints[index,0,0]==0 and poseKeypoints[index,15,0]==0 and poseKeypoints[index,16,0]==0 : 
            orientation=1 #"giving the back"
        elif poseKeypoints[kk,0,0]!=0 and (poseKeypoints[kk,15,0]==0 and poseKeypoints[kk,17,0]==0)  and poseKeypoints[kk,16,0]!=0: 
            orientation=2 #"showing the left side"
        elif poseKeypoints[kk,0,0]!=0 and poseKeypoints[kk,15,0]!=0 and (poseKeypoints[kk,18,0]==0 and poseKeypoints[kk,16,0]==0): 
            orientation=3 #"showing the right side"
        #Sitting or standing inference
        legs=True # initial assumption that legs are detected
        for k in range(0,n_joints):
            if k>=10 and k<=14: #Only consider keypoints in the lower part of the body
                if joints_x[k]==0 and joints_y[k]==0: #if at least one joint is not detected, then assume no legs are detected
                    legs=False
        if legs==True:
            d9_10=abs(joints_y[9]-joints_y[10])
            d12_13=abs(joints_y[12]-joints_y[13])
            d_mean_1=(d9_10+d12_13)/2
            d10_11=abs(joints_y[11]-joints_y[10])
            d14_13=abs(joints_y[14]-joints_y[13])
            d_mean_2=(d10_11+d14_13)/2
            if d_mean_1<=0.3*d_mean_2:
                standing=False
            else:
                standing=True       
        else: # if not legs are detected, then assume standing is False
            standing=False
            
        #Final inference based on standing and orientation variables           
        if standing==True:    
            posture[0]=orientation
        else:
            posture[0]=orientation+4
        posture[1]=1 # 1 confidence
        return gesture, posture
        
################################################################################################
        
class robot_class:
    def __init__(self): #It is done only the first iteration
        self.position=[0,0,0]

    def robot_pose_callback(self,pose):
        self.position[0]=pose.position.x
        self.position[1]=pose.position.y
        quat = pose.orientation    
        # From quaternion to Euler
        angles = euler_from_quaternion((quat.x,quat.y,quat.z,quat.w))
        theta = angles[2]
        self.position[2]=np.unwrap([theta])[0]
        
###############################################################################################
# Main Script

if __name__ == '__main__':
    # Initialize our node
    time_init=time.time() 
    human=human_class()  
    robot=robot_class()
    rospy.init_node('gesture_estimator',anonymous=True)
    # Setup and call subscription
    rospy.Subscriber('/robot_pose', Pose, robot.robot_pose_callback) 
    ids_sub=message_filters.Subscriber('/people_ids', IdsList) 
    roi_sub=message_filters.Subscriber('/people_roi', NormalizedRegionOfInterest2D) 
    ts1 = message_filters.ApproximateTimeSynchronizer([roi_sub,ids_sub], 5, 1)
    ts1.registerCallback(human.roi_ids_callback)
   
    image_front_sub = message_filters.Subscriber(camera_topics[0], Image) 
    depth_front_sub = message_filters.Subscriber(camera_topics[1], Image) 
    ts2 = message_filters.ApproximateTimeSynchronizer([image_front_sub, depth_front_sub], 5, 1)
    ts2.registerCallback(human.rgbd_callback)
    #Rate setup
    rate = rospy.Rate(1/pub_hz) # main loop frecuency in Hz
    while not rospy.is_shutdown():
        
        #if human.new_data==True:
        #    human.new_data=False   
        rate.sleep() #to keep fixed the publishing loop rate
