#!/bin/bash

### INSTALL PREREQUISITES

# TMULE
pip install tmule
sudo apt install tmux
echo "Tmule installed"
# JOBLIB
sudo apt-get install -y python3-joblib
echo "Joblib installed"

# SKLEARN
pip3 install scipy
>> pip install -U scikit-learn==0.21.3
echo "Scikit-learn installed"


#WEB_VIDEO_SERVER
if [ -d "~/catkin_ws/src/web_video_server" ] 
then
    echo "ROS package web_video_server exists" 
else
    cd ~/catkin_ws/src/
    git clone https://github.com/RobotWebTools/web_video_server
    cd ~/catkin_ws
    catkin_make --only-pkg-with-deps web_video_server
    source devel/setup.bash
    echo "ROS package web_video_server installed"
fi

