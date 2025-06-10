#!/bin/bash

container_name="yoloe_dev1"
docker_image="intflow/yoloe:v0.0.0.1"

# xhost +

sudo docker run \
--name=${container_name} \
--net=host \
--privileged \
--ipc=host \
--gpus all \
-e DISPLAY=$DISPLAY \
-v /tmp/.X11-unix:/tmp/.X11-unix \
-v /home/$USER/.Xauthority:/root/.Xauthority:rw \
-w /works \
-v /home/intflow/works:/works \
-v /DL_data_super_hdd:/DL_data_super_hdd \
-it ${docker_image} /bin/bash
