#!/bin/bash

container_name="yoloe"
docker_image="intflow/yoloe:v0.0.0.1"

# xhost +

sudo docker run \
--name=${container_name} \
--net=host \
--privileged \
--ipc=host \
--gpus all \
-w /works \
-v /home/intflow/works:/works \
-v /DL_data_super_hdd:/DL_data_super_hdd \
-v /DL_data_super_ssd:/DL_data_super_ssd \
-it ${docker_image} /bin/bash
