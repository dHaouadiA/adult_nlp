#!/bin/bash
docker rm mongodb -f
docker run -d\
    --hostname mongodb \
    --name=mongodb \
    --restart=always \
    --volume /root/Desktop/Mongo/db:/data/db:rw \
    mongo:build