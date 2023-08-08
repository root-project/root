docker kill root
docker rm root
#docker run -it -d --name root --network=host --mount type=bind,src=/home/ionna/root/root_build,destination=/home/ioanna/root_build --mount type=bind,src=/home/ionna/root/root_src/,destination=/home/ioanna/root_src root-img
#docker run -it -d --name root --network=host root-img
sudo docker run -it -d --name root --network=host --device /dev/dri -v /dev/dri/card0:/dev/dri/card0 --mount type=bind,src=/data/ipanagou/root_src,destination=/home/ioanna/root_src ipanagou/root:latest