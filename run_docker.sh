docker kill root
docker rm root
docker run -it -d --name root --network=host --mount type=bind,src=/home/ionna/root/root_src,destination=/home/ioanna/root_src/ root-img

