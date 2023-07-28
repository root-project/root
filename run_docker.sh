docker kill root
docker rm root
docker run -it -d --name root --network=host root-img

