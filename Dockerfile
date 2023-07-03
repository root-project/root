FROM ubuntu:latest

# Update packages and install dependencies
RUN apt-get update && \
    apt-get install -y curl python3 python3-pip

# Install Docker
RUN curl -fsSL https://get.docker.com -o get-docker.sh && \
    sh get-docker.sh

# Install openstacksdk
RUN pip3 install openstacksdk

# Set the default command
CMD ["/bin/bash"]
