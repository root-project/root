FROM ubuntu:22.04
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections
RUN ln -snf /usr/share/zoneinfo/Europe/Athens /etc/localtime && echo Europe/Athens > /etc/timezone
RUN apt-get update && apt-get install -y --no-install-recommends apt-utils sudo
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y keyboard-configuration
RUN apt-get update && apt-get -y install sudo

RUN adduser --disabled-password --gecos '' ioanna
RUN adduser ioanna sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER ioanna
RUN sudo apt-get update && sudo apt-get -y upgrade
RUN sudo apt-get install -y git

WORKDIR /home/ioanna

# Install python, pip
RUN sudo apt-get install -y python3
RUN sudo apt-get install -y pip
RUN pip install --upgrade pip

# Install ROOT prerequisites
RUN sudo apt-get install -y dpkg-dev cmake g++ gcc binutils libx11-dev libxpm-dev \
libxft-dev libxext-dev libssl-dev

RUN sudo apt-get install -y gfortran libpcre3-dev \
xlibmesa-glu-dev libglew-dev libftgl-dev \
libmysqlclient-dev libfftw3-dev libcfitsio-dev \
graphviz-dev libavahi-compat-libdnssd-dev \
libldap2-dev libxml2-dev libkrb5-dev \
libgsl0-dev qtwebengine5-dev

# Install BLAS
RUN sudo apt-get install -y libblas-dev liblapack-dev

# Install Protobuf
RUN sudo apt-get install -y libprotobuf-dev protobuf-compiler

# Install pip packages
RUN pip3 install numpy
RUN pip3 install torch==1.13.0
RUN pip3 install onnx==1.14.0
RUN pip3 install protobuf==3.20.3

# Create build, install directories
RUN mkdir root_build
RUN mkdir root_install
RUN mkdir root_src

# Install Intel Oneapi
RUN sudo apt-get install -y --no-install-recommends \
    curl ca-certificates gpg-agent software-properties-common && \
    sudo rm -rf /var/lib/apt/lists/*

RUN sudo curl -fsSL https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2023.PUB | sudo gpg --dearmor | sudo tee /usr/share/keyrings/intel-oneapi-archive-keyring.gpg

RUN sudo sh -c  "echo 'deb [signed-by=/usr/share/keyrings/intel-oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main'  > /etc/apt/sources.list.d/oneAPI.list"

RUN sudo apt-get update && sudo apt-get upgrade -y && \
    sudo apt-get install -y --no-install-recommends \
    curl ca-certificates gpg-agent software-properties-common && \
    sudo rm -rf /var/lib/apt/lists/*

RUN sudo curl -fsSL https://repositories.intel.com/graphics/intel-graphics.key | sudo gpg --dearmor | sudo tee /usr/share/keyrings/intel-graphics-archive-keyring.gpg
RUN sudo sh -c "echo 'deb [signed-by=/usr/share/keyrings/intel-graphics-archive-keyring.gpg arch=amd64] https://repositories.intel.com/graphics/ubuntu jammy flex' > /etc/apt/sources.list.d/intel-graphics.list"

RUN sudo apt-get update && sudo apt-get upgrade -y && \ 
    sudo apt-get install -y --no-install-recommends \
    ca-certificates build-essential pkg-config gnupg libarchive13 openssh-server openssh-client wget net-tools git intel-basekit-getting-started intel-oneapi-advisor intel-oneapi-ccl-devel intel-oneapi-common-licensing intel-oneapi-common-vars intel-oneapi-compiler-dpcpp-cpp intel-oneapi-dal-devel intel-oneapi-dev-utilities intel-oneapi-dnnl-devel intel-oneapi-dpcpp-debugger intel-oneapi-ipp-devel intel-oneapi-ippcp-devel intel-oneapi-libdpstd-devel intel-oneapi-mkl-devel intel-oneapi-tbb-devel intel-oneapi-vtune intel-level-zero-gpu level-zero  && \
  sudo rm -rf /var/lib/apt/lists/*

RUN echo "source /opt/intel/oneapi/setvars.sh" >> /home/ioanna/.bashrc
RUN echo "export PATH=/home/ioanna/.local/bin:$PATH" >> /home/ioanna/.bashrc

COPY root_src /home/ioanna/root_src/ 

# Important to update after intel installation
RUN sudo apt-get update

# Install tbb 2018
WORKDIR /home/ioanna
RUN wget https://github.com/oneapi-src/oneTBB/archive/refs/tags/v2021.10.0.tar.gz
RUN tar -zxvf v2021.10.0.tar.gz
RUN cd oneTBB-2021.10.0
WORKDIR /home/ioanna/oneTBB-2021.10.0
RUN mkdir build && cd build
RUN mkdir /home/ioanna/tbb_install
RUN cmake -DCMAKE_INSTALL_PREFIX=/home/ioanna/tbb_install/ -DTBB_TEST=off
RUN cmake --build . -j16
RUN cmake --install .

WORKDIR /home/ioanna
RUN /bin/bash -c "source /opt/intel/oneapi/setvars.sh"
ENV PATH=/home/ioanna/.local/bin:$PATH
RUN cd root_build && cmake -DCMAKE_INSTALL_PREFIX=/home/ioanna/root_install -Dtmva-sofie=On -Dtmva-pymva=On -DPython3_executable=/usr/bin/python3 -Dtesting=On -DBLAS_LIBRARIES=/usr/lib/x86_64-linux-gnu/blas/libblas.so -DProtobuf_LIBRARIES=/usr/lib/x86_64-linux-gnu/libprotobuf.so -DTBB_LIBRARIES=/home/ioanna/tbb_install/lib64/libtbb.so /home/ioanna/root_src && sudo cmake --build . -j16 --target install
