FROM opensycl:latest

# Setup parameters
ARG ROOT_GIT=git@github.com:jolly-chen/root.git
ARG ROOT_BRANCH=sycl_histogram_wip

# Install dependencies
RUN sudo dnf install -y openssl openssl-devel xrootd xrootd-devel \
    xrootd-client xrootd-client-devel ninja-build freetype freetype-devel \
    lz4 lz4-devel xz xz-devel tbb tbb-devel xxhash xxhash-devel libzstd \
    libzstd-devel python3-devel libX11 libX11-devel Xorg \
    xorg-x11-server-devel libXpm libXpm-devel libXft libXft-devel \
    libXext libXext-devel
# RUN sudo dnf install cmake

# # Clone and compile ROOT
# RUN mkdir root && cd root && mkdir build && git clone ${ROOT_GIT}
# RUN cd root/root && git checkout ${ROOT_BRANCH} && cd ../build && \
#     cmake ../root -G Ninja -DCMAKE_CXX_STANDARD=17 -Dcuda=ON \
#     -Dsycl=ON -DOPENSYCL_TARGETS="omp;cuda:sm_70" \
#     -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
#     -DCMAKE_CUDA_ARCHITECTURES=70 -DCMAKE_BUILD_TYPE=RelWithDebInfo && \
#     cmake --build .

RUN mkdir root
RUN mkdir build
WORKDIR .
ADD . root/
WORKDIR build

RUN  cmake ../root -G Ninja -DCMAKE_CXX_STANDARD=17 -Dcuda=ON \
    -Dsycl=ON -DOPENSYCL_TARGETS="omp;cuda:sm_75" \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DCMAKE_CUDA_ARCHITECTURES=75
RUN  ninja


cmake ../root -G Ninja -DCMAKE_CXX_STANDARD=17 -Dtesting=OFF -Dcuda=ON -Dsycl=ON -DOPENSYCL_TARGETS="omp;cuda:sm_75" -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.8/bin/nvcc -DOpenSYCL_DIR=/home/jollychen/OpenSYCL/lib/cmake/OpenSYCL -Dtmva=OFF -Dcudnn=OFF -Dwebgui=OFF -Dspectrum=OFF

# Finish
RUN echo "source /home/sycl/root/build/bin/thisroot.sh" >> /home/sycl/.bashrc
