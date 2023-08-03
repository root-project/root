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

RUN sudo apt-get install -y --no-install-recommends \
    curl ca-certificates gpg-agent software-properties-common && \
    sudo rm -rf /var/lib/apt/lists/*

# repository to install Intel(R) oneAPI Libraries
RUN sudo curl -fsSL https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2023.PUB | sudo gpg --dearmor | sudo tee /usr/share/keyrings/intel-oneapi-archive-keyring.gpg
RUN sudo sh -c "echo 'deb [signed-by=/usr/share/keyrings/intel-oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main'  > /etc/apt/sources.list.d/oneAPI.list"

RUN sudo apt-get update && sudo apt-get upgrade -y && \
    sudo apt-get install -y --no-install-recommends \
    curl ca-certificates gpg-agent software-properties-common && \
    sudo rm -rf /var/lib/apt/lists/*
# repository to install Intel(R) GPU drivers
RUN sudo curl -fsSL https://repositories.intel.com/graphics/intel-graphics.key | sudo gpg --dearmor | sudo tee /usr/share/keyrings/intel-graphics-archive-keyring.gpg
RUN sudo sh -c "echo 'deb [signed-by=/usr/share/keyrings/intel-graphics-archive-keyring.gpg arch=amd64] https://repositories.intel.com/graphics/ubuntu jammy flex'  > /etc/apt/sources.list.d/intel-graphics.list"

RUN sudo apt-get update && sudo apt-get upgrade -y && \
    sudo apt-get install -y --no-install-recommends \
    ca-certificates build-essential pkg-config gnupg libarchive13 openssh-server openssh-client wget net-tools git intel-basekit-getting-started intel-oneapi-advisor intel-oneapi-ccl-devel intel-oneapi-common-licensing intel-oneapi-common-vars intel-oneapi-compiler-dpcpp-cpp intel-oneapi-dal-devel intel-oneapi-dev-utilities intel-oneapi-dnnl-devel intel-oneapi-dpcpp-debugger intel-oneapi-ipp-devel intel-oneapi-ippcp-devel intel-oneapi-libdpstd-devel intel-oneapi-mkl-devel intel-oneapi-tbb-devel intel-oneapi-vtune intel-level-zero-gpu level-zero  && \
    sudo rm -rf /var/lib/apt/lists/*

ENV LANG=C.UTF-8
ENV ACL_BOARD_VENDOR_PATH='/opt/Intel/OpenCLFPGA/oneAPI/Boards'
ENV ADVISOR_2023_DIR='/opt/intel/oneapi/advisor/2023.2.0'
ENV APM='/opt/intel/oneapi/advisor/2023.2.0/perfmodels'
ENV CCL_CONFIGURATION='cpu_gpu_dpcpp'
ENV CCL_ROOT='/opt/intel/oneapi/ccl/2021.10.0'
ENV CLASSPATH='/opt/intel/oneapi/mpi/2021.10.0//lib/mpi.jar:/opt/intel/oneapi/dal/2023.2.0/lib/onedal.jar'
ENV CMAKE_PREFIX_PATH='/opt/intel/oneapi/tbb/2021.10.0/env/..:/opt/intel/oneapi/ipp/2021.9.0/lib/cmake/ipp:/opt/intel/oneapi/dnnl/2023.2.0/cpu_dpcpp_gpu_dpcpp/../lib/cmake:/opt/intel/oneapi/dal/2023.2.0:/opt/intel/oneapi/compiler/2023.2.0/linux/IntelDPCPP:/opt/intel/oneapi/ccl/2021.10.0/lib/cmake/oneCCL'
ENV CMPLR_ROOT='/opt/intel/oneapi/compiler/2023.2.0'
ENV CPATH='/opt/intel/oneapi/tbb/2021.10.0/env/../include:/opt/intel/oneapi/mpi/2021.10.0//include:/opt/intel/oneapi/mkl/2023.2.0/include:/opt/intel/oneapi/ippcp/2021.8.0/include:/opt/intel/oneapi/ipp/2021.9.0/include:/opt/intel/oneapi/dpl/2022.2.0/linux/include:/opt/intel/oneapi/dnnl/2023.2.0/cpu_dpcpp_gpu_dpcpp/include:/opt/intel/oneapi/dev-utilities/2021.10.0/include:/opt/intel/oneapi/dal/2023.2.0/include:/opt/intel/oneapi/compiler/2023.2.0/linux/lib/oclfpga/include:/opt/intel/oneapi/ccl/2021.10.0/include/cpu_gpu_dpcpp'
ENV DAALROOT='/opt/intel/oneapi/dal/2023.2.0'
ENV DALROOT='/opt/intel/oneapi/dal/2023.2.0'
ENV DAL_MAJOR_BINARY='1'
ENV DAL_MINOR_BINARY='1'
ENV DIAGUTIL_PATH='/opt/intel/oneapi/vtune/2023.2.0/sys_check/vtune_sys_check.py:/opt/intel/oneapi/debugger/2023.2.0/sys_check/debugger_sys_check.py:/opt/intel/oneapi/compiler/2023.2.0/sys_check/sys_check.sh:/opt/intel/oneapi/advisor/2023.2.0/sys_check/advisor_sys_check.py:'
ENV DNNLROOT='/opt/intel/oneapi/dnnl/2023.2.0/cpu_dpcpp_gpu_dpcpp'
ENV DPL_ROOT='/opt/intel/oneapi/dpl/2022.2.0'
ENV FI_PROVIDER_PATH='/opt/intel/oneapi/mpi/2021.10.0//libfabric/lib/prov:/usr/lib/x86_64-linux-gnu/libfabric'
ENV FPGA_VARS_ARGS=''
ENV FPGA_VARS_DIR='/opt/intel/oneapi/compiler/2023.2.0/linux/lib/oclfpga'
ENV GDB_INFO='/opt/intel/oneapi/debugger/2023.2.0/documentation/info/'
ENV INFOPATH='/opt/intel/oneapi/debugger/2023.2.0/gdb/intel64/lib'
ENV INTELFPGAOCLSDKROOT='/opt/intel/oneapi/compiler/2023.2.0/linux/lib/oclfpga'
ENV INTEL_PYTHONHOME='/opt/intel/oneapi/debugger/2023.2.0/dep'
ENV IPPCP_TARGET_ARCH='intel64'
ENV IPPCRYPTOROOT='/opt/intel/oneapi/ippcp/2021.8.0'
ENV IPPROOT='/opt/intel/oneapi/ipp/2021.9.0'
ENV IPP_TARGET_ARCH='intel64'
ENV I_MPI_ROOT='/opt/intel/oneapi/mpi/2021.10.0'
ENV LD_LIBRARY_PATH='/opt/intel/oneapi/tbb/2021.10.0/env/../lib/intel64/gcc4.8:/opt/intel/oneapi/mpi/2021.10.0//libfabric/lib:/opt/intel/oneapi/mpi/2021.10.0//lib/release:/opt/intel/oneapi/mpi/2021.10.0//lib:/opt/intel/oneapi/mkl/2023.2.0/lib/intel64:/opt/intel/oneapi/ippcp/2021.8.0/lib/intel64:/opt/intel/oneapi/ipp/2021.9.0/lib/intel64:/opt/intel/oneapi/dnnl/2023.2.0/cpu_dpcpp_gpu_dpcpp/lib:/opt/intel/oneapi/debugger/2023.2.0/gdb/intel64/lib:/opt/intel/oneapi/debugger/2023.2.0/libipt/intel64/lib:/opt/intel/oneapi/debugger/2023.2.0/dep/lib:/opt/intel/oneapi/dal/2023.2.0/lib/intel64:/opt/intel/oneapi/compiler/2023.2.0/linux/lib:/opt/intel/oneapi/compiler/2023.2.0/linux/lib/x64:/opt/intel/oneapi/compiler/2023.2.0/linux/lib/oclfpga/host/linux64/lib:/opt/intel/oneapi/compiler/2023.2.0/linux/compiler/lib/intel64_lin:/opt/intel/oneapi/ccl/2021.10.0/lib/cpu_gpu_dpcpp:/opt/intel/oneapi/compiler/2023.2.0/linux/compiler/lib/intel64_lin:/opt/intel/oneapi/ccl/2021.10.0/lib/cpu_gpu_dpcpp'
ENV LIBRARY_PATH='/opt/intel/oneapi/tbb/2021.10.0/env/../lib/intel64/gcc4.8:/opt/intel/oneapi/mpi/2021.10.0//libfabric/lib:/opt/intel/oneapi/mpi/2021.10.0//lib/release:/opt/intel/oneapi/mpi/2021.10.0//lib:/opt/intel/oneapi/mkl/2023.2.0/lib/intel64:/opt/intel/oneapi/ippcp/2021.8.0/lib/intel64:/opt/intel/oneapi/ipp/2021.9.0/lib/intel64:/opt/intel/oneapi/dnnl/2023.2.0/cpu_dpcpp_gpu_dpcpp/lib:/opt/intel/oneapi/dal/2023.2.0/lib/intel64:/opt/intel/oneapi/compiler/2023.2.0/linux/compiler/lib/intel64_lin:/opt/intel/oneapi/compiler/2023.2.0/linux/lib:/opt/intel/oneapi/ccl/2021.10.0/lib/cpu_gpu_dpcpp'
ENV MANPATH='/opt/intel/oneapi/mpi/2021.10.0/man:/opt/intel/oneapi/debugger/2023.2.0/documentation/man:/opt/intel/oneapi/compiler/2023.2.0/documentation/en/man/common::'
ENV MKLROOT='/opt/intel/oneapi/mkl/2023.2.0'
ENV NLSPATH='/opt/intel/oneapi/mkl/2023.2.0/lib/intel64/locale/%l_%t/%N:/opt/intel/oneapi/compiler/2023.2.0/linux/compiler/lib/intel64_lin/locale/%l_%t/%N'
ENV OCL_ICD_FILENAMES='libintelocl_emu.so:libalteracl.so:/opt/intel/oneapi/compiler/2023.2.0/linux/lib/x64/libintelocl.so'
ENV ONEAPI_ROOT='/opt/intel/oneapi'
ENV PATH='/opt/intel/oneapi/vtune/2023.2.0/bin64:/opt/intel/oneapi/mpi/2021.10.0//libfabric/bin:/opt/intel/oneapi/mpi/2021.10.0//bin:/opt/intel/oneapi/mkl/2023.2.0/bin/intel64:/opt/intel/oneapi/dev-utilities/2021.10.0/bin:/opt/intel/oneapi/debugger/2023.2.0/gdb/intel64/bin:/opt/intel/oneapi/compiler/2023.2.0/linux/lib/oclfpga/bin:/opt/intel/oneapi/compiler/2023.2.0/linux/bin/intel64:/opt/intel/oneapi/compiler/2023.2.0/linux/bin:/opt/intel/oneapi/advisor/2023.2.0/bin64:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'
ENV PKG_CONFIG_PATH='/opt/intel/oneapi/vtune/2023.2.0/include/pkgconfig/lib64:/opt/intel/oneapi/tbb/2021.10.0/env/../lib/pkgconfig:/opt/intel/oneapi/mpi/2021.10.0/lib/pkgconfig:/opt/intel/oneapi/mkl/2023.2.0/lib/pkgconfig:/opt/intel/oneapi/ippcp/2021.8.0/lib/pkgconfig:/opt/intel/oneapi/dpl/2022.2.0/lib/pkgconfig:/opt/intel/oneapi/dnnl/2023.2.0/cpu_dpcpp_gpu_dpcpp/../lib/pkgconfig:/opt/intel/oneapi/dal/2023.2.0/lib/pkgconfig:/opt/intel/oneapi/compiler/2023.2.0/lib/pkgconfig:/opt/intel/oneapi/ccl/2021.10.0/lib/pkgconfig:/opt/intel/oneapi/advisor/2023.2.0/include/pkgconfig/lib64:'
ENV PYTHONPATH='/opt/intel/oneapi/advisor/2023.2.0/pythonapi'
ENV SETVARS_COMPLETED='1'
ENV TBBROOT='/opt/intel/oneapi/tbb/2021.10.0/env/..'
ENV VTUNE_PROFILER_2023_DIR='/opt/intel/oneapi/vtune/2023.2.0'
ENV VTUNE_PROFILER_DIR='/opt/intel/oneapi/vtune/2023.2.0'

WORKDIR /home/ioanna

RUN sudo apt-get update && sudo apt-get upgrade -y
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

RUN echo "source /opt/intel/oneapi/setvars.sh" >> /home/ioanna/.bashrc
RUN echo "export PATH=/home/ioanna/.local/bin:$PATH" >> /home/ioanna/.bashrc

WORKDIR /home/ioanna
ENV PATH=/home/ioanna/.local/bin:$PATH
# COPY root_src /home/ioanna/root_src/ 
RUN cd root_build && cmake -DCMAKE_INSTALL_PREFIX=/home/ioanna/root_install -Dtmva-sofie=On -Dtmva-pymva=Off -DPython3_executable=/usr/bin/python3 -Dtesting=On -DBLAS_LIBRARIES=/usr/lib/x86_64-linux-gnu/blas/libblas.so -DProtobuf_LIBRARIES=/usr/lib/x86_64-linux-gnu/libprotobuf.so -Dtmva-sycl=On -DCMAKE_C_COMPILER=/opt/intel/oneapi/compiler/2023.2.1/linux/bin/icx -DCMAKE_CXX_COMPILER=/opt/intel/oneapi/compiler/2023.2.1/linux/bin/icpx -DIntelSYCL_DIR=/opt/intel/oneapi/compiler/2023.2.1/linux/IntelSYCL/ -Dxrootd=Off -Dimt=Off -Dbuiltin_xrootd=Off /home/ioanna/root_src && cmake --build . -j16 --target install

