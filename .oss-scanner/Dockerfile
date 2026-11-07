FROM ubuntu:26.10

ENV LANG=C.UTF-8

ADD https://raw.githubusercontent.com/root-project/root/master/requirements.txt requirements-root.txt

RUN apt-get update -qq \
 && ln -sf /usr/share/zoneinfo/UTC /etc/localtime  \
 && apt-get -y install automake binutils ca-certificates ccache civetweb cmake cppzmq-dev curl davix-dev dcap-dev dpkg-dev fonts-freefont-ttf g++ gcc gfortran git graphviz-dev libblas-dev libcfitsio-dev libcivetweb-dev libcrypt-dev libcurl4-openssl-dev libfftw3-dev libfreetype6-dev libftgl-dev libgif-dev libgl2ps-dev libglew-dev libglu-dev libgmock-dev libgraphviz-dev libgsl-dev libgsl0-dev libgtest-dev libjpeg-dev liblz4-dev liblzma-dev libpcre2-dev libpng-dev libpq-dev libprotobuf-dev libprotoc-dev libsm-dev libsqlite3-dev libssl-dev libtbb-dev libtiff-dev libunuran-dev libuuid1 libvdt-dev libx11-dev libxext-dev libxft-dev libxml2-dev libxpm-dev libxrootd-client-dev libxxhash-dev libz-dev libzmq3-dev libzstd-dev locales xvfb xauth make ninja-build nlohmann-json3-dev openjdk-21-jre-headless protobuf-compiler python3 python3-dev python3-onnx python3-venv uuid-dev wget xlibmesa-glu-dev xrootd-plugins --no-install-recommends \
 && apt-get autoremove -y \
 && apt-get clean -y \
 && rm -rf /var/cache/apt/archives/* \
 && rm -rf /var/lib/apt/lists/*

RUN update-ca-certificates

RUN mkdir -p /py-venv \
 && python3 -m venv /py-venv/ROOT-CI \
 && /py-venv/ROOT-CI/bin/pip install --no-cache-dir --upgrade pip \
 && /py-venv/ROOT-CI/bin/pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
 && /py-venv/ROOT-CI/bin/pip install --no-cache-dir -r requirements-root.txt openstacksdk \
 && rm -f requirements-root.txt

 RUN git clone https://github.com/root-project/root \
 && mkdir build \
 && cd build \
 && cmake -Dminimal=ON -Dtesting=ON -Droottest=ON ../root \
 && make -j 16