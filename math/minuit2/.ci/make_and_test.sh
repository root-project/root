# This is run by the CI system from the main Minuit2 directory

set -evx

mkdir build
cd build
cmake .. -Dminuit2_standalone=OFF -DCMAKE_INSTALL_PREFIX=install
make -j2
make test
make install
make clean

cmake .. -Dminuit2_standalone=ON -DCMAKE_INSTALL_PREFIX=install
make -j2
make test
make purge
