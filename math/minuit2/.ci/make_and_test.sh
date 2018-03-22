# This is run by the CI system from the main Minuit2 directory

mkdir build
cd build
cmake .. -Dminuit2-standalone=OFF -DCMAKE_INSTALL_PREFIX=install
make -j2
make test
make install
make clean

cmake .. -Dminuit2-standalone=ON -DCMAKE_INSTALL_PREFIX=install
make -j2
make test
make purge
