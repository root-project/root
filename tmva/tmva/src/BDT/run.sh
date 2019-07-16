#!/bin/bash

python train.py
echo "\n ***** Finished with python ***** \n"

#g++ main.cxx -o main -std=c++11
make distclean && make
./main.exe
echo "\n ***** Finished with first C++ ***** \n"

rm ./test.exe
g++ test.cxx `root-config --libs --cflags` -o test.exe -std=c++11
./test.exe
