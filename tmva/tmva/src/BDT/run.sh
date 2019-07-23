#!/bin/bash

# colorize stdin according to parameter passed (GREEN, CYAN, BLUE, YELLOW)
colorize(){
    GREEN="\033[0;32m"
    CYAN="\033[0;36m"
    GRAY="\033[0;37m"
    BLUE="\033[0;34m"
    YELLOW="\033[0;33m"
    NORMAL="\033[m"
    color=\$${1:-NORMAL}
    # activate color passed as argument
    echo -ne "`eval echo ${color}`"
    # read stdin (pipe) and print from it:
    cat
    # Note: if instead of reading from the pipe, you wanted to print
    # the additional parameters of the function, you could do:
    # shift; echo $*
    # back to normal (no color)
    echo -ne "${NORMAL}"
}

function bPrint() { echo && echo "$1" | colorize CYAN; }


bPrint " ***** Running python ***** "
python train.py


#g++ main.cxx -o main -std=c++11
make distclean && make
bPrint " ***** Running main C++ ***** "
./main.exe


rm ./test.exe
g++ test.cxx -std=c++11 `root-config --libs --cflags` -o test.exe
bPrint "***** Running test.cxx *****"
./test.exe

bPrint "***** Running check_preds.py *****"
python check_preds.py

bPrint "***** Benchmarking *****"
rm ./mybenchmark.exe
g++ -std=c++11 -isystem benchmark/include -Lbuild/src benchmark.cxx -lbenchmark -lpthread -O2 -o mybenchmark.exe
./mybenchmark.exe
