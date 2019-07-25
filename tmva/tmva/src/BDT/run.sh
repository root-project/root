#!/bin/bash

# colorize stdin according to parameter passed (GREEN, CYAN, BLUE, YELLOW)
colorize(){
    GREEN="\033[0;32m"
    CYAN="\033[0;36m"
    GRAY="\033[0;37m"
    BLUE="\033[0;34m"
    YELLOW="\033[0;33m"
    NORMAL="\033[m"
    RED="\033[0;31m"

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
function rPrint() { echo && echo "$1" | colorize RED; }
function gPrint() { echo && echo "$1" | colorize GREEN; }


echo "Coloring: python(cyan); c++(yellow); messages(green); normal for the rest"

gPrint " ***** Running python ***** "
python train.py | colorize CYAN


#g++ main.cxx -o main -std=c++11
#make distclean
make
gPrint " ***** Running main C++ ***** "
./main.exe | colorize YELLOW


#rm ./test.exe
g++ test.cxx -std=c++11 `root-config --libs --cflags` -o test.exe
gPrint "***** Running test.cxx *****"
./test.exe | colorize YELLOW

gPrint "***** Running check_preds.py *****"
python check_preds.py | colorize CYAN

gPrint "***** Benchmarking *****"
rm ./mybenchmark.exe
#make -f makefile_bench.make distclean
make -f makefile_bench.make
#g++ -std=c++11 -isystem benchmark/include -Lbuild/src benchmark.cxx -lbenchmark -lpthread -O2 -o mybenchmark.exe
./mybenchmark.exe
