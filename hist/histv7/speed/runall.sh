#!/bin/sh

what=$1

CXX=g++
prefix="rvalgrind --tool=callgrind --callgrind-out-file=callgrind.out.$1.%p --dump-instr=yes"

. ${ROOTSYS}/bin/thisroot.sh

set -x

function run {
   echo "Data Class: $1" | tee $1.1e6.$$.speedlog | tee $1.1e8.$$.speedlog
   ${CXX} -o speedtest histspeedtest.cxx `root-config --cflags --libs` -O3 "-DSTATCLASSES=$1"
   # ${prefix} ./speedtest 1e6 > $1.1e6.$$.log
   # rvalgrind --tool=callgrind --callgrind-out-file=callgrind.out.$1.%p --dump-instr=yes ./speedtest 1e6 > $1.1e6.$$.vallog
   ./speedtest 1e6 $what >> $1.1e6.$$.speedlog
   ./speedtest 1e8 $what >> $1.1e8.$$.speedlog
}

run THistStatContent
run "THistStatContent,THistStatUncertainty"
run "THistStatContent,THistStatUncertainty,THistStatTotalSumOfWeights"
run "THistStatContent,THistStatUncertainty,THistStatTotalSumOfWeights,THistStatTotalSumOfSquaredWeights"
run "THistStatContent,THistStatUncertainty,THistStatTotalSumOfWeights,THistStatTotalSumOfSquaredWeights,THistDataMomentUncert"

echo 'Not running with THistDataRuntime'
# run THistDataRuntime
