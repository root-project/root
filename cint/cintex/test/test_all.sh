#!/bin/sh

export ROOTSYS=`pwd`
export PYTHONDIR=$1/../../bin

export PATH=${ROOTSYS}/bin:${PATH}
export PYTHONPATH=${ROOTSYS}/lib:${PYTHONPATH}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${ROOTSYS}/cint/cintex/dict/lib:${ROOTSYS}/lib:${PYTHONDIR}/../lib:${ROOTSYS}/cint/cintex/test/dict

bin/root -b -q -l cint/cintex/test/test_Cintex.C
bin/root -b -q -l cint/cintex/test/test_Persistency.C
${PYTHONDIR}/python cint/cintex/test/test_PyCintex_basics.py -b
