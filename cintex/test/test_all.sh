#!/bin/sh

export ROOTSYS=`pwd`
export PYTHONDIR=$1/../../bin

export PATH=${ROOTSYS}/bin:${PATH}
export PYTHONPATH=${ROOTSYS}/lib:${PYTHONPATH}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${ROOTSYS}/lib:${PYTHONDIR}/../lib:${ROOTSYS}/cintex/test/dict

bin/root -b -q -l cintex/test/test_Cintex.C
bin/root -b -q -l cintex/test/test_Persistency.C
${PYTHONDIR}/python cintex/test/test_PyCintex_basics.py -b
