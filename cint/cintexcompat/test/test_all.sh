#!/bin/sh

export ROOTSYS=`pwd`

export PATH=${ROOTSYS}/bin:${PATH}
export PYTHONPATH=${ROOTSYS}/lib:${PYTHONPATH}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${ROOTSYS}/cint/cintex/dict/lib:${ROOTSYS}/lib:${ROOTSYS}/cint/cintex/test/dict

bin/root -b -q -l cint/cintex/test/test_Cintex.C
bin/root -b -q -l cint/cintex/test/test_Persistency.C
python cint/cintex/test/test_PyCintex_basics.py -b
