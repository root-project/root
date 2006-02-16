#!/bin/sh

export ROOTSYS=`pwd`
export PYTHONDIR=$1/..

export PATH=${ROOTSYS}/bin:${PATH}
export PYTHONPATH=${ROOTSYS}/lib;${PYTHONPATH}

bin/root -b -q -l cintex/test/test_Cintex.C
bin/root -b -q -l cintex/test/test_Persistency.C
${PYTHONDIR}/python cintex/test/test_PyCintex_basics.py
