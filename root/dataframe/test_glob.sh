#!/bin/bash -e

TESTNAME=test_glob

if root -l -b -q $TESTNAME.C+ 2>${TESTNAME}.err | tee ${TESTNAME}.out; then
   :
else
   echo ""
   echo "*** Standard Error ***"
   echo ""
   cat ${TESTNAME}.err
fi
