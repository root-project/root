#!/bin/sh -e

TESTNAME=ttree_read_imt
NTHREADS=4
NENTRIES=500
INPUTFILE="root://eospublic.cern.ch//eos/root-eos/testfiles/ttree_read_imt.root"

#ROOTDEBUG=1 ./$TESTNAME $NTHREADS $NENTRIES $INPUTFILE 1>${TESTNAME}.out 2>${TESTNAME}.err

if ROOTDEBUG=1 ./$TESTNAME $NTHREADS $NENTRIES $INPUTFILE 1>${TESTNAME}.out 2>${TESTNAME}.err ; then
   :
else
   echo "A problem was detected running the test"
   echo ""
   echo "*** Standard Output (last 100 lines) ***"
   echo ""
   tail -n 100 ${TESTNAME}.out
   echo ""
   echo "*** Standard Error ***"
   echo ""
   cat ${TESTNAME}.err
fi

# Print IMT messages from the application
cat ${TESTNAME}.out | grep -e " \[IMT\]"

grep -v -e "Info in"  -e 'HEAD http' -e 'GET http' -e 'Host:' -e 'User-Agent:' -e 'Range: ' -e '^\s*$' ${TESTNAME}.err | cat > /dev/stderr

# Print number of tasks executed
NUMTASKS=`cat ${TESTNAME}.err | grep -e "\[IMT\] Running task" | wc -l`
echo "NUM TASKS: $NUMTASKS"
