#!/bin/sh -e

TESTNAME=ttree_read_imt_allpar
NTHREADS=4
NENTRIES=500
INPUTFILE=./ttree_read_imt.root

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

grep -v -e "Info in" ${TESTNAME}.err | cat > /dev/stderr

# Print number of threads actually used
OBSERVED_NTHREADS=`cat ${TESTNAME}.err | grep -E "\[IMT\] Thread ([0-9]+|0x[0-9a-f]+)$" | sort | uniq | wc -l`
echo "NUM THREADS: $OBSERVED_NTHREADS"

# Print number of tasks executed
NUMTASKS=`cat ${TESTNAME}.err | grep -e "\[IMT\] Running task" | wc -l`
echo "NUM TASKS: $NUMTASKS"
