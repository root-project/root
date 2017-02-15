#!/bin/bash -e

TESTNAME=ttree_read_imt
NTHREADS=4
NENTRIES=500
INPUTFILE=ttree_read_imt.root

#ROOTDEBUG=1 ./$TESTNAME $NTHREADS $NENTRIES $INPUTFILE 1>${TESTNAME}.out 2>${TESTNAME}.err

if [ !resp=$(ROOTDEBUG=1 ./$TESTNAME $NTHREADS $NENTRIES $INPUTFILE 1>${TESTNAME}.out 2>${TESTNAME}.err) ]; then
   echo "A problem was detected running the test"
   echo ""
   echo "*** Standard Output ***"
   echo ""
   cat ${TESTNAME}.out
   echo ""
   echo "*** Standard Error ***"
   echo ""
   cat ${TESTNAME}.err
else
   echo "Everything ok"
fi

# Print IMT messages from the application
cat ${TESTNAME}.out | grep -e " \[IMT\]"

grep -v -e "Info in" ${TESTNAME}.err | cat > /dev/stderr

# Print number of threads actually used
OBSERVED_NTHREADS=`cat ${TESTNAME}.err | grep -e "\[IMT\] Thread" | sort | uniq | wc -l`
echo "NUM THREADS: $OBSERVED_NTHREADS"

# Print number of tasks executed
NUMTASKS=`cat ${TESTNAME}.err | grep -e "\[IMT\] Running task" | wc -l`
echo "NUM TASKS: $NUMTASKS"
