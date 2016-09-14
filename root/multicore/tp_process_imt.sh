#!/bin/bash -e

TESTNAME=tp_process_imt
NTHREADS=4
INPUTFILE=http://root.cern.ch/files/tp_process_imt.root
TREENAME=events

./$TESTNAME $NTHREADS $INPUTFILE $TREENAME 1>${TESTNAME}.out

# Print IMT messages from the application
cat ${TESTNAME}.out | grep -v "Finished task"

# Print number of tasks executed
NUMTASKS=`cat ${TESTNAME}.out | grep -e "\[IMT\] Finished task" | wc -l`
echo "NUM TASKS: $NUMTASKS"

