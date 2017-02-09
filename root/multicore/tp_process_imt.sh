#!/bin/bash -e

TESTNAME=tp_process_imt
NTHREADS=4
INPUTFILE=http://root.cern.ch/files/tp_process_imt_small.root
TREENAME=events

#### Filename constructor 
./$TESTNAME $NTHREADS $INPUTFILE $TREENAME "filename" 1>${TESTNAME}.out

# Print IMT messages from the application
cat ${TESTNAME}.out | grep -v "Finished task"

# Print number of tasks executed
NUMTASKS=`cat ${TESTNAME}.out | grep -e "\[IMT\] Finished task" | wc -l`
echo "NUM TASKS (filename): $NUMTASKS"


#### Collection-of-file-names constructor
./$TESTNAME $NTHREADS $INPUTFILE $TREENAME "collection" 1>${TESTNAME}.out

# Print IMT messages from the application
cat ${TESTNAME}.out | grep -v "Finished task"

# Print number of tasks executed
NUMTASKS=`cat ${TESTNAME}.out | grep -e "\[IMT\] Finished task" | wc -l`
echo "NUM TASKS (collection): $NUMTASKS"


#### Chain constructor
./$TESTNAME $NTHREADS $INPUTFILE $TREENAME "chain" 1>${TESTNAME}.out

# Print IMT messages from the application
cat ${TESTNAME}.out | grep -v "Finished task"

# Print number of tasks executed
NUMTASKS=`cat ${TESTNAME}.out | grep -e "\[IMT\] Finished task" | wc -l`
echo "NUM TASKS (chain): $NUMTASKS"
