#!/bin/sh -e

TESTNAME=tp_process_imt
NTHREADS=4
INPUTFILE=tp_process_imt.root
TREENAME=myTree

#### Filename constructor 
./$TESTNAME $NTHREADS $INPUTFILE $TREENAME "filename" 1>${TESTNAME}.out

# Print IMT messages from the application
cat ${TESTNAME}.out | grep -v "\[IMT\] Finished task"

# Print number of tasks executed
NUMTASKS=`cat ${TESTNAME}.out | grep -e "\[IMT\] Finished task" | wc -l`
echo "NUM TASKS (filename): $NUMTASKS"


#### Collection-of-file-names constructor
./$TESTNAME $NTHREADS $INPUTFILE $TREENAME "collection" 1>${TESTNAME}.out

# Print IMT messages from the application
cat ${TESTNAME}.out | grep -v "\[IMT\] Finished task"

# Print number of tasks executed
NUMTASKS=`cat ${TESTNAME}.out | grep -e "\[IMT\] Finished task" | wc -l`
echo "NUM TASKS (collection): $NUMTASKS"


#### Chain constructor
./$TESTNAME $NTHREADS $INPUTFILE $TREENAME "chain" 1>${TESTNAME}.out

# Print IMT messages from the application
cat ${TESTNAME}.out | grep -v "\[IMT\] Finished task"

# Print number of tasks executed
NUMTASKS=`cat ${TESTNAME}.out | grep -e "\[IMT\] Finished task" | wc -l`
echo "NUM TASKS (chain): $NUMTASKS"


#### List-of-entries constructor
./$TESTNAME $NTHREADS $INPUTFILE $TREENAME "entrylist" 1>${TESTNAME}.out

# Print IMT messages from the application
cat ${TESTNAME}.out | grep -v "\[IMT\] Finished task" | grep -v "\[IMT\] Processed"

# Print number of tasks executed
NUMTASKS=`cat ${TESTNAME}.out | grep -e "\[IMT\] Finished task" | wc -l`
echo "NUM TASKS (tentrylist): $NUMTASKS"

# Print number of tasks that processed no entries
NUMETASKS=`cat ${TESTNAME}.out | grep -e "\[IMT\] Processed 0 entries" | wc -l`
echo "NUM EMPTY TASKS (tentrylist): $NUMETASKS" 

# Print number of tasks that processed exactly one entry
NUM1TASKS=`cat ${TESTNAME}.out | grep -e "\[IMT\] Processed 1 entries" | wc -l`
echo "NUM 1-ENTRY TASKS (tentrylist): $NUM1TASKS"

#### Friends case
./$TESTNAME $NTHREADS $INPUTFILE $TREENAME "friends" 1>${TESTNAME}.out 
