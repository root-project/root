#!/bin/sh -e

TESTNAME=test_progressiveCSV

# Run test
ROOTDEBUG=1 ./$TESTNAME 1>${TESTNAME}.out 2>${TESTNAME}.err

# Print only messages about lines being read from CSV file
cat ${TESTNAME}.out | grep "Total num lines"
cat ${TESTNAME}.err | grep "GetEntryRanges"

