#!/bin/bash

OUTPUT=$(mktemp /tmp/testPerfMap_XXXXX)
CLING_PROFILE=1 root.exe -b -q -e "int testFunc42() { return 42; }" -e "testFunc42()" > ${OUTPUT} &
PERF_MAP=/tmp/perf-${!}.map

wait

grep -q 42 ${OUTPUT} || { echo "Root process did not return the expected result in ${OUTPUT}" >&2; exit 1; }
test -f ${PERF_MAP} || { echo "No perf map found in ${PERF_MAP}" >&2; exit 1; }

if [ -f ${PERF_MAP} ]; then
  c++filt < ${PERF_MAP} | grep 'testFunc42\(\)' || { echo "The test function was not exported to the perf map." >&2; exit 2; }
fi
