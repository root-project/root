#!/bin/bash

ROOT=$1
WORKDIR=$(mktemp -d /tmp/testPerfJITDump_XXXXX)
OUTPUT=${WORKDIR}/stdout.log

cd ${WORKDIR}
CLING_PROFILE=1 perf record ${ROOT} -b -q -e "int testFuncForPerf() { return 42; }" -e "testFuncForPerf()" -e "return 0;" > ${OUTPUT}  || { echo "Could not run perf"; exit 127; }

grep -q 42 ${OUTPUT} || { echo "Root process did not write the expected result in ${OUTPUT}" >&2; exit 2; }

JITDUMP=$(perf inject -v -j -i perf.data -o perf_inject.data 2>&1 | grep "jit marker found")
JITDUMP=${JITDUMP#*: }

if [ -f "${JITDUMP}" ]; then
  grep 'testFuncForPerf' ${JITDUMP} || { echo "The test function was not exported to the jitdump file." >&2; exit 1; }
else
  echo "Could not find a jitdump file in ${JITDUMP}"; exit 1
fi
