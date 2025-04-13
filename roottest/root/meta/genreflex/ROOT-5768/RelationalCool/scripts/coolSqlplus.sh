#!/usr/bin/env bash
# See https://twiki.cern.ch/twiki/bin/view/PSSGroup/RlWrap
if [ "${SCRAM_ARCH:0:5}" = "win32" ]; then
    cmd=`basename $0`
    echo "ERROR! $cmd is not supported for $SCRAM_ARCH: use sqlplus instead"
    exit 1
elif [ "${CMTCONFIG:0:5}" = "win32" ]; then
    cmd=`basename $0`
    echo "ERROR! $cmd is not supported for $CMTCONFIG: use sqlplus instead"
    exit 1
else
    rlwrap -h > /dev/null 2&>1
    if [ $? == 0 ]; then
        rlwrap \sqlplus $@
    else
        \sqlplus $@
    fi
fi

