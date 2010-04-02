#!/bin/sh

result=$1
logfile=$2
testname=$3
if [ "x$logfile" != "x" ] ; then
  cat $2
  if [ "x$testname" != "x" ] ; then
     echo "'root.exe -b -l -q $testname' exited with error code: $result" >> $logfile
  fi
fi
exit $result
