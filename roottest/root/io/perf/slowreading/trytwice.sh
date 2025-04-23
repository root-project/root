#!/bin/sh

callroot="${1}"
library=$2
file=$3
log=$4
firstlog=first.attempt.$log

if [ "x$CMDECHO" != "x@" ] ; then set -x; fi

eval "$callroot" -q -b -l '"Read.C(\"$library\", \"$file\", 0)"' > $log 2>&1 || ( mv $log $firstlog ; sleep 5; eval "$callroot" -q -b -l '"Read.C(\"$library\", \"$file\", 0)"'  > $log 2>&1 )
result=$?
if [ $result -gt 0 ] ; then
   if [ -e $firstlog ] ; then 
      cat $firstlog
   fi
   cat $log
fi
exit $result
