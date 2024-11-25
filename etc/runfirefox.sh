#!/bin/bash

dumpfile=$1
shift

profile=$1
shift

firefox=$1
shift

args=

while [ "$1" != "" ]; do
   args="$args $1"
   shift
done

if [ "$dumpfile" == "__nodump__" ]; then
   # Helper script to run firefox and delete temporary profile at exit

   if [ "$profile" != "__dummy__" ]; then
      trap "rm -rf $profile" 0 1 2 3 6
   fi

   $firefox $args >/dev/null 2>/dev/null
else
   # Helper script to run headlerss firefox and kill it when dump is ready

   rm -f $dumpfile

   $firefox $args 1>$dumpfile 2>&1 &

   PID=$!

   res=""
   # wait maximal 30 seconds
   cnt=300
   while [ -z "$res" ] && [[ $cnt -gt 0 ]]
   do
      sleep 0.1
      res=`tail --lines=10 $dumpfile | grep "###batch###job###done###"`
      cnt=$((cnt - 1))
   done

   kill $PID 2>/dev/null

   if [ "$profile" != "__dummy__" ]; then
      rm -rf $profile
   fi

fi
