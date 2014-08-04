#!/bin/bash

nprocess=$1
nhist=$2
ndims=$3
nbins=$4

usage() {
   echo "parallelMergeTest.sh nprocess nhist [ndims] [nbins]"
}

if [ "x$nprocess" == "x" ] ; then
  echo "Must specify the number of client process."
  usage
  exit 1;
fi

if [ "x$nhist" == "x" ] ; then
  echo "Must specify the number of histograms."
  usage
  exit 1;
fi

if [ "x$ndims" == "x" ] ; then
   ndims=1
fi

if [ "x$nbins" == "x" ] ; then
   nbins=100
fi

# make sure the script is compiled
echo '.L  parallelMergeTest.C+' | root.exe -b -l
res=$?
if [ $res -ne 0 ] ; then
  exit $res;
fi

echo '.L  parallelMergeServer.C+' | root.exe -b -l
res=$?
if [ $res -ne 0 ] ; then
  exit $res;
fi

root.exe -b -l -q $ROOTSYS/tutorials/net/parallelMergeServer.C+ &
res=$?
if [ $res -ne 0 ] ; then
  exit $res;
fi
# give sometimes to start
sleep 2

if type seq > /dev/null 2>&1 ; then
   sequence=`seq ${nprocess}`
elif type jot > /dev/null 2>&1 ; then
   sequence=`jot ${nprocess} 1`
else
   echo "Neither seq nor jot is available, we can't count"
   exit 1
fi

for index in ${sequence}
do
  root.exe -b -l -q "parallelMergeTest.C+($nhist,$ndims,$nbins)" & # > output.$index &
done

wait
