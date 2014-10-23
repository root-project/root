#!/bin/sh
#
# Build a pch for the headers and linkdefs in root-build-dir/etc/dictpch/.
# root-build-dir is first tried as ./ - if that doesn't exist, $ROOTSYS
# is taken as root-build-dir.
#
# $1: PCH output file name
# $2: cxxflags (optional; required if extra headers are supplied)
# $3: extra headers to be included in the PCH (optional)
#
# exit code 1 for invocation errors; else exit code of rootcling invocation.
#
# Copyright (c) 2014 Rene Brun and Fons Rademakers
# Author: Axel Naumann <axel@cern.ch>, 2014-10-16

rootdir=.
cfgdir=etc/dictpch
allheaders=$cfgdir/allHeaders.h
alllinkdefs=$cfgdir/allLinkdefs.h
cppflags=$cfgdir/allCppflags.txt
pch=$1
shift

if [ "x$pch" = "x" ]; then
    echo 'Output PCH file name must be passed as first argument!' >& 2
    exit 1
fi

if ! [ -f $rootdir/$allheaders ]; then
    rootdir=$ROOTSYS
    if ! [ -f $rootdir/$allheaders ]; then
        echo 'Neither ./'$allheaders' nor $ROOTSYS/'$allheaders' exists!' >& 2
        exit 1
    fi
fi

cxxflags="-D__CLING__ -D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS -I$rootdir/include -I$rootdir/etc -I$rootdir/etc/cling `cat $rootdir/$cppflags`"

if ! [ "x$1" = "x" ]; then
    cxxflags="$cxxflags $1"
fi

# generate pch
touch allDict.cxx.h
$rootdir/bin/rootcling -1 -f allDict.cxx -noDictSelection -c $cxxflags $allheaders $@ $alllinkdefs
res=$?
if [ $res -eq 0 ] ; then
  mv allDict_rdict.pch $pch
  res=$?
fi
rm -f allDict.*

exit $res
