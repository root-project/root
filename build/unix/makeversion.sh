#! /bin/sh

# Script to update base/inc/RVersion.h.
# Called by main Makefile as soon as build/version_number has been updated.
#
# Author: Fons Rademakers, 28/4/2000

ROOTEXE=bin/root.exe
SCRIPT=build/version.cxx
CORETEAM=build/unix/git_coreteam.py

$ROOTEXE -q -b -l $SCRIPT

ncpu=`bin/root-config --ncpu`

python $CORETEAM
if [ "$?" -eq "0" ] ; then
   mv rootcoreteam.h rootx/src/rootcoreteam.h
fi

if test "x`uname | grep -i cygwin`" != "x"; then
    echo 'Need to run "dos2unix base/inc/RVersion.h"'
    dos2unix core/base/inc/RVersion.h
    echo 'Need to run "dos2unix "rootx/src/rootcoreteam.h"'
    dos2unix rootx/src/rootcoreteam.h
fi

echo "Update also doc/vXXX/index.html to `cat build/version_number`."
echo ""
echo "New version is `cat build/version_number`. Updating dependencies..."

# compile all files that were out-of-date prior to makeversion.sh
make -j $ncpu -o core/base/inc/RVersion.h

# touch all files that don't need recompilation (need to do this 3 times
# to walk through chain of dependencies)
make -j $ncpu -s -t; make -j $ncpu -s -t; make -j $ncpu -s -t

# recompile only core/base/src/TROOT.cxx
touch core/base/src/TROOT.cxx
touch core/base/inc/TVersionCheck.h
touch rootx/src/rootxx.cxx
make -j $ncpu

echo "root-config --version reports: `bin/root-config --prefix=. --version`"
