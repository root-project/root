#! /bin/sh

# Script to update base/inc/RVersion.h.
# Called by main Makefile as soon as build/version_number has been updated.
#
# Author: Fons Rademakers, 28/4/2000

CINT=cint/main/cint_tmp
SCRIPT=build/version.cxx

$CINT $SCRIPT

echo "New version is `cat build/version_number`. Updating dependencies..."

# compile all files that were out-of-date prior to makeversion.sh
make -o base/inc/RVersion.h

# touch all files that don't need recompilation (need to do this 3 times
# to walk through chain of dependencies)
make -s -t; make -s -t; make -s -t

# recompile only base/src/TROOT.cxx
touch base/src/TROOT.cxx
make

echo "root-config --version reports: `bin/root-config --prefix=. --version`"
