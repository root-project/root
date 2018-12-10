#!/bin/sh

# Script to update base/inc/RVersion.h.
# Called by main Makefile as soon as build/version_number has been updated.
#
# Author: Fons Rademakers, 28/4/2000

ROOTEXE=$1/bin/root.exe
SCRIPT=build/version.cxx
VERSION=`cat build/version_number`

$ROOTEXE -q -b -l $SCRIPT || exit 1

build/unix/coreteam.sh rootx/src/rootcoreteam.h

if test "x`uname | grep -i cygwin`" != "x"; then
    dos2unix core/base/inc/RVersion.h
    dos2unix rootx/src/rootcoreteam.h
fi

echo "Committing changes."
git commit core/base/inc/RVersion.h rootx/src/rootcoreteam.h build/version_number documentation/doxygen/Doxyfile \
  -m "Update ROOT version files to v$VERSION." || exit 1

echo "Update also doc/vXXX/index.html to $VERSION."
echo ""
echo "New version is $VERSION."
echo "See https://root.cern/release-checklist for the next steps,"
echo "for instance tagging if this is a release."
