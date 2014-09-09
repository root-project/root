#! /bin/sh

# Script to update base/inc/RVersion.h.
# Called by main Makefile as soon as build/version_number has been updated.
#
# Author: Fons Rademakers, 28/4/2000

CINT=$1/bin/cint_tmp
SCRIPT=build/version.cxx
VERSION=`cat build/version_number`

build/unix/gitinfo.sh
$CINT $SCRIPT

if test "x`uname | grep -i cygwin`" != "x"; then
    dos2unix core/base/inc/RVersion.h
fi

echo "Committing changes."
git commit core/base/inc/RVersion.h build/version_number \
  -m "Update ROOT version files to v$VERSION." || exit 1

echo "Update also doc/vXXX/index.html to $VERSION."
echo ""
echo "New version is $VERSION."
echo "See http://root.cern.ch/drupal/howtorelease for the next steps,"
echo "for instance tagging if this is a release."
