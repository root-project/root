#!/bin/sh -e 
#
# $Id$
#
# Writes an examples file entry to debian/root-<pkg>.examples
#
. build/package/lib/common.sh debian

if [ $# -lt 1 ] ; then 
    echo "$0: I need a package name - giving up"
    exit 2
fi

# save package name in logical variable 
pkg=$1

# Make sure we get a fresh file 
rm -f ${tgtdir}/${pkg}.examples

# See if file exists in common directory 
if [ -f $cmndir/$pkg.examples ] ; then 
    # Prepend each line with a '/', and ignore comment lines
    grep -v "^#" $cmndir/$pkg.examples > ${tgtdir}/${pkg}.examples
fi 

#
# $Log$
#
