#!/bin/sh -e 
#
# $Id$
#
# Writes a documentation file entry to debian/root-<pkg>.docs
#
. build/package/lib/common.sh debian

if [ $# -lt 1 ] ; then 
    echo "$0: I need a package name - giving up"
    exit 2
fi

# save package name in logical variable 
pkg=$1

# Make sure we get a fresh file 
rm -f ${tgtdir}/${pkg}.docs

# See if file exists in common directory 
if [ -f $cmndir/$pkg.docs ] ; then 
    # Prepend each line with a '/', and ignore comment lines
    grep -v "^#" $cmndir/$pkg.docs > ${tgtdir}/${pkg}.docs
fi 

#
# $Log$
#
