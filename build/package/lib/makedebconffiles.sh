#!/bin/sh -e 
#
# $Id$
#
# Writes a conffiles file entry to debian/<pkg>.confiles
#
. build/package/lib/common.sh debian

if [ $# -lt 1 ] ; then 
    echo "$0: I need a package name - giving up"
    exit 2
fi

# save package name in logical variable 
pkg=$1

# Make sure we get a fresh file 
rm -f ${tgtdir}/${pkg}.conffiles

# See if file exists in common directory 
if [ -f $cmndir/$pkg.conffiles ] ; then 
    # Prepend each line with a '/', and ignore comment lines
    grep -v "^#" $cmndir/$pkg.conffiles | \
	sed 's|^\([^/]\)|/\1|'  > ${tgtdir}/${pkg}.conffiles
fi 

#
# $Log$
#
