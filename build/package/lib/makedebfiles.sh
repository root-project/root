#!/bin/sh -e 
#
# $Id$
#
# Writes a files file entry to standard debian/root-<pkg>.files
#
. build/package/lib/common.sh debian

if [ $# -lt 1 ] ; then 
    echo "$0: I need a package name - giving up"
    exit 2
fi

# save package name in logical variable 
pkg=$1

# List of file types to put into files
types="shlibs conffiles files" 

# Make sure we get a fresh file 
rm -f ${tgtdir}/${pkg}.files

# for each type convert to one line
for i in $types ; do 
    # Check if we've got shared libraries for this package 
    if [ -f $cmndir/$pkg.$i ] ; then 
	grep -v -e "^#" $cmndir/$pkg.$i | \
	    sed 's/^\/\(.*\)/\1/' | \
	    tr "\n" " " >> ${tgtdir}/${pkg}.files
    fi 
done 

#
# $Log$
#
