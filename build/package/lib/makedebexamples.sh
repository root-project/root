#!/bin/sh -e 
#
# $Id: makedebexamples.sh,v 1.1 2001/04/23 14:11:47 rdm Exp $
#
# Writes an examples file entry to debian/root-<pkg>.examples
#
tgtdir=$1 ; shift 
cmndir=$1 ; shift 
prefix=$1 ; shift
etcdir=$1 ; shift
pkg=$1

# Make sure we get a fresh file 
rm -f ${tgtdir}/${pkg}.examples

# See if file exists in common directory 
if [ -f $cmndir/$pkg.examples ] ; then 
    # Prepend each line with a '/', and ignore comment lines
    grep -v "^#" $cmndir/$pkg.examples | \
	sed -e "s,@etcdir@,${etcdir},g" \
	    -e "s,@prefix@,${prefix},g" \
		> ${tgtdir}/${pkg}.examples
fi 

#
# $Log: makedebexamples.sh,v $
# Revision 1.1  2001/04/23 14:11:47  rdm
# part of the debian and redhat build system.
#
#
