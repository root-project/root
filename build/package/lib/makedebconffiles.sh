#!/bin/sh -e 
#
# $Id: makedebconffiles.sh,v 1.1 2001/04/23 14:11:47 rdm Exp $
#
# Writes a conffiles file entry to debian/<pkg>.confiles
#
tgtdir=$1 ; shift 
cmndir=$1 ; shift 
etcdir=$1 ; shift
pkg=$1

# Make sure we get a fresh file 
rm -f ${tgtdir}/${pkg}.conffiles

# See if file exists in common directory 
if [ -f $cmndir/$pkg.conffiles ] ; then 
    # Prepend each line with a '/', and ignore comment lines
    grep -v "^#" $cmndir/$pkg.conffiles | \
	sed -e "s|@etcdir@|/${etcdir}|"  > ${tgtdir}/${pkg}.conffiles
fi 

#
# $Log: makedebconffiles.sh,v $
# Revision 1.1  2001/04/23 14:11:47  rdm
# part of the debian and redhat build system.
#
#
