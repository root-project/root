#!/bin/sh -e 
#
# $Id: makedebfiles.sh,v 1.1 2001/04/23 14:11:47 rdm Exp $
#
# Writes a files file entry to standard debian/root-<pkg>.files
#
tgtdir=$1 ; shift 
cmndir=$1 ; shift
prefix=$1 ; shift 
etcdir=$1 ; shift 
docdir=$1 ; shift 
pkg=$1

### echo %%% List of file types to put into files 
types="shlibs conffiles files" 

### echo %%% Make sure we get a fresh file 
rm -f ${tgtdir}/${pkg}.files

### echo %%% for each type convert to one line
for i in $types ; do 
    ### echo %%% Check if we have got a $i file for this $pkg in $cmndir
    if test -f $cmndir/$pkg.$i  ; then 
	### echo %%% Got $cmndir/$pkg.$i
	grep -v -e "^#" $cmndir/$pkg.$i | \
	    sed -e "s,@prefix@,${prefix},g" \
		-e "s,@etcdir@,${etcdir},g" \
		-e "s,@docdir@,${docdir},g" | \
	    tr "\n" " " >> ${tgtdir}/${pkg}.files
    fi 
done 

#
# $Log: makedebfiles.sh,v $
# Revision 1.1  2001/04/23 14:11:47  rdm
# part of the debian and redhat build system.
#
#
