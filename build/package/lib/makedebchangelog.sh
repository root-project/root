#!/bin/sh -e 
#
# $Id: makedebchangelog.sh,v 1.1 2001/04/23 14:11:47 rdm Exp $
#
# Write a changelog file 
#
tgtdir=$1 ; shift
debdir=$1 ; shift 
versi=$1

cp ${debdir}/changelog ${tgtdir}

if `grep -q "root ($versi" ${tgtdir}/changelog` ; then 
    echo "Not a new ROOT version"
else
    dch -v ${versi} "Bumped Debian GNU/Linux version with ROOT"
    cp ${tgtdir}/changelog ${debdir}/
fi

#
# $Log: makedebchangelog.sh,v $
# Revision 1.1  2001/04/23 14:11:47  rdm
# part of the debian and redhat build system.
#
#
