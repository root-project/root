#!/bin/sh 
#
# $Id: makedebshlocal.sh,v 1.1 2001/04/23 14:11:47 rdm Exp $
#
# Write lines for <pkg> to debian/shlibs.local
#
tgtdir=$1 ; shift 
debdir=$1 ; shift
cmndir=$1 ; shift 
versi=$1  ; shift
major=$1  ; shift
minor=$1  ; shift 
pkg=$1

### echo %%% save package name in logical variable 
pkg=$1

### echo %%% save major.minor version number in short name
v=${major}.${minor}

if [ -f ${cmndir}/${pkg}.shlibs ] ; then 
    p=${pkg} 
    sed -e "/^#.*/d" \
        -e "/^[ \t]*$/d" \
        -e "s|^@prefix@/lib/root/\(lib.*\)\.so.*|\1 $v $p (>= $versi)|" \
        <  ${cmndir}/${pkg}.shlibs >> ${tgtdir}/shlibs.local
fi

#
# $Log: makedebshlocal.sh,v $
# Revision 1.1  2001/04/23 14:11:47  rdm
# part of the debian and redhat build system.
#
#
