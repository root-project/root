#!/bin/sh 
#
# $Id$
#
# Write lines for <pkg> to debian/shlibs.local
#
. build/package/lib/common.sh debian

if [ $# -lt 1 ] ; then 
    echo "$0: I need a package name - giving up"
    exit 2
fi

### echo %%% save package name in logical variable 
pkg=$1

### echo %%% save major.minor version number in short name
v=${major}.${minor}

if [ -f ${cmndir}/${pkg}.shlibs ] ; then 
    p=${pkg} 
    sed -e "/^#.*/d" \
        -e "/^[ \t]*$/d" \
        -e "s|^usr/lib/root/\(lib.*\)\.so.*|\1 $v $p (>= $versi)|" \
        <  ${cmndir}/${pkg}.shlibs >> ${tgtdir}/shlibs.local
fi

#
# $Log$
#
