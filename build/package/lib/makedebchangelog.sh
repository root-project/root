#!/bin/sh -e 
#
# $Id$
#
# Write a changelog file 
#
. build/package/lib/common.sh debian 

cp ${debdir}/changelog ${tgtdir}

if `grep -q "root ($versi" ${tgtdir}/changelog` ; then 
    echo "Not a new ROOT version"
else
    dch -v ${versi} "Bumped Debian GNU/Linux version with ROOT"
    cp ${tgtdir}/changelog ${debdir}/
fi

#
# $Log$
#
