#!/bin/sh -e 
#
# $Id$
#
# Writes a script files to debian/root-<pkg>.<scr> 
#
. build/package/lib/common.sh debian

if [ $# -lt 2 ] ; then 
    echo "$0: I need a package and script name - giving up"
    exit 2
fi

# save package name in logical variable 
pkg=$1
scr=$2

# make sure we get a fresh file 
rm -f ${tgtdir}/{pkg}.${scr}

# test to see if full file exist, if so, cat it to standard out. 
if [ -f ${debdir}/${pkg}.${scr} ] 
then 
    cp ${debdir}/${pkg}.${scr} ${tgtdir}/${pkg}.${scr}
# if a skeleton and body exist, we insert the body into the skeleton 
# and cat it to standard out 
elif [ -f ${cmndir}/${pkg}.${scr} ] && [ -f ${debdir}/${pkg}.${scr}.in ] 
then 

    # first split the file at mark '@<scr>@'
    csplit -q -f ${cmndir}/tmp. \
	-k ${debdir}/${pkg}.${scr}.in "/@${scr}@/"

    # Then output the full file 
    cat ${cmndir}/tmp.00 \
	${cmndir}/${pkg}.${scr} \
	${cmndir}/tmp.01 | \
	sed "/${scr}/d"  > ${tgtdir}/${pkg}.${scr}

    # Clean up
    rm -f ${cmndir}/tmp.00 ${cmndir}/tmp.01
fi

#
# $Log$
#
