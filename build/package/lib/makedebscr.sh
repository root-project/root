#!/bin/sh -e 
#
# $Id: makedebscr.sh,v 1.1 2001/04/23 14:11:47 rdm Exp $
#
# Writes a script files to debian/root-<pkg>.<scr> 
#

# save package name in logical variable 
tgtdir=$1 ; shift
debdir=$1 ; shift
cmndir=$1 ; shift 
prefix=$1 ; shift
etcdir=$1 ; shift 
docdir=$1 ; shift 
pkg=$1    ; shift
scr=$1    ; shift

# make sure we get a fresh file 
rm -f ${tgtdir}/{pkg}.${scr}

# test to see if full file exist, if so, cat it to standard out. 
if [ -f ${debdir}/${pkg}.${scr} ] 
then 
    sed -e "s,@prefix@,/${prefix},g" \
	-e "s,@etcdir@,/${etcdir},g" \
	-e "s,@docdir@,/${docdir},g" \
	< ${debdir}/${pkg}.${scr} > ${tgtdir}/${pkg}.${scr}

# if a skeleton and body exist, we insert the body into the skeleton 
# and cat it to standard out 
elif [ -f ${cmndir}/${pkg}.${scr} ] && [ -f ${debdir}/${pkg}.${scr}.in ] 
then 

    # first split the file at mark '@<scr>@'
    csplit -q -f ${cmndir}/tmp. \
	-k ${debdir}/${pkg}.${scr}.in "/@${scr}@/"

    # Then output the skeleton full file to temporary
    cat ${cmndir}/tmp.00 \
	${cmndir}/${pkg}.${scr} \
	${cmndir}/tmp.01 | \
	sed "/${scr}/d"  > ${cmndir}/tmp.02 

    # Do expansion on file 
    sed -e "s,@prefix@,/${prefix},g" \
	-e "s,@etcdir@,/${etcdir},g" \
	< ${cmndir}/tmp.02 > ${tgtdir}/${pkg}.${scr}

    # Clean up
    rm -f ${cmndir}/tmp.00 ${cmndir}/tmp.01 ${cmndir}/tmp.02
fi

#
# $Log: makedebscr.sh,v $
# Revision 1.1  2001/04/23 14:11:47  rdm
# part of the debian and redhat build system.
#
#
