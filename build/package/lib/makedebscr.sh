#!/bin/sh -e 
#
# $Id: makedebscr.sh,v 1.2 2002/01/20 14:23:52 rdm Exp $
#
# Writes a script files to debian/<pkg>.<scr> 
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

# test to see if full file exist, if so, cat it to standard out, doing
# substitutions   
if [ -f ${debdir}/${pkg}.${scr}.in ] 
then 
    sed -e "s,@prefix@,/${prefix},g" \
	-e "s,@etcdir@,/${etcdir},g" \
	-e "s,@docdir@,/${docdir},g" \
	< ${debdir}/${pkg}.${scr}.in > ${tgtdir}/${pkg}.${scr}
fi

#
# EOF
#
