#!/bin/sh -e 
#
# $Id$
#
# Make the debian packaging directory 
#
. build/package/lib/common.sh rpm

# Make the directory 
rm -rf ${tgtdir} 
rm -f  ${updir}/${base}.spec ${updir}/${base}.rpmrc ${updir}/${base}.rpmmac
#
# $Log$
#
