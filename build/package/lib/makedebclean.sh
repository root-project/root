#!/bin/sh -e 
#
# $Id$
#
# Make the debian packaging directory 
#
. build/package/lib/common.sh debian 

# Make the directory 
rm -rf ${tgtdir} 
rm -f build-stamp  

#
# $Log$
#
