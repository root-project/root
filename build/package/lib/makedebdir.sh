#!/bin/sh
#
# $Id: makedebdir.sh,v 1.8 2004/12/06 18:32:36 rdm Exp $
#
# Make the debian packaging directory 
#

### echo %%% Make the directory 
mkdir -p debian

### echo %%% Copy files to directory, making subsitutions if needed
cp -a build/package/debian/* debian/
chmod a+x debian/rules 
chmod a+x build/package/lib/*

### echo %%% Make skeleton control file 
# cat build/package/debian/control.in build/package/common/*.control \
#     > debian/control.in 

#
# EOF
#
