#!/bin/sh
#
# $Id: makedebdir.sh,v 1.11 2005/07/13 17:14:28 rdm Exp $
#
# Make the debian packaging directory 
#

### echo %%% possibly update the changelog file
root_vers=`cat build/version_number | tr '/' '.'` 
last_vers=`head -n 1 build/package/debian/changelog | sed 's/root (\(.*\)).*/\1/'`
root_lvers=`echo $root_vers | awk 'BEGIN {FS="."} {printf "%d", (($1 * 1000) + $2) * 1000 + $3}'`
last_lvers=`echo $last_vers | awk 'BEGIN {FS="."} {printf "%d", (($1 * 1000) + $2) * 1000 + $3}'`
if test $root_lvers -gt $last_lvers ; then 
    dch -v ${root_vers}-1 -c build/package/debian/changelog "New upstream version"
fi

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
