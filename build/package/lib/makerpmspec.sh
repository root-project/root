#!/bin/sh -e 
#
# $Id: makerpmspec.sh,v 1.3 2002/01/22 10:53:28 rdm Exp $
#
# Make the rpm spec file in ../root.spec
#
#
### echo %%% Some general variables 
base=root
cmndir=build/package/common
libdir=build/package/lib
rpmdir=build/package/rpm 
vrsfil=build/version_number
tgtdir=rpm
curdir=`pwd`

### echo %%% Packages ordered by preference
pkgs="task-root root-daemon root-ttf root-zebra root-gl root-mysql root-pgsql root-table root-shift root-cint root-bin libroot-dev libroot"
pkgs=`./configure linux --pkglist --enable-soversion --enable-table --enable-thread --enable-shared | sed -n 's,packages: ,,p'`
lvls="preinst postinst prerm postrm"

# ROOT version 
major=`sed 's|\(.*\)\..*/.*|\1|' < ${vrsfil}`
minor=`sed 's|.*\.\(.*\)/.*|\1|' < ${vrsfil}`
revis=`sed 's|.*\..*/\(.*\)|\1|' < ${vrsfil}`
versi="${major}.${minor}.${revis}"

### echo %%% Make the directory 
# mkdir -p ${tgtdir} 

### echo %%% Copy the README file to the directory 
cp ${cmndir}/README ${rpmdir}/README.Redhat

### echo %%% Copy the header of the spec file to rpm/root.spec
if [ ! -f ${rpmdir}/head.spec.in ] ; then 
    echo "$0: Couldn't find ${rpmdir}/head.spec.in - very bad" 
    echo "Giving up. Something is very screwy"
    exit 10
fi
### echo %%% make sure we've got a fresh file 
rm -f root.spec

### echo %%% First thing to do, is to write version number. 
sed "s/@version@/${versi}/" < ${rpmdir}/head.spec.in > root.spec
echo "" >> root.spec

### echo %%% Make the sub-package stuff
for i in ${pkgs} ; do 
    echo "Processing for package $i ... "

    ### echo %%% first insert the missing peices
    ${libdir}/makerpmspecs.sh $tgtdir $cmndir $rpmdir "$lvls" $i >> root.spec
done 

### echo %%% finally cat the footer to the spec file 
echo "" >> root.spec

### echo %%% Cat the tail of the file to the to rpm/root.spec
if [ ! -f ${rpmdir}/tail.spec.in ] ; then 
    echo "$0: Couldn't find ${rpmdir}/tail.spec.in - very bad" 
    echo "Giving up. Something is very screwy"
    exit 10
fi

### echo %%% and finally the last part 
sed -e "s|@libdir@|${libdir}|" \
    -e "s|@cmndir@|${cmndir}|" \
    -e "s|@tgtdir@|${tgtdir}|" \
    -e "s|@pkglist@|${pkgs}|" \
    < ${rpmdir}/tail.spec.in >> root.spec

#
# $Log: makerpmspec.sh,v $
# Revision 1.3  2002/01/22 10:53:28  rdm
# port to Debian distribution of GNU/Hurd by Christian Holm.
#
# Revision 1.2  2002/01/20 14:23:52  rdm
# Mega patch by Christian Holm concerning the configure, build and
# Debian and RedHat packaging scripts. The configure script has been
# rationalized (introduction of two shell functions to find package
# headers and libraries). Extensive update of the INSTALL writeup,
# including description of all new packages (SapDB, PgSql, etc.).
# More options to the root-config script. Man page for memprobe.
# Big overhaul of the Debian and RedHat packaging scripts, supporting
# the new libraries.
#
# Revision 1.1  2001/04/23 14:11:47  rdm
# part of the debian and redhat build system.
#
#
