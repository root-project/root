#!/bin/sh 
#
# $Id$
#
# Common variables and such
#

#
# Some general variables 
#
base=root
cmndir=build/package/common
libdir=build/package/lib
rpmdir=build/package/rpm 
debdir=build/package/debian
vrsfil=build/version_number
curdir=`pwd`
updir=`dirname $curdir` 

if [ "x$1" = "xdebian" ] ; then 
    tgtdir="debian"
    arch="linuxdeb2"
elif [ "x$1" = "xrpm" ] ; then 
    tgtdir="rpm" 
    arch="linuxegcs"
else 
    echo "Unknown package system '$1'. Must be one of debian or rpm"
    exit 1
fi

#
# Installation directories
#
etcdir=/etc/root
prefix=/usr
docdir=/usr/share/doc/root
mandir=/usr/share/man/man1
cintdir=/usr/share/root/cint

#
# Packages ordered by preference
#
pkgs="task-root root-daemon root-ttf root-zebra root-gl root-mysql root-pythia root-star root-shift root-cint root-bin libroot-dev libroot"
lvls="preinst postinst prerm postrm"

#
# ROOT version 
#
major=`sed 's|\(.*\)\..*/.*|\1|' < ${vrsfil}`
minor=`sed 's|.*\.\(.*\)/.*|\1|' < ${vrsfil}`
revis=`sed 's|.*\..*/\(.*\)|\1|' < ${vrsfil}`
versi="${major}.${minor}.${revis}"

#
# $Log$
#
