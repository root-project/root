#!/bin/sh
# $Id$

# Script to produce source distribution and optionally binary distribution fluka_vmc.
# Called by main Makefile.
#
# According to: 
# $ROOTSYS/build/unix/makedist.sh
# Author: Fons Rademakers, 29/2/2000
#
# Usage: makedist.sh [gcc_version] [lib]
#
# By I.Hrivnacova, 12/12/2007

CURDIR=`pwd`

# gmake is called from fluka_vmc/source
cd ../..

if [ "x$1" = "xlib" ]; then
   GCC_VERS=""
   MAKELIB="fluka_vmc/lib"
elif [ "x$2" = "xlib" ]; then
   GCC_VERS=$1
   MAKELIB="fluka_vmc/lib"
else
   GCC_VERS=$1
fi
VERSION=`cat fluka_vmc/version_number`
MACHINE=`root-config --arch`
if [ "x$MAKELIB" = "xfluka_vmc/lib" ]; then
   if [ "x$GCC_VERS" = "x" ]; then  
      TYPE=$MACHINE.
   else
      TYPE=$MACHINE.$GCC_VERS.
   fi
else   
  TYPE=""
fi  
TARFILE=fluka_vmc.$VERSION.$TYPE"tar"

TAR=`which gtar`
dum=`echo $TAR | grep "no gtar"`
stat=$?
if [ "$TAR" = '' ] || [ $stat = 0 ]; then
   TAR="tar cvf"
   rm -f $TARFILE.gz
   EXCLUDE=
else 
   TAR=$TAR" zcvf"
   rm -f $TARFILE.gz
   TARFILE=$TARFILE".gz"
   EXCLUDE="--exclude .svn"
fi

$TAR $TARFILE $EXCLUDE fluka_vmc/README fluka_vmc/"history" fluka_vmc/version_number  \
   fluka_vmc/input fluka_vmc/"source" $MAKELIB
cd $CURDIR

exit 0
