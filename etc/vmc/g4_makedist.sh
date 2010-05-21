#!/bin/sh
# $Id$

# Script to produce source and optionally binary distribution of geant4_vmc.
# Called by main Makefile.
#
# According to: 
# $ROOTSYS/build/unix/makedist.sh
# Author: Fons Rademakers, 29/2/2000
#
# Usage: g4_makedist.sh [gcc_version] [lib]
#
# By I.Hrivnacova, 7/10/2002

CURDIR=`pwd`

# gmake is called from geant4_vmc/source
cd ../..

if [ "x$1" = "xlib" ]; then
   GCC_VERS=""
   MAKELIB="geant4_vmc/include geant4_vmc/lib"
elif [ "x$2" = "xlib" ]; then
   GCC_VERS=$1
   MAKELIB="geant4_vmc/include geant4_vmc/lib"
else
   GCC_VERS=$1
fi
VERSION=`cat geant4_vmc/version_number`
MACHINE=`root-config --arch`
if [ "x$MAKELIB" = "xgeant4_vmc/lib" ]; then
   if [ "x$GCC_VERS" = "x" ]; then  
      TYPE=$MACHINE.
   else
      TYPE=$MACHINE.$GCC_VERS.
   fi
else   
  TYPE=""
fi  
TARFILE=geant4_vmc.$VERSION.$TYPE"tar"

TAR="tar zcvf"
rm -f $TARFILE.gz
TARFILE=$TARFILE".gz"
EXCLUDE="--exclude .svn"

$TAR $TARFILE $EXCLUDE geant4_vmc/README geant4_vmc/LICENSE geant4_vmc/Makefile \
   geant4_vmc/"history" geant4_vmc/Geant4VMC.html geant4_vmc/version_number  \
   geant4_vmc/g4root geant4_vmc/"source" geant4_vmc/examples $MAKELIB

cd $CURDIR

exit 0
