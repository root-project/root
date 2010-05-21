#!/bin/sh
# $Id$

# Script to produce source distribution and optionally binary distribution
# for VMC packages
# Called by main Makefile.
#
# Usage: fl_makedist.sh -p package [OPTIONS]
#           -p package        package name: geant3, geant4_vmc, fluka_vmc
#           -c gcc_version    gcc version 
#           -b                make binary distribution 
#           -i                make installation in the path defined in dedicated
#                             environment variable  
#
# By I.Hrivnacova, 01/04/2010

#set -x

CURDIR=`pwd`

# default options
MAKETAR=1
BINDIST=0
INSTALL=0
GCC_VERS=""
PREFIX=""

while getopts "bic:p:" option
do
  case $option in
    p ) PACKAGE=$OPTARG;;
    c ) GCC_VERS=$OPTARG;;
    b ) BINDIST=1
        MAKETAR=1
        ;;
    i ) INSTALL=1
        MAKETAR=0
        ;;
    * ) echo "Unimplemented option chosen."
        echo "Usage:"
        echo "makedist.sh  -p package [OPTIONS]"
        echo "   -p package        package name: geant3, geant4_vmc, fluka_vmc" 
        echo "   -c gcc_version    gcc version"  
        echo "   -b                make binary distribution"  
        echo "   -i                make installation in path defined in dedicated"
        echo "                     environment variable" 
        EXIT=1
        ;;
  esac
done

#
#  Packages specific definitions:
#  run directory, input files, installation destination
#

# geant321_+vmc
#
if [ "x$PACKAGE" = "xgeant321" ]; then
  RUNDIR=$CURDIR/..
  PREFIX=$G3VMC_INSTALL
  DIRNAME="geant3"
  # special extension for distribution file name
  NAMEEXT="21+_vmc" 
  BINFILES="geant3/lib"
  cd $RUNDIR
  MACHINE=`root-config --arch`
  SOURCES=`ls geant3`
  SOURCES=`echo $SOURCES | sed s/"tgt_$MACHINE"//g`
  SOURCES=`echo $SOURCES | sed s/lib//g`
  for param in $SOURCES; do
    TARFILES="$TARFILES geant3/$param"
  done
fi        

# geant4_vmc
#
if [ "x$PACKAGE" = "xgeant4vmc" ]; then
   RUNDIR=$CURDIR/../..
   PREFIX=$G4VMC_INSTALL
   DIRNAME="geant4_vmc"
   NAMEEXT=""
   BINFILES="geant4_vmc/include geant4_vmc/lib"
   TARFILES="geant4_vmc/README geant4_vmc/LICENSE geant4_vmc/Makefile \
             geant4_vmc/"history" geant4_vmc/Geant4VMC.html geant4_vmc/version_number  \
             geant4_vmc/g4root geant4_vmc/"source" geant4_vmc/examples"
fi

# fluka_vmc
#
if [ "x$PACKAGE" = "xfluka" ]; then
   RUNDIR=$CURDIR/../..
   PREFIX=$FLVMC_INSTALL
   DIRNAME="fluka_vmc"
   NAMEEXT=""
   BINFILES="fluka_vmc/lib"
   TARFILES="fluka_vmc/README fluka_vmc/"history" fluka_vmc/version_number  \
             fluka_vmc/input fluka_vmc/"source""
fi

# include binary files if selected
#
if [ "$BINDIST" = "1" ]; then
  TARFILES="$TARFILES $BINFILES"
fi  

# go to run directory
#
cd $RUNDIR

# make tar ball
#
if [ "$MAKETAR" = "1" ]; then
  VERSION=`cat $DIRNAME/version_number`
  MACHINE=`root-config --arch`
  TYPE=""
  if [ "$BINDIST" = "1" ]; then
     if [ "x$GCC_VERS" = "x" ]; then  
        TYPE=$MACHINE.
     else
        TYPE=$MACHINE.$GCC_VERS.
     fi
  fi  
  TARFILE=$DIRNAME$NAMEEXT.$VERSION.$TYPE"tar"

  TAR="tar zcvf"
  rm -f $TARFILE.gz
  TARFILE=$TARFILE".gz"
  EXCLUDE="--exclude .svn"

  $TAR $TARFILE $EXCLUDE $TARFILES
fi

# make installation
#
if [ "$INSTALL" = "1" ]; then
  if [ "x$PREFIX" = "x" ]; then
    echo "Destination directory $PREFIX not defined"        
    EXIT=1
  fi  
  if [ ! -d $PREFIX ]; then
    echo "Destination directory $PREFIX not found" 
    EXIT=1
  fi  
  echo "Installing $DIRNAME in $PREFIX ..." 
  cp -r $BINFILES $PREFIX
fi   

cd $CURDIR

exit 0
