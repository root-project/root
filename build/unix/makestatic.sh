#! /bin/sh

# Script to generate statically linked ROOT executables.
# Called by main Makefile.
#
# Author: Fons Rademakers, 21/01/2001

PLATFORM=$1
CXX=$2
CC=$3
LD=$4
LDFLAGS=$5
XLIBS=$6
SYSLIBS=$7
EXTRALIBS=$8
STATICOBJECTLIST=$9

ROOTALIB=lib/libRoot.a
ROOTAEXE=bin/roota
PROOFAEXE=bin/proofserva

rm -f $ROOTAEXE $PROOFAEXE

gobjs=`$STATICOBJECTLIST -d`

# If linking with Cocoa framework, then don't use XLIBS
if echo $EXTRALIBS | grep ' Cocoa' > /dev/null 2>& 1 ; then
    XLIBS=
fi

echo "Making $ROOTAEXE..."
echo $LD $LDFLAGS -o $ROOTAEXE main/src/rmain.o $gobjs $ROOTALIB \
   $XLIBS $SYSLIBS $EXTRALIBS
$LD $LDFLAGS -o $ROOTAEXE main/src/rmain.o $gobjs $ROOTALIB \
   $XLIBS $SYSLIBS $EXTRALIBS

linkstat=$?
if [ $linkstat -ne 0 ]; then
   exit $linkstat
fi

echo "Making $PROOFAEXE..."
echo $LD $LDFLAGS -o $PROOFAEXE main/src/pmain.o $gobjs $ROOTALIB \
   $XLIBS $SYSLIBS $EXTRALIBS
$LD $LDFLAGS -o $PROOFAEXE main/src/pmain.o $gobjs $ROOTALIB \
   $XLIBS $SYSLIBS $EXTRALIBS

linkstat=$?
if [ $linkstat -ne 0 ]; then
   exit $linkstat
fi

exit 0
