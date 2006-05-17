#! /bin/sh

# Script to produce binary distribution of ROOT.
# Called by main Makefile.
#
# Author: Fons Rademakers, 29/2/2000

ROOTVERS=`cat build/version_number | sed -e 's/\//\./'`
TYPE=`bin/root-config --arch`
if [ "x$TYPE" = "xmacosx" ]; then
   TYPE=$TYPE-`uname -p`
fi

# debug build?
DEBUG=`grep ROOTBUILD config/Makefile.config | sed 's,^ROOTBUILD.*= \([^[:space:]]*\)$,\1,'`
if [ "x${DEBUG}" != "x" ]; then
   DEBUG=".debug"
fi

# MSI?
if [ "x$1" = "x-msi" ]; then
   MSI=1
   shift
fi

# compiler specified?
COMPILER=$1
if [ "x${COMPILER}" != "x" ]; then
   COMPILER="-${COMPILER}"
fi

TARFILE=root_v${ROOTVERS}.${TYPE}${COMPILER}${DEBUG}
# figure out what tar to use
if [ "x$MSI" == "x1" ]; then
   TAR=build/package/msi/makemsi.sh
   TARFILE=../${TARFILE}.msi
else
   TARFILE=${TARFILE}.tar
   if [ "x`which gtar 2>/dev/null | awk '{if ($1~/gtar/) print $1;}'`" != "x" ]; then
      TAR="gtar zcvf"
      TARFILE=${TARFILE}".gz"
   else
      TAR="tar cvf"
      DOGZIP="y"
   fi
fi

cp -f main/src/rmain.cxx include/
pwd=`pwd`
if [ "x${MSI}" = "x" ]; then
   dir=`basename $pwd`
   cd ..
fi

${pwd}/build/unix/distfilelist.sh $dir > ${TARFILE}.filelist
rm -f ${TARFILE}
$TAR ${TARFILE} -T ${TARFILE}.filelist || exit 1
rm ${TARFILE}.filelist 

if [ "x$DOGZIP" = "xy" ]; then
   rm -f ${TARFILE}.gz
   gzip $TARFILE
fi

cd $pwd
rm -f include/rmain.cxx

exit 0
