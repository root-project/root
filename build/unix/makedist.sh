#! /bin/sh

# Script to produce binary distribution of ROOT.
# Called by main Makefile.
#
# Author: Fons Rademakers, 29/2/2000

ROOTVERS=`cat build/version_number | sed -e 's/\//\./'`
TYPE=`bin/root-config --arch`
if [ "x`bin/root-config --platform`" = "xmacosx" ]; then
   TYPE=$TYPE-`sw_vers -productVersion | cut -d . -f1 -f2`
   TYPE=$TYPE-`uname -p`
   # /usr/bin/tar on OSX is BSDTAR which is for our purposes GNU tar compatible
   TAR=/usr/bin/tar
fi
if [ "x`bin/root-config --platform`" = "xsolaris" ]; then
   TYPE=$TYPE-`uname -r`
   TYPE=$TYPE-`uname -p`
fi

# debug build?
DEBUG=
BUILDOPT=`grep ROOTBUILD config/Makefile.config`
if [ "x$BUILDOPT" != "x" ]; then
   if echo $BUILDOPT | grep debug > /dev/null 2>& 1 ; then
      DEBUG=".debug"
   fi
else
   if echo $ROOTBUILD | grep debug > /dev/null 2>& 1 ; then
      DEBUG=".debug"
   fi
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
# figure out which tar to use
if [ "x$MSI" = "x1" ]; then
   TARFILE=../${TARFILE}.msi
   TARCMD="build/package/msi/makemsi.sh ${TARFILE} -T ${TARFILE}.filelist"
else
   TARFILE=${TARFILE}.tar
   ISGNUTAR="`tar --version 2>&1 | grep GNU`"
   if [ "x${ISGNUTAR}" != "x" ]; then
      TAR=tar
   else
      if [ "x`which gtar 2>/dev/null | awk '{if ($1~/gtar/) print $1;}'`" != "x" ]; then
	 TAR=gtar
      fi
   fi
   if [ "x${TAR}" != "x" ]; then
      TARFILE=${TARFILE}".gz"
      TARCMD="${TAR} zcvf ${TARFILE} -T ${TARFILE}.filelist"
   else
      # use r to append to archive which is needed when using xargs
      TARCMD="tar rvf ${TARFILE}"
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
if [ "x${TAR}" != "x" ] || [ "x$MSI" = "x1" ]; then
   $TARCMD || exit 1
else
   (cat ${TARFILE}.filelist | xargs $TARCMD) || exit 1
fi
rm ${TARFILE}.filelist 

if [ "x$DOGZIP" = "xy" ]; then
   rm -f ${TARFILE}.gz
   gzip $TARFILE
fi

cd $pwd
rm -f include/rmain.cxx

exit 0
