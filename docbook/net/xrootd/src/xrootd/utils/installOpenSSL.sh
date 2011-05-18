#!/bin/sh

#
# Script to install a given version of OpenSSL with settings adapted 
# for optimal use within XROOTD/SCALLA
#
# Syntax:
#          ./installOpenSSL.sh <installdir> [<version-or-tarball>] [<version>]
#
# where
#          <installdir>: the directory where the lib and include/openssl folders
#                        will appear
#          <version-or-tarball> :
#                        - version in the form x.j.w[patch letter] ;
#                          current default 0.9.8k
#                        or 
#                        - full path to source tarball
#          <version> : version when the 2nd argument is a tarball
#
# When relevant, the script uses 'wget' ('curl' on MacOsX) to retrieve the tarball

printhelp()
{
     echo "    "
     echo "       Small script to install a given version of OpenSSL with settings adapted"
     echo "       for optimal use within XROOTD/SCALLA"
     echo "    "
     echo "       Syntax:"
     echo "                ./installOpenSSL.sh <installdir> [<version-or-tarball>] [<version>]"
     echo "    "
     echo "       where"
     echo "                <installdir>: the directory where the lib and include/openssl folders"
     echo "                              will appear"
     echo "                <version-or-tarball> :"
     echo "                              - version in the form x.j.w[patch letter] ;"
     echo "                                current default 0.9.8k"
     echo "                              or " 
     echo "                              - full path to source tarball"
     echo "                <version> : version when the 2nd argument is a tarball"
     echo "    "
     echo "       When relevant, the script uses 'wget' ('curl' on MacOsX) to retrieve the tarball"
     echo "    "
}

ARCH=`uname -s`

XMK=make

WRKDIR=$PWD

TGTDIR=$1
if test "x$TGTDIR" =  "x" ; then
   echo " Install dir undefined!"
   printhelp
   exit
fi
echo "Installing in: $TGTDIR"

VERS="0.9.8k"
if test ! "x$2" =  "x" ; then
   VERS=$2
fi

retrieve="yes"
TARBALL="openssl-$VERS.tar.gz"
if test -f $VERS ; then
   retrieve="no"
   TARBALL=$VERS
   echo "Taking source from tarball $TARBALL"
   if test ! "x$3" = "x" ; then
      VERS=$3
   else
      tmpver=`basename $TARBALL`
      VERS=`echo $tmpver | cut -c9-14`
   fi
else
   echo "Taking source from tarball http://www.openssl.org/source/$TARBALL"
fi
echo "Version: $VERS"

# Build dir
BUILDDIR="/tmp/openssl-$VERS"
if test ! -d $BUILDDIR ; then
   mkdir $BUILDDIR
else
   # CLeanup build dir
   rm -fr $BUILDDIR/*
fi
echo "Build dir: $BUILDDIR"

# Check install dir
if test ! -d $TGTDIR ; then
   echo "Install dir does not exists: creating ..."
   mkdir -p $TGTDIR
fi

cd $BUILDDIR

# Retrieving source
if test "x$retrieve" = "xyes" ; then
   if test "x$ARCH" = "xDarwin" ; then
      curl http://www.openssl.org/source/$TARBALL -o $TARBALL
   else
      wget http://www.openssl.org/source/$TARBALL
   fi
   if test ! -f $TARBALL ; then
      echo "Tarball retrieval failed!"
      cd $WRKDIR
      exit
   fi
fi

# Untar tarball
tar xzf $TARBALL
if test ! -d openssl-$VERS ; then
   echo "Could not find source sub-directory openssl-$VERS"
   cd $WRKDIR
   exit
fi
cd openssl-$VERS

# Architecture dependent
if test "x$ARCH" = "xDarwin" ; then
   if `sysctl machdep.cpu.extfeatures | grep "64" > /dev/null  2>&1` ; then
      target="darwin64-x86_64-cc"
   fi
fi
if test "x$target" = "x" ; then
   targets=`./config | grep Configuring`
   set $targets
   target=$3
fi

echo "Machine-OS: $target"
./Configure $target no-krb5 shared no-asm -DPURIFY --prefix=$TGTDIR --openssldir=$TGTDIR

# Build
$XMK

# Test
$XMK test

# Install
$XMK install

# Go back where we started
cd $WRKDIR
