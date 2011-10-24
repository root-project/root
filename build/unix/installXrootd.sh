#!/bin/sh

#
# Script to install a given version of Xrootd/Scalla
#
# Syntax:
#          ./installXrootd.sh <installdir> [-h|--help] [-d|--debug] [-o|--optimized]
#                                          [-v <version>|--version=<version>]
#                                          [-t <tarball>|--tarball=<tarball>]
#                                           [-b <where-to-build>|--builddir=<where-to-build>]
#                                           [--xrdopts="<opts-to-xrootd>"]
#
# See printhelp for a description of the options.
#

printhelp()
{

        echo " "
        echo "  Script to install a given version of Xrootd/Scalla"
        echo " "
        echo "  Syntax:"
        echo "           ./installXrootd.sh <installdir> [-h|--help] [-d|--debug] [-o|--optimized]"
        echo "                                           [-v <version>|--version=<version>]"
        echo "                                           [-t <tarball>|--tarball=<tarball>]"
        echo "                                           [-b <where-to-build>|--builddir=<where-to-build>]"
        echo "                                           [--xrdopts=\"<opts-to-xrootd>\"]"
        echo " "
        echo "  where"
        echo "           <installdir>: the directory where the bin, lib, include/xrootd, share and man folders"
        echo "                         will appear"
        echo "           -b <where-to-build>, --builddir=<where-to-build>"
        echo "                         directory where to build; default /tmp/xrootd-<version>"
        echo "           -d,--debug    build in debug mode (no optimization)"
        echo "           -h, --help    print this help screen"
        echo "           -o,--optimized build in optimized mode without any debug symbol"
        echo "           -t <tarball>, --tarball=<tarball>"
        echo "                         - full local path to source tarball"
        echo "           -v <version>, --version=<version>"
        echo "                         - version in the form x.j.w[-hash-or-tag] ;"
        echo "                           current default 3.1.0"
        echo "           --xrdopts=<opts-to-xrootd>"
        echo "                         additional configuration options to xrootd (see xrootd web site)"
        echo " "
        echo "  When relevant, the script uses 'wget' ('curl' on MacOsX) to retrieve the tarball"
}

DBGOPT="-DCMAKE_BUILD_TYPE=RelWithDebInfo"
TGTDIR=""
VERS=""
TARBALL=""
BUILDDIR=""
XRDOPTS=""

#
# Parse long options first
other_args=
short_opts=
is_short="no"
for i in $@ ; do
   opt=""
   case $i in
      --*) opt=`echo "$i" | sed 's/--//'` ;;
      -*) if test "x$short_opts" = "x" ; then
            short_opts="$i" ;
         else
            short_opts="$short_opts $i" ;
         fi; is_short="yes" ;;
      *) if test "x$is_short" = "xyes" ; then
            if test "x$short_opts" = "x" ; then
               short_opts="$i" ;
            else
               short_opts="$short_opts $i" ;
            fi;
            is_short="no"
         else
            if test "x$other_args" = "x" ; then
               other_args="$i";
            else
               other_args="$other_args $i";
            fi;
         fi;
   esac
   if test ! "x$opt" = "x" ; then
      case "$opt" in
         *=*) oarg=`echo "$opt" | sed 's/[-_a-zA-Z0-9]*=//'`;;
         *) oarg= ;;
      esac ;
      case $opt in
         builddir=*) BUILDDIR="$oarg" ;;
         debug)      DBGOPT="-DCMAKE_BUILD_TYPE=Debug" ;;
         help)       printhelp ; exit ;;
         optimized)  DBGOPT="-DCMAKE_BUILD_TYPE=Release" ;;
         tarball=*)  TARBALL="$oarg" ;;
         version=*)  VERS="$oarg" ;;
         xrdopts=*)  XRDOPTS="$oarg" ;;
      esac
   fi
done

if test ! "x$short_opts" = "x" ; then
   while getopts b:t:v:dho i $short_opts ; do
      case $i in
      b) BUILDDIR="$OPTARG" ;;
      d) DBGOPT="-DCMAKE_BUILD_TYPE=Debug" ;;
      h) printhelp ; exit ;;
      o) DBGOPT="-DCMAKE_BUILD_TYPE=Release" ;;
      t) TARBALL="$OPTARG" ;;
      v) VERS="$OPTARG" ;;
      \?) printhelp; exit 1 ;;
      esac
      if test ! "x$OPTARG" = "x" ; then
         noa=
         for a in $other_args ; do
            if test ! "x$OPTARG" = "x$a" ; then
               if test "x$noa" = "x" ; then
                  noa="$a"
               else
                  noa="$noa $a"
               fi
            fi
         done
         other_args=$noa
      fi
   done

   # Fill empty fields with any non-prefixed argument
   if test ! "x$other_args" = "x" ; then
      TGTDIR="$other_args"
   fi
fi

XMK=make

WRKDIR=$PWD

if test "x$TGTDIR" =  "x" ; then
   echo " Install dir undefined!"
   printhelp
   exit
fi
echo "Installing in: $TGTDIR"

if test "x$VERS" =  "x" ; then
   VERS="3.1.0"
fi
echo "Version: $VERS"

retrieve="yes"
if test ! "x$TARBALL" = "x" && test -f $TARBALL ; then
   retrieve="no"
   TGTBALL=$TARBALL
fi
if test "x$retrieve" = "xyes" ; then
   if test "x$TARBALL" = "x" ; then
      TARBALL="http://xrootd.slac.stanford.edu/download/v$VERS/xrootd-$VERS.tar.gz"
      TGTBALL="xrootd-$VERS.tar.gz"
   else
      TGTBALL=`basename $TARBALL`
   fi
fi
if test "x$retrieve" = "xyes" ; then
   echo "Retrieving source from tarball $TARBALL"
else
   echo "Building tarball $TARBALL"
fi

# Build dir
if test "x$BUILDDIR" = "x"; then
   BUILDDIR="/tmp/xrootd-$VERS"
fi
if test ! -d $BUILDDIR ; then
   mkdir -p $BUILDDIR
   if test ! -d $BUILDDIR ; then
      echo "Could not create build dir '$BUILDDIR': cannot continue"
      exit 1
   fi
else
   # CLeanup build dir
   rm -fr $BUILDDIR/*
fi
echo "Build dir: $BUILDDIR"

# Check install dir
if test ! -d $TGTDIR ; then
   echo "Install dir does not exists: creating ..."
   mkdir -p $TGTDIR
   if test ! -d $TGTDIR ; then
      echo "Could not create install dir '$TGTDIR': make sure that you have the rights to do it;"
      echo "for example, run"
      echo "     sudo mkdir -p $TGTDIR"
      echo "     sudo chown $USER $TGTDIR"
      echo "before re-running this script"
      exit 1
   fi
fi

cd $BUILDDIR

# Retrieving source
ARCH=`uname -s`
if test "x$retrieve" = "xyes" ; then
   if test "x$ARCH" = "xDarwin" ; then
      curl $TARBALL -o $TGTBALL
   else
      wget $TARBALL -O $TGTBALL
   fi
   if test ! -f $TGTBALL ; then
      echo "Tarball retrieval failed!"
      cd $WRKDIR
      exit 1
   fi
fi

# Untar tarball
if test "x$ARCH" = "xSunOS" ; then
   XMK="gmake"
   gunzip -c $TGTBALL > "$TGTBALL.tar"
   tar xf "$TGTBALL.tar"
   rm -f "$TGTBALL.tar"
else
   tar xzf $TGTBALL
fi
if test ! -d xrootd-$VERS ; then
   echo "Could not find source sub-directory xrootd-$VERS"
   cd $WRKDIR
   exit 1
fi
cd xrootd-$VERS

# CMake or old {make,configure} ?
if test -f CMakeLists.txt ; then

   # CMake: check if there
   XCMK=`which cmake 2> /dev/null`
   echo "XCMK = '$XCMK'"
   if test "x$XCMK" =  "x" || test ! -f $XCMK ; then
      echo " "
      echo "To build xrootd cmake is required: "
      echo "you can get it from http://cmake.org/cmake/resources/software.html"
      echo "or from the software manager of your system"
      echo " "
      exit 1
   fi

   # Check that we can build this version
   if test ! -r VERSION_INFO ; then
      echo "VERSION_INFO file not found: this xrootd version is probably too old and cannot be built by this script"
      cd $WRKDIR
      exit 1
   fi

   # Create build directory
   mkdir build
   cd build

   # Configure
   $XCMK -DCMAKE_INSTALL_PREFIX=$TGTDIR $DBGOPT $XRDOPTS ..

   # Build
   $XMK
   if [ "$?" != "0" ] ; then
      echo "Problems running $XMK ..."
      cd $WRKDIR
      exit "$?"
   fi

   # Install
   $XMK install
   if [ "$?" != "0" ] ; then
      echo "Problems running $XMK install ..."
      cd $WRKDIR
      exit "$?"
   fi

else

   # Old {configure,make}

   # Check that we can build this version
   if test ! -r configure.classic ; then
      echo "configure.classic file not found: this xrootd version cannot be built by this script"
      cd $WRKDIR
      exit 1
   fi

   # Configure options
   if test "x$DBGOPT" = "xRelease" ; then
      DBGOPT=""
   else
      DBGOPT="--build=debug"
   fi
   CFGOPT="--disable-krb4 --no-arch-subdirs --disable-mon --enable-krb5"

   # Configure
   ./configure.classic --prefix=$TGTDIR $DBGOPT $CFGOPT $XRDOPTS
   if [ "$?" != "0" ] ; then
      echo "Problems running configure.classic ..."
      cd $WRKDIR
      exit "$?"
   fi

   # Make
   $XMK
   if [ "$?" != "0" ] ; then
      echo "Problems running $XMK ..."
      cd $WRKDIR
      exit "$?"
   fi

   # Install
   $XMK install

fi

# Go back where we started
cd $WRKDIR

exit
