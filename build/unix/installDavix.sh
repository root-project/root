#!/bin/bash

#
# Script to install a given version of Davix
#  created from the xrootd similar script
#
# Syntax:
#  ./installDavix.sh [<installdir>] [-h|--help] [-d|--debug] [-o|--optimized]
#                    [-v <version>|--version=<version>]
#                    [-t <tarball>|--tarball=<tarball>]
#                    [-b <where-to-build>|--builddir=<where-to-build>]
#                    [--dvxopts="<opts-to-davix>"]
#                    [--vers-subdir[=<version-root>]] [--no-vers-subdir]
#                    [-j <concurrent-build-jobs>|--jobs=<concurrent-build-jobs>]
#                    [-k|--keep] [--bzip2]
#
# See printhelp for a description of the options.
#

printhelp()
{
        echo " "
        echo "  Script to install a given version of Davix"
        echo " "
        echo "  Syntax:"
        echo "   ./installDavix.sh [<installdir>] [-h|--help] [-d|--debug] [-o|--optimized]"
        echo "                      [-v <version>|--version=<version>]"
        echo "                      [-t <tarball>|--tarball=<tarball>]"
        echo "                      [-b <where-to-build>|--builddir=<where-to-build>]"
        echo "                      [--dvxopts=\"<opts-to-davix>\"]"
        echo "                      [-j <concurrent-build-jobs>|--jobs=<concurrent-build-jobs>]"
        echo "                      [--vers-subdir[=<version-root>]] [--no-vers-subdir]"
        echo "                      [-k|--keep] [--bzip2]"
        echo " "
        echo "  where"
        echo "   <installdir>: the directory where the bin, lib, include/davix, share and"
        echo "                 man directories will appear (see also --vers-subdir)"
        echo "                 The default is ."
        echo "   -b <where-to-build>, --builddir=<where-to-build>"
        echo "                 directory where to build; default /tmp/davix-<version>"
        echo "   -d,--debug    build in debug mode (no optimization)"
        echo "   -h, --help    print this help screen"
        echo "   -o,--optimized build in optimized mode without any debug symbol"
        echo "   -t <tarball>, --tarball=<tarball>"
        echo "                 full local path to source tarball"
        echo "   -v <version>, --version=<version>"
        echo "                 version in the form x.j.w[-hash-or-tag] ;"
        echo "                 current default 0.2.7-3"
        echo "   --dvxopts=<opts-to-davix>"
        echo "                 additional configuration options to davix"
        echo "                 (see davix web site)"
        echo "   --no-vers-subdir install in <installdir> instead of <installdir>/davix-<version>"
        echo "                 (or <installdir>/<version-root><version>, see --vers-subdir"
        echo "   --vers-subdir[=<version-root>]"
        echo "                 install in <installdir>/<version-root><version> instead of"
        echo "                 <installdir>/davix-<version> or <installdir>. Has priority"
        echo "                 over --no-vers-subdir. Default <version-root>=davix-."
        echo "                 This option is on by default."
        echo "   -j <jobs>, --jobs=<jobs>"
        echo "                 number of build jobs to run simultaneously when bulding;"
        echo "                 default is <number-of-cores> + 1."
        echo "   -k, --keep"
        echo "                 keep the build directory"
        echo "   --bzip2"
        echo "                 use bzip2 to manage the tarball (when extension is .b2z)"
        echo " "
        echo "  When relevant, the script uses 'wget' ('curl' on MacOS X) to retrieve"
        echo "  the tarball"
}

cleanbuilddir()
{
   if test ! "x$KEEP" = "xyes"; then
      if test ! "x$BUILDDIR" = "x" && test -d $BUILDDIR ; then
         rm -rf $BUILDDIR
      fi
   fi
}

DBGOPT="-DCMAKE_BUILD_TYPE=RelWithDebInfo"
TGTDIR="."
VERS=""
TARBALL=""
BUILDDIR=""
XRDOPTS=""
VSUBDIR="davix-"
MAKEMJ=""
KEEP=""
UNZIP="gunzip"
TUNZIP="xzf"

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
         jobs)       MAKEMJ="-j$OPTARG" ;;
         no-vers-subdir) VSUBDIR="" ;;
         optimized)  DBGOPT="-DCMAKE_BUILD_TYPE=Release" ;;
         tarball=*)  TARBALL="$oarg" ;;
         version=*)  VERS="$oarg" ;;
         vers-subdir) VSUBDIR="davix-" ;;
         vers-subdir=*) VSUBDIR="$oarg" ;;
         xrdopts=*)  XRDOPTS="$oarg" ;;
         keep)       KEEP="yes" ;;
         bzip2)      UNZIP="bunzip2" ; TUNZIP="xjf" ;;
      esac
   fi
done

if test ! "x$short_opts" = "x" ; then
   while getopts b:j:t:v:dhok i $short_opts ; do
      case $i in
      b) BUILDDIR="$OPTARG" ;;
      d) DBGOPT="-DCMAKE_BUILD_TYPE=Debug" ;;
      h) printhelp ; exit ;;
      j) MAKEMJ="-j$OPTARG" ;;
      o) DBGOPT="-DCMAKE_BUILD_TYPE=Release" ;;
      t) TARBALL="$OPTARG" ;;
      v) VERS="$OPTARG" ;;
      k) KEEP="yes" ;;
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
fi

# Fill empty fields with any non-prefixed argument
if test ! "x$other_args" = "x" ; then
   TGTDIR="$other_args"
fi

XMK=make

WRKDIR=$PWD

if test "x$TGTDIR" =  "x" ; then
   echo " Install dir undefined!"
   printhelp
   exit 1
else
   tgtd="$TGTDIR"
   TGTDIR=`(cd $tgtd && pwd)`
   if [ "$?" -ne "0" ]; then
      echo "Install dir $tgtd does not exist, please create it first."
      exit 1
   fi
fi

if test "x$VERS" =  "x" ; then
   VERS="0.2.10"
fi
echo "Version: $VERS"

if test ! "x$VSUBDIR" =  "x" ; then
   TGTDIR="$TGTDIR/$VSUBDIR$VERS"
fi
echo "Installing in: $TGTDIR"

retrieve="yes"
if test ! "x$TARBALL" = "x" && test -f $TARBALL ; then
   retrieve="no"
   TGTBALL=$TARBALL
fi
if test "x$retrieve" = "xyes" ; then
   if test "x$TARBALL" = "x" ; then
      TARBALL="http://grid-deployment.web.cern.ch/grid-deployment/dms/lcgutil/tar/davix/davix-$VERS.tar.gz"
      TGTBALL="davix-$VERS.tar.gz"
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
   BUILDDIR="/tmp/davix-$VERS-$RANDOM"
fi
if test ! -d $BUILDDIR ; then
   mkdir -p $BUILDDIR
   if test ! -d $BUILDDIR ; then
      echo "Could not create build dir $BUILDDIR, exiting..."
      exit 1
   fi
else
   # Builddir already exists, exit
   echo "Build dir $BUILDDIR already exists, exiting..."
   exit 1
fi
echo "Build dir: $BUILDDIR"

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
      cleanbuilddir
      exit 1
   fi
fi

# Untar tarball
if test "x$ARCH" = "xSunOS" ; then
   XMK="gmake"
   $UNZIP -c $TGTBALL > "$TGTBALL.tar"
   tar xf "$TGTBALL.tar"
   rm -f "$TGTBALL.tar"
else
   tar $TUNZIP $TGTBALL
fi
if test ! -d davix-$VERS ; then
   echo "Could not find source sub-directory davix-$VERS"
   cd $WRKDIR
   cleanbuilddir
   exit 1
fi
cd davix-$VERS

# CMake or old {make,configure} ?
if test -f CMakeLists.txt ; then

   # CMake: check if there
   XCMK=`which cmake 2> /dev/null`
   echo "XCMK = '$XCMK'"
   if test "x$XCMK" =  "x" || test ! -f $XCMK ; then
      echo " "
      echo "To build davix cmake is required: "
      echo "you can get it from http://cmake.org/cmake/resources/software.html"
      echo "or from the software manager of your system"
      echo " "
      cd $WRKDIR
      cleanbuilddir
      exit 1
   fi

   # Check that we can build this version
   #if test ! -r VERSION_INFO ; then
   #   echo "VERSION_INFO file not found: this davix version is probably too old and cannot be built by this script"
   #   cd $WRKDIR
   #   cleanbuilddir
   #   exit 1
   #fi

   # Create build directory
   mkdir build
   cd build

   # Configure
   $XCMK -DCMAKE_INSTALL_PREFIX=$TGTDIR $DBGOPT $XRDOPTS ..

   # Get the '-j' setting if not specified
   if test "x$MAKEMJ" = "x" ; then
      MJ=`grep -c bogomips /proc/cpuinfo 2> /dev/null`
      [ "$?" != 0 ] && MJ=`sysctl hw.ncpu | cut -b10 2> /dev/null`
      let MJ++
      MAKEMJ="-j$MJ"
   fi

   # Build
   $XMK $MAKEMJ
   if [ "$?" != "0" ] ; then
      echo "Problems running $XMK  $MAKEMJ ..."
      cd $WRKDIR
      cleanbuilddir
      exit "$?"
   fi

   # Install
   $XMK install
   if [ "$?" != "0" ] ; then
      echo "Problems running $XMK install ..."
      cd $WRKDIR
      cleanbuilddir
      exit "$?"
   fi

else

   # Old {configure,make}

   # Check that we can build this version
   if test ! -r configure.classic ; then
      echo "configure.classic file not found: this davix version cannot be built by this script"
      cd $WRKDIR
      cleanbuilddir
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
      cleanbuilddir
      exit "$?"
   fi

   # Make
   $XMK
   if [ "$?" != "0" ] ; then
      echo "Problems running $XMK ..."
      cd $WRKDIR
      cleanbuilddir
      exit "$?"
   fi

   # Install
   $XMK install

fi

# Go back where we started
cd $WRKDIR
cleanbuilddir

exit
