#!/bin/sh

# 
#    About the usage of installOpenAFS.sh
#    ------------------------------------
# 
#    This script is supposed to provide a way to build the relevant AFS authentication static
#    libraries for usage in the XrdSecProtocolpwd module. It allows to use a local AFS installation
#    avoiding the build errors due to relocatrion or missing symbols.
# 
#    The only mandatory argument is the location of a directory where to do the build; this
#    directory must be writable, but it can also be /tmp as it usage is limited to the time
#    the module libXrdSecpwd is build.
# 
#    By default version 1.5.60 of openafs is used. The switch '-v 1.4' can be used to force usage
#    of 1.4.10
#
#    The source tarball is either automatically downloaded or taken from the path specified via
#    the '-t <tarball>' option.
# 
#    The option '-c <afsetcdir>' allows to pass the path containing 'openafs/ThisCell' and 
#    other configuration files. Use '-c transarc' if the classic Trabsarc conventions are
#    used for the local AFS installation.
# 
#    Additional options to the openafs configure can be passed via the '-o <afsopt>' option.
# 
#    1. Typical usage: 'ThisCell' under /etc/openafs
# 
#       .a Run
#                utils/installOpenAFS.sh <afsinstalldir>
# 
#       .b Configure xrootd with
# 
#                ./configure.classic --with-afs=<afsinstalldir>/openafs-1.5.60 <other_configuration_otions>
# 
#    2. Typical usage: 'ThisCell' under /usr/vice/etc
# 
#       .a Run
#                utils/installOpenAFS.sh <afsinstalldir> -c transarc
# 
#       .b Configure xrootd with
# 
#                ./configure.classic --with-afs=<afsinstalldir>/openafs-1.5.60 <other_configuration_otions>
# 
#    3. Typical usage: 'ThisCell' under /usr/vice/etc, linux kernel under /usr/src/kernels/mykernel
# 
#       This is useful is an error like 
# 
# checking your AFS sysname... configure: error: Couldn't guess your Linux version. Please use the --with-afs-sysname option to configure an AFS sysname.
# make: *** No rule to make target `libafsauthent'.  Stop.
# 
#        occurs
# 
#       .a Run
#                utils/installOpenAFS.sh <afsinstalldir> -c transarc -o --with-linux-kernel-headers=/usr/src/kernels/mykernel
# 
#       .b Configure xrootd with
# 
#                ./configure.classic --with-afs=<afsinstalldir>/openafs-1.5.60 <other_configuration_otions>
# 
#    4. Typical usage: 'ThisCell' under /etc/openafs, version 1.4.10
# 
#       .a Run
#                utils/installOpenAFS.sh <afsinstalldir> -v 1.4
# 
#       .b Configure xrootd with
# 
#                ./configure.classic --with-afs=<afsinstalldir>/openafs-1.4.10 <other_configuration_otions>

printhelp()
{
     echo "    "
     echo "       Script to install a given version of OpenAFS with settings adapted"
     echo "       for optimal use within XROOTD/SCALLA"
     echo "    "
     echo "       Syntax:"
     echo "                ./installOpenAFS.sh <builddir> [-v <version>] [-c <afsetcdir>] [-t <tarball>] [-o <afsopt>]"
     echo "    "
     echo "       where"
     echo "                <builddir>: the directory where the relevant libs will be built"
     echo "                <version> : openafs version: currently supported 1.4 and 1.5"
     echo "                <afsetcdir> : path containing openafs/ThisCell or 'transarc' if classic"
     echo "                               transarc paths are to be used, eg. /usr/vice/etc/ThisCell"
     echo "                <tarball> : tarball to be used (no retrieve)"
     echo "                <afsopt> : any option for the AFS configure script; use multiple -o directives"
     echo "                           for multiple options"
     echo "    "
     echo "       When relevant, the script uses 'wget' ('curl' on MacOsX) to retrieve the tarball"
     echo "    "
}

ARCH=`uname -s`
MACH=`uname -m`
XMK=make

ONEFOURV="1.4.10"
ONEFIVEV="1.5.60"

builddir=""
confdir=""
afsopt=""
tarball=""
version=""
isc=""
iso=""
ist=""
isv=""
for i in $@ ; do
  if test "x$i" = "x-c" ; then
     isc="yes"
  elif test "x$i" = "x-o" ; then
     iso="yes"
  elif test "x$i" = "x-t" ; then
     ist="yes"
  elif test "x$i" = "x-v" ; then
     isv="yes"
  else
     # auxilliary option
     if test "x$isc" = "xyes" ; then
        confdir="$i"
        isc=""
     elif test "x$iso" = "xyes" ; then
        if test "x$afsopt" = "x" ; then
           afsopt="$i"
        else
           afsopt="$afsopt $i"
        fi
        iso=""
     elif test "x$ist" = "xyes" ; then
        tarball="$i"
        ist=""
     elif test "x$isv" = "xyes" ; then
        version="$i"
        isv=""
     else
        builddir="$i"
     fi
  fi
done

# Check build dir
if test "x$builddir" = "x" ; then
   echo "Build directory is mandatory"
   printhelp
   exit
else
   if test ! -d $builddir ; then
      echo "Build directory does not exists: create? [Y/n]"
      read cr
      if test "x$cr" = x || test "x$cr" = "xY" || test "x$cr" = "xy" ; then
         mkdir -p $builddir
      else
         echo "Build directory is mandatory"
         printhelp
         exit
      fi
   fi
fi

# Check conf dir: must contain openafs/ThisCell
if test "x$confdir" = "xtransarc"; then
   # Transarc paths
   optconf="--enable-transarc-paths"
else
   if test "x$confdir" = "x"; then
      # Check the default
      confdir="/etc"
      if test ! -f "$confdir/openafs/ThisCell" ; then
         echo "AFS default etc dir '$confdir/openafs' does not contain 'ThisCell'"
         printhelp
         exit
      fi
   fi
   # Assume the path is the correct one
   optconf="--sysconfdir=$confdir"
fi

retrieve="yes"
# Check tarball and version consistency
if test ! "x$tarball" = "x" && test -f $tarball ; then
   retrieve="no"
   if test ! "x$version" = "x" ; then
      # Check version consistency
      if `echo $tarball | grep "$version" > /dev/null 2>&1` ; then
         echo "Using tarball: $tarball ; specified version $version is consistent"
      else
         echo "Using tarball: $tarball ; specified version $version is not consistent:"
         echo "guessing version from tarball"
         version=""
      fi
   fi
fi

# Check version
if test "x$version" = "x" ; then
   if test ! "x$tarball" = "x" ; then
      if `echo $tarball | grep "1.4" > /dev/null 2>&1` ; then
         version="1.4"
         openafs="openafs-$ONEFOURV"
         echo "Using tarball: $tarball ; version set to: $version"
      elif `echo $tarball | grep "1.5" > /dev/null 2>&1` ; then
         version="1.5"
         openafs="openafs-$ONEFIVEV"
         echo "Using tarball: $tarball ; version set to: $version"
      else
         version="1.5"
         openafs="openafs-$ONEFIVEV"
         echo "Using tarball: $tarball ; untested version: set to: $version"
      fi
   else
      version="1.5"
      echo "Using default version: $version"
   fi
fi

# Prepare for retrieve, if needed
dldir=""
if test "x$retrieve" = "xyes" ; then
   tarball=""
   if test "x$version" = "x1.4" ; then
      tarball="openafs-$ONEFOURV-src.tar.gz"
      dldir="http://www.openafs.org/dl/openafs/$ONEFOURV"
      openafs="openafs-$ONEFOURV"
   elif test "x$version" = "x1.5" ; then
      tarball="openafs-$ONEFIVEV-src.tar.gz"
      dldir="http://www.openafs.org/dl/openafs/$ONEFIVEV"
      openafs="openafs-$ONEFIVEV"
   else
      echo "Unknown version: $version - must be 1.4 or 1.5"
      printhelp
      exit
   fi
fi

# Go to the build directory
CURRDIR=$PWD
cd $builddir

# Retrieving source, if needed
if test "x$retrieve" = "xyes" ; then
   if test "x$ARCH" = "xDarwin" ; then
      curl $dldir/$tarball -o $tarball
   else
      wget $dldir/$tarball
   fi
   if test ! -f $tarball ; then
      echo "Tarball retrieval failed!"
      cd $CURRDIR
      exit
   fi
fi

echo "$PWD"
echo "$tarball"
echo "$openafs"

# Untar tarball
rm -fr $openafs
tar xzf $tarball
if test ! -d $openafs ; then
   echo "Could not find source sub-directory $openafs"
   cd $CURRDIR
   exit
fi
cd $openafs

# Apply patch
if test "x$version" = "x1.4" ; then
   if test "x$MACH" = "xx86_64" ; then
      rm -f configure.orig
      cp -rp configure configure.orig
      sed -e "s|\"-g -O2 -D_LARGEFILE64_SOURCE\"|\"-g -O2 -D_LARGEFILE64_SOURCE -fPIC\"|" < configure.orig > configure
      echo "Patch for $MACH applied"
   fi
elif test "x$version" = "x1.5" ; then
   if test "x$MACH" = "xx86_64" ; then
      rm -f configure.orig
      cp -rp configure configure.orig
      sed -e "s|XCFLAGS=\"-D_LARGEFILE64_SOURCE\"|XCFLAGS=\"-D_LARGEFILE64_SOURCE -fPIC\"|" < configure.orig > configure
      echo "Patch for $MACH applied"
   fi
else
   echo "Unknown version: $version - must be 1.4 or 1.5"
   printhelp
   exit
fi

# Configure
./configure $optconf $afsopt

# Build
$XMK libafsauthent

# Check the build
files="libafsauthent.a libafsrpc.a"
for f in $files ; do
   if test ! -f "lib/$f" ; then
      echo "Build failed: lib/$f not found!"
      break
   fi
done
files="stds.h kautils.h com_err.h"
for f in $files ; do
   if test ! -f "include/afs/$f" ; then
      echo "Build failed: include/afs/$f not found!"
      break
   fi
done

# Go back where we started
cd $CURRDIR

