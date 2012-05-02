#
# Source this to set all what you need to use Xrootd at <xrd_install_path> 
#
# Usage:
#      source /Path/to/xrd-etc/setxrd.sh <xrd_install_path>
#
# Author: Gerardo Ganis

drop_from_path()
{
   # Assert that we got enough arguments
   if test $# -ne 2 ; then
      echo "drop_from_path: needs 2 arguments"
      return 1
   fi

   p=$1
   drop=$2

   newpath=`echo $p | sed -e "s;:${drop}:;:;g" \
                          -e "s;:${drop};;g"   \
                          -e "s;${drop}:;;g"   \
                          -e "s;${drop};;g"`
}

xrdset="xrdset"
xrdsys=$1
binpath=""
libpath=""
manpath=""
if test "x$xrdsys" = "x"; then
   echo "$xrdset: ERROR, you must specify the path to the xrootd installed"
   return 1;
fi
binpath="$xrdsys/bin"
if test ! -d "$binpath" ; then
   echo "$xrdset: ERROR, directory $binpath does not exist or not a directory!"
   return 1;
fi
libpath="$xrdsys/lib"
if test ! -d "$libpath" ; then
   libemsg="$libpath"
   libpath="$xrdsys/lib64"
   if test ! -d "$libpath" ; then
      echo "$xrdset: ERROR, directory $libemsg nor $libpath do not exist or not directories!"
      return 1;
   fi
fi
manpath="$xrdsys/man"
if test ! -d "$manpath" ; then
   manemsg="$manpath"
   manpath="$xrdsys/share/man"
   if test ! -d "$manpath" ; then
      echo "$xrdset: WARNING, directory $manemsg and $manpath do not exist or not directories; MANPATH unchanged"
      manpath=""
   fi
fi

# Strip present settings, if there
if test ! "x$XRDSYS" = "x" ; then
   # Trim $PATH
   if [ -n "${PATH}" ]; then
      drop_from_path $PATH $binpath
      PATH=$newpath
   fi

   # Trim $LD_LIBRARY_PATH
   if [ -n "${LD_LIBRARY_PATH}" ]; then
      drop_from_path $LD_LIBRARY_PATH $libpath
      LD_LIBRARY_PATH=$newpath
   fi

   # Trim $DYLD_LIBRARY_PATH
   if [ -n "${DYLD_LIBRARY_PATH}" ]; then
      drop_from_path $DYLD_LIBRARY_PATH $libpath
      DYLD_LIBRARY_PATH=$newpath
   fi

   # Trim $MANPATH
   if [ -n "${MANPATH}" ]; then
      drop_from_path $MANPATH $manpath
      MANPATH=$newpath
   fi
fi

if [ -z "${MANPATH}" ]; then
   # Grab the default man path before setting the path to avoid duplicates
   if `which manpath > /dev/null 2>&1` ; then
      default_manpath=`manpath`
   else
      default_manpath=`man -w 2> /dev/null`
   fi
fi

#echo "Using XRD at $xrdsys"

export XRDSYS="$xrdsys"
if [ -z "${PATH}" ]; then
   PATH=$binpath; export PATH
else
   PATH=$binpath:$PATH; export PATH
fi
if [ -z "${LD_LIBRARY_PATH}" ]; then
   LD_LIBRARY_PATH=$libpath; export LD_LIBRARY_PATH
else
   LD_LIBRARY_PATH=$libpath:$LD_LIBRARY_PATH; export LD_LIBRARY_PATH
fi
if [ -z "${DYLD_LIBRARY_PATH}" ]; then
   DYLD_LIBRARY_PATH=$libpath; export DYLD_LIBRARY_PATH       
else
   DYLD_LIBRARY_PATH=$libpath:$DYLD_LIBRARY_PATH; export DYLD_LIBRARY_PATH
fi
if [ -z "${MANPATH}" ]; then
   MANPATH=$manpath:${default_manpath}; export MANPATH
else
   MANPATH=$manpath:$MANPATH; export MANPATH
fi
