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
xrdbinpath=""
xrdlibpath=""
xrdmanpath=""
if test "x$xrdsys" = "x"; then
   echo "$xrdset: ERROR, you must specify the path to the xrootd installed"
   return 1;
fi
xrdbinpath="$xrdsys/bin"
if test ! -d "$xrdbinpath" ; then
   echo "$xrdset: ERROR, directory $xrdbinpath does not exist or not a directory!"
   return 1;
fi
xrdlibpath="$xrdsys/lib"
if test ! -d "$xrdlibpath" ; then
   libemsg="$xrdlibpath"
   xrdlibpath="$xrdsys/lib64"
   if test ! -d "$xrdlibpath" ; then
      echo "$xrdset: ERROR, directory $libemsg nor $xrdlibpath do not exist or not directories!"
      return 1;
   fi
fi
xrdmanpath="$xrdsys/man"
if test ! -d "$xrdmanpath" ; then
   manemsg="$xrdmanpath"
   xrdmanpath="$xrdsys/share/man"
   if test ! -d "$xrdmanpath" ; then
      echo "$xrdset: WARNING, directory $manemsg and $xrdmanpath do not exist or not directories; MANPATH unchanged"
      xrdmanpath=""
   fi
fi

# Strip present settings, if there
if test ! "x$XRDSYS" = "x" ; then
   # Trim $PATH
   if [ -n "${PATH}" ]; then
      drop_from_path "$PATH" "$xrdbinpath"
      PATH=$newpath
   fi

   # Trim $LD_LIBRARY_PATH
   if [ -n "${LD_LIBRARY_PATH}" ]; then
      drop_from_path "$LD_LIBRARY_PATH" "$xrdlibpath"
      LD_LIBRARY_PATH=$newpath
   fi

   # Trim $DYLD_LIBRARY_PATH
   if [ -n "${DYLD_LIBRARY_PATH}" ]; then
      drop_from_path "$DYLD_LIBRARY_PATH" "$xrdlibpath"
      DYLD_LIBRARY_PATH=$newpath
   fi

   # Trim $MANPATH
   if [ -n "${MANPATH}" ]; then
      drop_from_path "$MANPATH" "$xrdmanpath"
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
   PATH=$xrdbinpath; export PATH
else
   PATH=$xrdbinpath:$PATH; export PATH
fi
if [ -z "${LD_LIBRARY_PATH}" ]; then
   LD_LIBRARY_PATH=$xrdlibpath; export LD_LIBRARY_PATH
else
   LD_LIBRARY_PATH=$xrdlibpath:$LD_LIBRARY_PATH; export LD_LIBRARY_PATH
fi
if [ -z "${DYLD_LIBRARY_PATH}" ]; then
   DYLD_LIBRARY_PATH=$xrdlibpath; export DYLD_LIBRARY_PATH       
else
   DYLD_LIBRARY_PATH=$xrdlibpath:$DYLD_LIBRARY_PATH; export DYLD_LIBRARY_PATH
fi
if [ -z "${MANPATH}" ]; then
   MANPATH=$xrdmanpath:${default_manpath}; export MANPATH
else
   MANPATH=$xrdmanpath:$MANPATH; export MANPATH
fi

unset -f drop_from_path
