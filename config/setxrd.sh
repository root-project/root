#
# Source this to set all what you need to use Xrootd at <xrd_install_path> 
#
# Usage:
#            source /Path/to/xrd-etc/setxrd.sh <xrd_install_path>
#
#
xrdsys=$1
binpath=""
libpath=""
manpath=""
if test "x$xrdsys" = "x"; then
   echo "ERROR: specifying the path to the installed distribution is mandatory"
   return 1;
fi
binpath="$xrdsys/bin"
if test ! -d "$binpath" ; then
   echo "ERROR: directory $binpath does not exist or not a directory!"
   return 1;
fi
libpath="$xrdsys/lib"
if test ! -d "$libpath" ; then
   libemsg="$libpath"
   libpath="$xrdsys/lib64"
   if test ! -d "$libpath" ; then
      echo "ERROR: directory $libemsg nor $libpath do not exist or are not directories!"
      return 1;
   fi
fi
manpath="$xrdsys/man"
if test ! -d "$manpath" ; then
   manemsg="$manpath"
   manpath="$xrdsys/share/man"
   if test ! -d "$manpath" ; then
      echo "WARNING: directory $manemasg and $manpath do not exist or are not directories; MANPATH unchanged"
      manpath=""
   fi
fi

ismac=`uname -s`
if test "x$ismac" = "xDarwin"; then
   ismac="yes"
else
   ismac="no"
fi

# Strip present settings, if there
if test ! "x$XRDSYS" = "x" ; then

   # Trim $PATH
   tpath=""
   oldpath=`echo $PATH | tr -s ':' ' '`
   for pp in $oldpath; do
      if test ! "x$binpath" = "x$pp" ; then
         if test ! "x$pp" = "x" ; then
            tpath="$tpath:$pp"
         fi
      fi
   done

   # Trim $LD_LIBRARY_PATH
   tldpath=""
   oldldpath=`echo $LD_LIBRARY_PATH | tr -s ':' ' '`
   for pp in $oldldpath; do
      if test ! "x$libpath" = "x$pp" ; then
         if test ! "x$pp" = "x" ; then
            tldpath="$tldpath:$pp"
         fi
      fi
   done

   # Trim $DYLD_LIBRARY_PATH
   tdyldpath=""
   if test "x$ismac" = "xyes"; then
      olddyldpath=`echo $DYLD_LIBRARY_PATH | tr -s ':' ' '`
      for pp in $olddyldpath; do
         if test ! "x$libpath" = "x$pp" ; then
            if test ! "x$pp" = "x" ; then
               tdyldpath="$tdyldpath:$pp"
            fi
         fi
      done
   fi

   # Trim $MAN_PATH
   tmanpath=""
   if test ! "x$manpath" = "x"; then
      oldmanpath=`echo $MANPATH | tr -s ':' ' '`
      for pp in $oldmanpath; do
         if test ! "x$manpath" = "x$pp" ; then
            if test ! "x$pp" = "x" ; then
               tmanpath="$tmanpath:$pp"
            fi
         fi
      done
   fi

else

   # Do not touch
   tpath="$PATH"
   tldpath="$LD_LIBRARY_PATH"
   if test "x$ismac" = "xyes"; then
      tdyldpath="$DYLD_LIBRARY_PATH"
   fi
   if test ! "x$manpath" = "x"; then
      tmanpath="$MANPATH"
   fi
fi

echo "Using XRD at $xrdsys"

export XRDSYS="$xrdsys"
export PATH="$binpath:$tpath"
export LD_LIBRARY_PATH="$libpath:$tldpath"
if test "x$ismac" = "xyes"; then
   export DYLD_LIBRARY_PATH="$libpath:$tdyldpath"
fi
if test ! "x$manpath" = "x"; then
   export MANPATH="$manpath:$tmanpath"
fi

