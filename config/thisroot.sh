# Source this script to set up the ROOT build that this script is part of.
#
# Conveniently an alias like this can be defined in .bashrc:
#   alias thisroot=". bin/thisroot.sh"
#
# This script if for the bash like shells, see thisroot.csh for csh like shells.
#
# Author: Fons Rademakers, 18/8/2006

if [ -n "${ROOTSYS}" ] ; then
   OLD_ROOTSYS=${ROOTSYS}
fi

if [ "x${BASH_ARGV[0]}" = "x" ]; then
    if [ ! -f bin/thisroot.sh ]; then
        echo ERROR: must "cd where/root/is" before calling ". bin/thisroot.sh" for this version of bash!
        ROOTSYS=; export ROOTSYS
        return 1
    fi
    ROOTSYS="$PWD"; export ROOTSYS
else
    # get param to "."
    THIS=$(dirname ${BASH_ARGV[0]})
    ROOTSYS=$(cd ${THIS}/..;pwd); export ROOTSYS
fi

if [ -n "${OLD_ROOTSYS}" ] ; then
   if [ ! -e @bindir@/drop_from_path ]; then
      echo "ERROR: the utility drop_from_path has not been build yet. Do:"
      echo "make bin/drop_from_path"
      return 1
   fi
   if [ -n "${PATH}" ]; then
      PATH=`@bindir@/drop_from_path -e "${OLD_ROOTSYS}/bin"`
   fi
   if [ -n "${LD_LIBRARY_PATH}" ]; then
      LD_LIBRARY_PATH=`@bindir@/drop_from_path -D -e -p "${LD_LIBRARY_PATH}" "${OLD_ROOTSYS}/lib"`
   fi
   if [ -n "${DYLD_LIBRARY_PATH}" ]; then
      DYLD_LIBRARY_PATH=`@bindir@/drop_from_path -D -e -p "${DYLD_LIBRARY_PATH}" "${OLD_ROOTSYS}/lib"`
   fi
   if [ -n "${SHLIB_PATH}" ]; then
      SHLIB_PATH=`@bindir@/drop_from_path -D -e -p "${SHLIB_PATH}" "${OLD_ROOTSYS}/lib"`
   fi
   if [ -n "${LIBPATH}" ]; then
      LIBPATH=`@bindir@/drop_from_path -D -e -p "${LIBPATH}" "${OLD_ROOTSYS}/lib"`
   fi
   if [ -n "${PYTHONPATH}" ]; then
      PYTHONPATH=`@bindir@/drop_from_path -D -e -p "${PYTHONPATH}" "${OLD_ROOTSYS}/lib"`
   fi
   if [ -n "${MANPATH}" ]; then
      MANPATH=`@bindir@/drop_from_path -D -e -p "${MANPATH}" "${OLD_ROOTSYS}/man"`
   fi
fi

if [ -z "${MANPATH}" ]; then
   # Grab the default man path before setting the path to avoid duplicates 
   if `which manpath > /dev/null 2>&1` ; then
      default_manpath=`manpath`
   else
      default_manpath=`man -w`
   fi
fi

if [ -z "${PATH}" ]; then
   PATH=@bindir@; export PATH
else
   PATH=@bindir@:$PATH; export PATH
fi

if [ -z "${LD_LIBRARY_PATH}" ]; then
   LD_LIBRARY_PATH=@libdir@; export LD_LIBRARY_PATH       # Linux, ELF HP-UX
else
   LD_LIBRARY_PATH=@libdir@:$LD_LIBRARY_PATH; export LD_LIBRARY_PATH
fi

if [ -z "${DYLD_LIBRARY_PATH}" ]; then
   DYLD_LIBRARY_PATH=@libdir@; export DYLD_LIBRARY_PATH   # Mac OS X
else
   DYLD_LIBRARY_PATH=@libdir@:$DYLD_LIBRARY_PATH; export DYLD_LIBRARY_PATH
fi

if [ -z "${SHLIB_PATH}" ]; then
   SHLIB_PATH=@libdir@; export SHLIB_PATH                 # legacy HP-UX
else
   SHLIB_PATH=@libdir@:$SHLIB_PATH; export SHLIB_PATH
fi

if [ -z "${LIBPATH}" ]; then
   LIBPATH=@libdir@; export LIBPATH                       # AIX
else
   LIBPATH=@libdir@:$LIBPATH; export LIBPATH
fi

if [ -z "${PYTHONPATH}" ]; then
   PYTHONPATH=@libdir@; export PYTHONPATH
else
   PYTHONPATH=@libdir@:$PYTHONPATH; export PYTHONPATH
fi

if [ -z "${MANPATH}" ]; then
   MANPATH=`dirname @mandir@`:${default_manpath}; export MANPATH
else
   MANPATH=`dirname @mandir@`:$MANPATH; export MANPATH
fi

if [ "x`root-config --arch | grep -v win32gcc | grep -i win32`" != "x" ]; then
  ROOTSYS="`cygpath -w $ROOTSYS`"
fi
