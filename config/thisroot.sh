# Source this script to set up the ROOT build that this script is part of.
#
# Conveniently an alias like this can be defined in .bashrc:
#   alias thisroot=". bin/thisroot.sh"
#
# This script if for the bash like shells, see thisroot.csh for csh like shells.
#
# Author: Fons Rademakers, 18/8/2006

if [ "x${BASH_ARGV[0]}" = "x" ]; then
    if [ ! -f bin/thisroot.sh ]; then
        echo ERROR: must "cd where/root/is" before calling ". bin/thisroot.sh" for this version of bash!
        ROOTSYS=; export ROOTSYS
        return
    fi
    ROOTSYS="$PWD"; export ROOTSYS
else
    # get param to "."
    THIS=$(dirname ${BASH_ARGV[0]})
    ROOTSYS=$(cd ${THIS}/..;pwd); export ROOTSYS
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
   MANPATH=`dirname @mandir@`; export MANPATH
else
   MANPATH=`dirname @mandir@`:$MANPATH; export MANPATH
fi

if [ "x`root-config --arch | grep -v win32gcc | grep -i win32`" != "x" ]; then
  ROOTSYS="`cygpath -w $ROOTSYS`"
fi
