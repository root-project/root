# Source this script in the top of the ROOT directory that you want to
# make active, e.g.:
#   cd ~/root-test
#   . bin/thisroot.sh
#
# Conveniently an alias like this can be defined in .bashrc:
#   alias thisroot=". bin/thisroot.sh"
#
# This script if for the bash like shells, see thisroot.csh for csh like shells.
#
# Author: Fons Rademakers, 18/8/2006

ROOTSYS=`pwd`; export ROOTSYS

if [ -z "${PATH}" ]; then
   PATH=$ROOTSYS/bin; export PATH
else
   PATH=$ROOTSYS/bin:$PATH; export PATH
fi

if [ -z "${LD_LIBRARY_PATH}" ]; then
   LD_LIBRARY_PATH=$ROOTSYS/lib; export LD_LIBRARY_PATH       # Linux, ELF HP-UX
else
   LD_LIBRARY_PATH=$ROOTSYS/lib:$LD_LIBRARY_PATH; export LD_LIBRARY_PATH
fi

if [ -z "${DYLD_LIBRARY_PATH}" ]; then
   DYLD_LIBRARY_PATH=$ROOTSYS/lib; export DYLD_LIBRARY_PATH   # Mac OS X
else
   DYLD_LIBRARY_PATH=$ROOTSYS/lib:$DYLD_LIBRARY_PATH; export DYLD_LIBRARY_PATH
fi

if [ -z "${SHLIB_PATH}" ]; then
   SHLIB_PATH=$ROOTSYS/lib; export SHLIB_PATH                 # legacy HP-UX
else
   SHLIB_PATH=$ROOTSYS/lib:$SHLIB_PATH; export SHLIB_PATH
fi

if [ -z "${LIBPATH}" ]; then
   LIBPATH=$ROOTSYS/lib; export LIBPATH                       # AIX
else
   export LIBPATH=$ROOTSYS/lib:$LIBPATH; export LIBPATH
fi

if [ -z "${MANPATH}" ]; then
   MANPATH=$ROOTSYS/man; export MANPATH
else
   MANPATH=$ROOTSYS/man:$MANPATH; export MANPATH
fi
