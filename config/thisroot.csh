# Source this script to set up the ROOT build that this script is part of.
#
# Conveniently an alias like this can be defined in ~/.cshrc:
#   alias thisroot "source bin/thisroot.sh"
#
# This script if for the csh like shells, see thisroot.sh for bash like shells.
#
# Author: Fons Rademakers, 18/8/2006

# $_ should be source .../thisroot.csh
set ARGS=($_)
set THIS="`dirname ${ARGS[2]}`"
setenv ROOTSYS "`(cd ${THIS}/..;pwd)`"

set path = (@bindir@ $path)

if ($?LD_LIBRARY_PATH) then
   setenv LD_LIBRARY_PATH @libdir@:$LD_LIBRARY_PATH      # Linux, ELF HP-UX
else
   setenv LD_LIBRARY_PATH @libdir@
endif

if ($?DYLD_LIBRARY_PATH) then
   setenv DYLD_LIBRARY_PATH @libdir@:$DYLD_LIBRARY_PATH  # Mac OS X
else
   setenv DYLD_LIBRARY_PATH @libdir@
endif

if ($?SHLIB_PATH) then
   setenv SHLIB_PATH @libdir@:$SHLIB_PATH                # legacy HP-UX
else
   setenv SHLIB_PATH @libdir@
endif

if ($?LIBPATH) then
   setenv LIBPATH @libdir@:$LIBPATH                      # AIX
else
   setenv LIBPATH @libdir@
endif

if ($?PYTHONPATH) then
   setenv PYTHONPATH @libdir@:$PYTHONPATH
else
   setenv PYTHONPATH @libdir@
endif

if ($?MANPATH) then
   setenv MANPATH `dirname @mandir@`:$MANPATH
else
   setenv MANPATH `dirname @mandir@`
endif
