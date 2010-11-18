# Source this script to set up the ROOT build that this script is part of.
#
# Conveniently an alias like this can be defined in ~/.cshrc:
#   alias thisroot "source bin/thisroot.sh"
#
# This script if for the csh like shells, see thisroot.sh for bash like shells.
#
# Author: Fons Rademakers, 18/8/2006

if ($?ROOTSYS) then
   setenv OLD_ROOTSYS "$ROOTSYS"
endif

# $_ should be source .../thisroot.csh
set ARGS=($_)
set THIS="`dirname ${ARGS[2]}`"
setenv ROOTSYS "`(cd ${THIS}/..;pwd)`"

if ($?OLD_ROOTSYS) then
   if ( ! -e @bindir@/drop_from_path ) then
      echo "ERROR: the utility drop_from_path has not been build yet. Do:"
      echo "make bin/drop_from_path"
      exit 1
   endif   
   setenv PATH `@bindir@/drop_from_path -e "$OLD_ROOTSYS/bin"`
   if ($?LD_LIBRARY_PATH) then
      setenv LD_LIBRARY_PATH `@bindir@/drop_from_path -D -e -p "$LD_LIBRARY_PATH" "$OLD_ROOTSYS/lib"`
   endif
   if ($?DYLD_LIBRARY_PATH) then
      setenv DYLD_LIBRARY_PATH `@bindir@/drop_from_path -D -e -p "$DYLD_LIBRARY_PATH" "$OLD_ROOTSYS/lib"`
   endif
   if ($?SHLIB_PATH) then
      setenv SHLIB_PATH `@bindir@/drop_from_path -D -e -p "$SHLIB_PATH" "$OLD_ROOTSYS/lib"`
   endif
   if ($?LIBPATH) then
      setenv LIBPATH `@bindir@/drop_from_path -D -e -p "$LIBPATH" "$OLD_ROOTSYS/lib"`
   endif
   if ($?PYTHONPATH) then
      setenv PYTHONPATH `@bindir@/drop_from_path -D -e -p "$PYTHONPATH" "$OLD_ROOTSYS/lib"`
   endif
   if ($?MANPATH) then
      setenv MANPATH `@bindir@/drop_from_path -D -e -p "$MANPATH" "$OLD_ROOTSYS/man"`
   endif
endif


if ($?MANPATH) then
# Nothing to do
else
   # Grab the default man path before setting the path to avoid duplicates 
   if ( -X manpath ) then
      set default_manpath = `manpath`
   else
      set default_manpath = `man -w`
   endif
endif

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
   setenv MANPATH `dirname @mandir@`:$default_manpath
endif
