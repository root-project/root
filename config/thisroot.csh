# Source this script to set up the ROOT build that this script is part of.
#
# Conveniently an alias like this can be defined in ~/.cshrc:
#   alias thisroot "source bin/thisroot.sh"
#
# This script if for the csh like shells, see thisroot.sh for bash like shells.
#
# Author: Fons Rademakers, 18/8/2006

if ($?ROOTSYS) then
   set old_rootsys="$ROOTSYS"
endif

# $_ should be source .../thisroot.csh
set ARGS=($_)
if ("$ARGS" != "") then
   set thisroot="`dirname ${ARGS[2]}`"
else
   # But $_ might not be set if the script is source non-interactively.
   # In [t]csh the sourced file is inserted 'in place' inside the
   # outer script, so we need an external source of information
   # either via the current directory or an extra parameter.
   if ( -e thisroot.csh ) then
      set thisroot=${PWD}
   else if ( -e bin/thisroot.csh ) then 
      set thisroot=${PWD}/bin
   else if ( "$1" != "" ) then
      if ( -e ${1}/bin/thisroot.csh ) then
         set thisroot=${1}/bin
      else if ( -e ${1}/thisroot.csh ) then
         set thisroot=${1}
      else 
         echo "thisroot.csh: ${1} does not contain a ROOT installation"
      endif 
   else
      echo 'Error: The call to "source where_root_is/bin/thisroot.csh" can not determine the location of the ROOT installation'
      echo "because it was embedded another script (this is an issue specific to csh)."
      echo "Use either:"
      echo "   cd where_root_is; source bin/thisroot.csh"
      echo "or"
      echo "   source where_root_is/bin/thisroot.csh where_root_is" 
   endif
endif

if ($?thisroot) then 

setenv ROOTSYS "`(cd ${thisroot}/..;pwd)`"

if ($?old_rootsys) then
   setenv PATH `echo $PATH | sed -e "s;:$old_rootsys/bin:;:;g" \
                                 -e "s;:$old_rootsys/bin;;g"   \
                                 -e "s;$old_rootsys/bin:;;g"   \
                                 -e "s;$old_rootsys/bin;;g"`
   if ($?LD_LIBRARY_PATH) then
      setenv LD_LIBRARY_PATH `echo $LD_LIBRARY_PATH | \
                             sed -e "s;:$old_rootsys/lib:;:;g" \
                                 -e "s;:$old_rootsys/lib;;g"   \
                                 -e "s;$old_rootsys/lib:;;g"   \
                                 -e "s;$old_rootsys/lib;;g"`
   endif
   if ($?DYLD_LIBRARY_PATH) then
      setenv DYLD_LIBRARY_PATH `echo $DYLD_LIBRARY_PATH | \
                             sed -e "s;:$old_rootsys/lib:;:;g" \
                                 -e "s;:$old_rootsys/lib;;g"   \
                                 -e "s;$old_rootsys/lib:;;g"   \
                                 -e "s;$old_rootsys/lib;;g"`
   endif
   if ($?SHLIB_PATH) then
      setenv SHLIB_PATH `echo $SHLIB_PATH | \
                             sed -e "s;:$old_rootsys/lib:;:;g" \
                                 -e "s;:$old_rootsys/lib;;g"   \
                                 -e "s;$old_rootsys/lib:;;g"   \
                                 -e "s;$old_rootsys/lib;;g"`
   endif
   if ($?LIBPATH) then
      setenv LIBPATH `echo $LIBPATH | \
                             sed -e "s;:$old_rootsys/lib:;:;g" \
                                 -e "s;:$old_rootsys/lib;;g"   \
                                 -e "s;$old_rootsys/lib:;;g"   \
                                 -e "s;$old_rootsys/lib;;g"`
   endif
   if ($?PYTHONPATH) then
      setenv PYTHONPATH `echo $PYTHONPATH | \
                             sed -e "s;:$old_rootsys/lib:;:;g" \
                                 -e "s;:$old_rootsys/lib;;g"   \
                                 -e "s;$old_rootsys/lib:;;g"   \
                                 -e "s;$old_rootsys/lib;;g"`
   endif
   if ($?MANPATH) then
      setenv MANPATH `echo $MANPATH | \
                             sed -e "s;:$old_rootsys/man:;:;g" \
                                 -e "s;:$old_rootsys/man;;g"   \
                                 -e "s;$old_rootsys/man:;;g"   \
                                 -e "s;$old_rootsys/man;;g"`
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

endif # if ("$thisroot" != "")

set thisroot=
set old_rootsys=

