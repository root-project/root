#
# Source this to set all what you need to use Xrootd at <xrd_install_path> 
#
# Usage:
#            source /Path/to/xrd-etc/setup.csh <xrd_install_path>
#
#
set xrdsys="$1"
set binpath=""
set libpath=""
set manpath=""
if ( "x$xrdsys" == "x" ) then
   echo "ERROR: specifying the path to the installed distribution is mandatory"
   exit 1
endif
set binpath="$xrdsys/bin"
if ( -d "$binpath" ) then
else
   echo "ERROR: directory $binpath does not exist or not a directory!"
   exit 1
endif
set libpath="$xrdsys/lib"
if ( -d "$libpath" ) then
else
   set libemsg="$libpath"
   set libpath="$xrdsys/lib64"
   if ( -d "$libpath" ) then
   else
      echo "ERROR: directories $libemsg nor $libpath do not exist or are not directories!"
      exit 1
   endif
endif
set manpath="$xrdsys/man"
if ( -d "$manpath" ) then
else
   set warnmsg="$manpath "
   set manpath="$xrdsys/share/man"
   if ( -d "$manpath" ) then
   else
      set warnmsg="$warnmsg $manpath"
      echo "WARNING: directories $warnmsg do not exist or are not directories: MANPATH unchanged"
      set manpath=""
   endif
endif

set arch="`uname -s`"
if ( "x$arch" == "xDarwin" ) then
   set ismac="yes"
else
   set ismac="no"
endif

# Strip present settings, if there
if ($?XRDSYS) then

   # Trim $PATH
   set tpath=""
   set oldpath="`echo $PATH | tr -s ':' ' '`"
   foreach pp ($oldpath)
      if ( "x$binpath" != "x$pp" ) then
         if ( "x$pp" != "x" )  then
            if ( "x$tpath" == "x" )  then
              set tpath=${pp}
            else
                set tpath=${tpath}:${pp}
            endif
         endif
      endif
   end

   # Trim $LD_LIBRARY_PATH
   set tldpath=""
   set oldldpath="`echo $LD_LIBRARY_PATH | tr -s ':' ' '`"
   foreach pp ($oldldpath)
      if ( "x$libpath" != "x$pp" ) then
         if ( "x$pp" != "x" ) then
            if ( "x$tldpath" == "x" )  then
              set tldpath=${pp}
            else
                set tldpath=${tldpath}:${pp}
            endif
         endif
      endif
   end

   # Trim $DYLD_LIBRARY_PATH
   set tdyldpath=""
   if ( "x$ismac" == "xyes" ) then
      set olddyldpath="`echo $DYLD_LIBRARY_PATH | tr -s ':' ' '`"
      foreach pp ($olddyldpath)
         if ( "x$libpath" != "x$pp" ) then
            if ( "x$pp" != "x" ) then
               if ( "x$tdyldpath" == "x" )  then
                 set tdyldpath=${pp}
               else
                   set tdyldpath=${tdyldpath}:${pp}
               endif
            endif
         endif
      end
   endif

   # Trim $MAN_PATH
   set tmanpath=""
   if ( "x$manpath" != "x" ) then
      set oldmanpath="`echo $MANPATH | tr -s ':' ' '`"
      foreach pp ($oldmanpath)
         if ( "x$manpath" != "x$pp" ) then
            if ( "x$pp" != "x" ) then
               if ( "x$tmanpath" == "x" )  then
                set tmanpath=${pp}
               else
                  set tmanpath=${tmanpath}:${pp}
               endif
            endif
         endif
      end
   endif

else

   # Do not touch
   set tpath="$PATH"
   set tldpath="$LD_LIBRARY_PATH"
   if ( "x$ismac" == "xyes" ) then
      set tdyldpath="$DYLD_LIBRARY_PATH"
   endif
   if ( "x$manpath" != "x" ) then
      set tmanpath="$MANPATH"
   endif
endif

echo "Using XRD at $xrdsys"

setenv XRDSYS ${xrdsys}
if ($?PATH) then
    setenv PATH  ${binpath}:${tpath}
else
    setenv PATH  ${binpath}
endif
if ($?LD_LIBRARY_PATH) then
    setenv LD_LIBRARY_PATH ${libpath}:${tldpath}
else
    setenv LD_LIBRARY_PATH ${libpath}
endif
if ( "x$ismac" == "xyes" ) then
    if ($?DYLD_LIBRARY_PATH) then
       setenv DYLD_LIBRARY_PATH ${libpath}:${tdyldpath}
    else
       setenv DYLD_LIBRARY_PATH ${libpath}
    endif
endif
if ( "x$manpath" != "x" ) then
    if ($?MANPATH) then
       setenv MANPATH ${manpath}:${tmanpath}
    else
       setenv MANPATH ${manpath}
    endif
endif

