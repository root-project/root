#! /bin/sh

# Script to generate a shared library (DLL) on Win32 for VC++.
# Called by main Makefile.
#
# Author: Fons Rademakers, 29/2/2000

# the -v and -x options are not used, but handled anyway
if [ "$1" = "-v" ] ; then
   R__MAJOR=$2
   R__MINOR=$3
   R__REVIS=$4
   shift
   shift
   shift
   shift
fi

if [ "$1" = "-x" ] ; then
   R__EXPLICIT="yes"
   shift
fi

R__PLATFORM=$1
R__LD=$2
R__LDFLAGS=$3
R__SOFLAGS=$4
R__SONAME=$5
R__LIB=$6
R__OBJS=$7
R__EXTRA=$8
R__LEXTRA=$9

lastsyslib=comctl32.lib
extralibs=$lastsyslib
syslibs="kernel32.lib advapi32.lib \
         user32.lib gdi32.lib comdlg32.lib winspool.lib \
         $extralibs"

name=`basename $R__LIB .dll`
targetdir=`dirname $R__LIB`

bindexp=bin/bindexplib

rm -f $R__LIB

if [ "$targetdir" = "lib" ]; then
	libdir=lib
	dlldir=bin
	relocated=x
else 
	libdir=$targetdir
	dlldir=$targetdir
fi

if [ "$R__PLATFORM" = "win32" ]; then
   if [ "`basename $R__LD`" = "ld.sh" ]; then
      echo "$bindexp $name $R__OBJS > $libdir/${name}.def"
      $bindexp $name $R__OBJS > $libdir/${name}.def
      cmd="lib -ignore:4049,4206,4217,4221 \
           -nologo -MACHINE:IX86 -out:$libdir/${name}.lib $R__OBJS \
           -def:$libdir/${name}.def $R__LEXTRA"
      echo $cmd
      $cmd
      if [ "$R__LIB" = "lib/libCint.dll" ]; then
         cmd="$R__LD $R__SOFLAGS $R__LDFLAGS -o $dlldir/${name}.dll $R__OBJS \
              $libdir/${name}.exp $syslibs"
      elif [ "$R__LIB" = "lib/libReflex.dll" ]; then
         cmd="$R__LD $R__SOFLAGS $R__LDFLAGS -o $dlldir/${name}.dll $R__OBJS \
              $libdir/${name}.exp $R__EXTRA $syslibs"
      elif [ "$R__LIB" = "lib/libCintex.dll" ]; then
         cmd="$R__LD $R__SOFLAGS $R__LDFLAGS -o $dlldir/${name}.dll $R__OBJS \
              $libdir/${name}.exp lib/libCore.lib lib/libReflex.lib \
              lib/libCint.lib $R__EXTRA $syslibs"
      elif [ "$R__LIB" = "lib/libCore.dll" ]; then
         cmd="$R__LD $R__SOFLAGS $R__LDFLAGS -o $dlldir/${name}.dll $R__OBJS \
              $libdir/${name}.exp lib/libCint.lib lib/libMathCore.lib \
              $R__EXTRA $syslibs shell32.lib WSock32.lib Oleaut32.lib Iphlpapi.lib"
      elif [ "$R__LIB" = "lib/libminicern.dll" ]; then
         cmd="$R__LD $R__SOFLAGS $R__LDFLAGS -o $dlldir/${name}.dll $R__OBJS \
              $libdir/${name}.exp $R__EXTRA $syslibs"
      else
         if [ "$(bin/root-config --dicttype)" != "cint" ]; then
             needReflex="lib/libCintex.lib lib/libReflex.lib"
         fi
         cmd="$R__LD $R__SOFLAGS $R__LDFLAGS -o $dlldir/${name}.dll $R__OBJS \
              $libdir/${name}.exp $R__EXTRA \
              $needReflex lib/libCore.lib lib/libCint.lib \
              $syslibs"
      fi
      echo $cmd
      $cmd
   fi
fi

linkstat=$?
if [ $linkstat -ne 0 ]; then
   exit $linkstat
fi

if [ "$relocated" = "x" ]; then 
   # dummy dll (real one in in bin/) to prevent rebuilds of the dll
   touch $R__LIB
fi

# once we have the .dll we don't need the .def anymore
rm -f $libdir/${name}.def

echo "==> $R__LIB done"

exit 0
