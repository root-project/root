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
syslibs="msvcrt.lib oldnames.lib kernel32.lib advapi32.lib \
         user32.lib gdi32.lib comdlg32.lib winspool.lib \
         $extralibs"
userlibs=lib/*.lib

name=`basename $R__LIB .dll`

bindexp=bin/bindexplib

rm -f $R__LIB

if [ "$R__PLATFORM" = "win32" ]; then
   if [ "$R__LD" = "build/win/ld.sh" ]; then
      echo "$bindexp $name $R__OBJS > lib/${name}.def"
      $bindexp $name $R__OBJS > lib/${name}.def
      echo lib -nologo -MACHINE:IX86 -out:lib/${name}.lib $R__OBJS \
           -def:lib/${name}.def $R__LEXTRA
      lib -nologo -MACHINE:IX86 -out:lib/${name}.lib $R__OBJS \
           -def:lib/${name}.def $R__LEXTRA
      if [ "$R__LIB" = "lib/libCint.dll" ]; then
         echo $R__LD $R__SOFLAGS $R__LDFLAGS -o bin/${name}.dll $R__OBJS \
              lib/${name}.exp $syslibs
         $R__LD $R__SOFLAGS $R__LDFLAGS -o bin/${name}.dll $R__OBJS \
              lib/${name}.exp $syslibs
      elif [ "$R__LIB" = "lib/libCore.dll" ]; then
         echo $R__LD $R__SOFLAGS $R__LDFLAGS -o bin/${name}.dll $R__OBJS \
              lib/${name}.exp lib/libCint.lib $syslibs WSock32.lib Oleaut32.lib
         $R__LD $R__SOFLAGS $R__LDFLAGS -o bin/${name}.dll $R__OBJS \
              lib/${name}.exp lib/libCint.lib $syslibs WSock32.lib Oleaut32.lib
      else
         echo $R__LD $R__SOFLAGS $R__LDFLAGS -o bin/${name}.dll $R__OBJS \
              lib/${name}.exp $R__EXTRA lib/libCore.lib lib/libCint.lib \
              $syslibs
         $R__LD $R__SOFLAGS $R__LDFLAGS -o bin/${name}.dll $R__OBJS \
              lib/${name}.exp $R__EXTRA lib/libCore.lib lib/libCint.lib \
              $syslibs
      fi
   fi
fi

linkstat=$?
if [ $linkstat -ne 0 ]; then
   exit $linkstat
fi

# dummy dll (real one in in bin/) to prevent rebuilds of the dll
touch $R__LIB

echo "==> $R__LIB done"

exit 0
