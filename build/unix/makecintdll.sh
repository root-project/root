#! /bin/sh

# Script to create auxiliary CINT dll's.
# Called by main Makefile.
#
# Author: Fons Rademakers, 27/7/2000

PLATFORM=$1 ; shift
if [ $PLATFORM != "clean" ]; then
   DLLNAME=$1       ; shift
   DLLDIRNAME=$1    ; shift
   DLLHEADERS=$1    ; shift
   CINT=$1          ; shift
   ROOTCINT=$1      ; shift
   MAKELIB=$1       ; shift
   CXX=$1           ; shift
   CC=$1            ; shift
   LD=$1            ; shift
   OPT=$1           ; shift
   CINTCXXFLAGS=$1  ; shift
   CINTCFLAGS=$1    ; shift
   LDFLAGS=$1       ; shift 
   SOFLAGS=$1       ; shift 
   SOEXT=$1         ; shift
   COMPILER=$1      ; shift
fi
if [ $PLATFORM = "macosx" ]; then
   macosx_minor=`sw_vers | sed -n 's/ProductVersion://p' | cut -d . -f 2`
   AUXCXXFLAGS=-fno-inline
fi
if [ $PLATFORM = "win32" ]; then
   EXESUF=.exe
   CC=`pwd`/$CC
fi
if [ "x$COMPILER" = "xgnu" ]; then
   GCC_MAJOR=`$CXX -dumpversion 2>&1 | cut -d'.' -f1`
   GCC_MINOR=`$CXX -dumpversion 2>&1 | cut -d'.' -f2`
fi

# Filter out the explicit link flag
if [ "x`echo $MAKELIB | grep build/unix/makelib.sh`" != "x" ]; then
   MAKELIB=`echo $MAKELIB | sed -e "s/ -x//g"`
fi

CINTDIRL=cint/lib
CINTDIRI=cint/include
CINTDIRS=cint/stl

clean() {
   rm -f $CINTDIRI/$DLLNAME.dll          $CINTDIRI/$DLLNAME.so.*
}

execute() {
   echo $1
   $1
}

rename() {
   if [ "$SOEXT" != "dll" ]; then
      if [ "$PLATFORM" = "macosx" ]; then
         if [ $macosx_minor -ge 4 ]; then
            mv $1.$SOEXT $1.dll
            rm -f $1.so
         else
            mv $1.so $1.dll
            rm -f $1.$SOEXT
         fi
      else
         mv $1.$SOEXT $1.dll
      fi
   fi
}

macrename() {
   if [ "$PLATFORM" = "macosx" ]; then
      if [ $macosx_minor -ge 4 ]; then
         mv -f $1.$SOEXT $1.so
      else
         rm -f $1.$SOEXT
      fi
   fi;
}

cpdllwin32() {
   mv -f bin/$DLLNAME.dll   $CINTDIRI
}

##### first delete old dll's #####

clean

if [ $PLATFORM = "clean" ]; then
   exit 0;
fi

##### $DLLNAME.dll  & stdcxxfunc.dll #####

DLLDIR=$CINTDIRL/$DLLDIRNAME

execute "$CINT -K -w1 -z$DLLNAME -n$DLLDIR/G__$DLLNAME.c -D__MAKECINT__ \
         -DG__MAKECINT -c-2 -Z0 -I$DLLDIR $DLLHEADERS"
execute "$CC $OPT $CINTCFLAGS -I. -o $DLLDIR/G__$DLLNAME.o -I$DLLDIR \
         -c $DLLDIR/G__$DLLNAME.c"
$MAKELIB $PLATFORM $LD "$LDFLAGS" "$SOFLAGS" $DLLNAME.$SOEXT \
   $CINTDIRI/$DLLNAME.$SOEXT "$DLLDIR/G__$DLLNAME.o"
rename $CINTDIRI/$DLLNAME

if [ $PLATFORM = "win32" ]; then
   cpdllwin32
fi

exit 0
