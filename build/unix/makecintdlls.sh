#! /bin/sh

# Script to create auxiliary CINT dll's.
# Called by main Makefile.
#
# Author: Fons Rademakers, 27/7/2000

PLATFORM=$1 ; shift
if [ $PLATFORM != "clean" ]; then
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
   SOEXT=so
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
   rm -f $CINTDIRI/stdfunc.dll          $CINTDIRI/stdfunc.so.*
   rm -f $CINTDIRI/stdcxxfunc.dll       $CINTDIRI/stdcxxfunc.so.*
   rm -f $CINTDIRI/posix.dll            $CINTDIRI/posix.so.*
   rm -f $CINTDIRI/systypes.h
   rm -f $CINTDIRI/sys/types.h
   rm -f $CINTDIRI/sys/ipc.dll          $CINTDIRI/sys/ipc.so.*
   rm -f $CINTDIRS/string.dll           $CINTDIRS/string.so.*
   rm -f $CINTDIRS/vector.dll           $CINTDIRS/vector.so.*
   rm -f $CINTDIRS/list.dll             $CINTDIRS/list.so.*
   rm -f $CINTDIRS/deque.dll            $CINTDIRS/deque.so.*
   rm -f $CINTDIRS/map.dll              $CINTDIRS/map.so.*
   rm -f $CINTDIRS/map2.dll             $CINTDIRS/map2.so.*
   rm -f $CINTDIRS/set.dll              $CINTDIRS/set.so.*
   rm -f $CINTDIRS/multimap.dll         $CINTDIRS/multimap.so.*
   rm -f $CINTDIRS/multimap2.dll        $CINTDIRS/multimap2.so.*
   rm -f $CINTDIRS/multiset.dll         $CINTDIRS/multiset.so.*
   rm -f $CINTDIRS/stack.dll            $CINTDIRS/stack.so.*
   rm -f $CINTDIRS/queue.dll            $CINTDIRS/queue.so.*
   rm -f $CINTDIRS/valarray.dll         $CINTDIRS/valarray.so.*
   rm -f $CINTDIRS/exception.dll        $CINTDIRS/exception.so.*
   rm -f $CINTDIRS/complex.dll          $CINTDIRS/complex.so.*
}

execute() {
   echo $1
   $1
}

rename() {
   if [ "$SOEXT" != "dll" ]; then
      mv $1.$SOEXT $1.dll;
   fi;
}

cpdllwin32() {
   mv -f bin/stdfunc.dll   $CINTDIRI
   mv -f bin/vector.dll    $CINTDIRS
   mv -f bin/multi*.dll    $CINTDIRS
   mv -f bin/deque.dll     $CINTDIRS
   mv -f bin/map*.dll      $CINTDIRS
   mv -f bin/queue.dll     $CINTDIRS
   mv -f bin/set.dll       $CINTDIRS
   mv -f bin/stack.dll     $CINTDIRS
   mv -f bin/exception.dll $CINTDIRS
   mv -f bin/list.dll      $CINTDIRS
   mv -f bin/complex.dll   $CINTDIRS
}

##### first delete old dll's #####

clean

if [ $PLATFORM = "clean" ]; then
   exit 0;
fi

##### stdfunc.dll  & stdcxxfunc.dll #####

STDFUNCDIR=$CINTDIRL/stdstrct

execute "$CINT -K -w1 -zstdfunc -n$STDFUNCDIR/G__c_stdfunc.c -D__MAKECINT__ \
         -DG__MAKECINT -c-2 -Z0 $STDFUNCDIR/stdfunc.h"
execute "$CC $OPT $CINTCFLAGS -I. -o $STDFUNCDIR/G__c_stdfunc.o \
         -c $STDFUNCDIR/G__c_stdfunc.c"
$MAKELIB $PLATFORM $LD "$LDFLAGS" "$SOFLAGS" stdfunc.$SOEXT \
   $CINTDIRI/stdfunc.$SOEXT "$STDFUNCDIR/G__c_stdfunc.o"
rename $CINTDIRI/stdfunc

#execute "$CINT -w1 -zstdcxxfunc -n$STDFUNCDIR/G__c_stdcxxfunc.cxx \
#         -D__MAKECINT__ -DG__MAKECINT -c-1 -A -Z0 $STDFUNCDIR/stdcxxfunc.h"
#execute "$CXX $OPT $CINTCXXFLAGS -I. -Icint -o $STDFUNCDIR/G__c_stdcxxfunc.o \
#         -c $STDFUNCDIR/G__c_stdcxxfunc.cxx"
#$MAKELIB $PLATFORM $LD "$LDFLAGS" "$SOFLAGS" stdcxxfunc.$SOEXT \
#   $CINTDIRI/stdcxxfunc.$SOEXT "$STDFUNCDIR/G__c_stdcxxfunc.o"
#rename $CINTDIRI/stdcxxfunc

rm -f $STDFUNCDIR/G__c_stdfunc.c $STDFUNCDIR/G__c_stdfunc.h \
      $STDFUNCDIR/G__c_stdfunc.o $STDFUNCDIR/G__c_stdcxxfunc.cxx \
      $STDFUNCDIR/G__c_stdcxxfunc.h $STDFUNCDIR/G__c_stdcxxfunc.o

##### posix.dll #####

if [ $PLATFORM != "win32" ]; then

POSIXDIR=$CINTDIRL/posix

pwd=`pwd`
cd $POSIXDIR
$CC $OPT -o mktypes$EXESUF mktypes.c
./mktypes
rm -f mktypes
cp -f ../../include/systypes.h ../../include/sys/types.h
cd $pwd

execute "$CINT -K -w1 -zposix -n$POSIXDIR/G__c_posix.c -D__MAKECINT__ \
         -DG__MAKECINT -c-2 -Z0 $POSIXDIR/posix.h $POSIXDIR/exten.h"
execute "$CC $OPT $CINTCFLAGS -I. -o $POSIXDIR/G__c_posix.o \
         -c $POSIXDIR/G__c_posix.c"
execute "$CC $OPT $CINTCFLAGS -I. -o $POSIXDIR/exten.o \
         -c $POSIXDIR/exten.c"
$MAKELIB $PLATFORM $LD "$LDFLAGS" "$SOFLAGS" posix.$SOEXT \
   $CINTDIRI/posix.$SOEXT "$POSIXDIR/G__c_posix.o $POSIXDIR/exten.o"
rename $CINTDIRI/posix

rm -f $POSIXDIR/G__c_posix.c $POSIXDIR/G__c_posix.h $POSIXDIR/G__c_posix.o \
      $POSIXDIR/exten.o

##### ipc.dll #####

IPCDIR=$CINTDIRL/ipc

execute "$CINT -K -w1 -zipc -n$IPCDIR/G__c_ipc.c -D__MAKECINT__ \
         -DG__MAKECINT -c-2 -Z0 $IPCDIR/ipcif.h"
execute "$CC $OPT $CINTCFLAGS -I. -o $IPCDIR/G__c_ipc.o -c $IPCDIR/G__c_ipc.c"
$MAKELIB $PLATFORM $LD "$LDFLAGS" "$SOFLAGS" ipc.$SOEXT \
   $CINTDIRI/sys/ipc.$SOEXT "$IPCDIR/G__c_ipc.o"
rename $CINTDIRI/sys/ipc

rm -f $IPCDIR/G__c_ipc.c $IPCDIR/G__c_ipc.h $IPCDIR/G__c_ipc.o

fi

##### STL dlls #####

STLDIR=$CINTDIRL/dll_stl
LINKDEFDIR=metautils/src

FAVOR_SYSINC=-I-
if [ $PLATFORM = "sgi" ]; then
   FAVOR_SYSINC=
fi

if [ "x$GCC_MAJOR" != "x" ] && [ `expr $GCC_MAJOR \>= 4` = 1 ]; then
   INCDIRS="-iquote. -iquote$STLDIR"
else
   INCDIRS="-I. -I$STLDIR $FAVOR_SYSINC"
fi

rm -f $CINTDIRS/*.$SOEXT

#execute "$CINT -w1 -zstring -n$STLDIR/G__cpp_string.cxx -D__MAKECINT__ \
#         -DG__MAKECINT -I$STLDIR -c-1 -A -Z0 $STLDIR/str.h"
#execute "$CXX $OPT $CINTCXXFLAGS $INCDIR -o $STLDIR/G__cpp_string.o \
#         -c $STLDIR/G__cpp_string.cxx $AUXCXXFLAGS"
#$MAKELIB $PLATFORM $LD "$LDFLAGS" "$SOFLAGS" string.$SOEXT \
#   $CINTDIRS/string.$SOEXT $STLDIR/G__cpp_string.o
#rename $CINTDIRS/string

execute "$CINT -w1 -zvector -n$STLDIR/G__cpp_vector.cxx -D__MAKECINT__ \
         -DG__MAKECINT -I$STLDIR -c-1 -A -Z0 $STLDIR/vec.h"
execute "$CXX $OPT $CINTCXXFLAGS $INCDIRS -o $STLDIR/G__cpp_vector.o
         -c $STLDIR/G__cpp_vector.cxx"
execute "$ROOTCINT -f $STLDIR/rootcint_vector.cxx -c vector \
         $LINKDEFDIR/vectorLinkdef.h"
execute "$CXX $OPT $CINTCXXFLAGS $INCDIRS -o $STLDIR/rootcint_vector.o  \
         -c $STLDIR/rootcint_vector.cxx"
execute "$CXX $OPT $CINTCXXFLAGS $INCDIRS -DWHAT=\"vector\" \
         -o $STLDIR/stlLoader_vector.o -c $LINKDEFDIR/stlLoader.cc"
$MAKELIB $PLATFORM $LD "$LDFLAGS" "$SOFLAGS" vector.$SOEXT \
   $CINTDIRS/vector.$SOEXT $STLDIR/G__cpp_vector.o $STLDIR/stlLoader_vector.o
$MAKELIB $PLATFORM $LD "$LDFLAGS" "$SOFLAGS" libvectorDict.$SOEXT \
   lib/libvectorDict.$SOEXT $STLDIR/rootcint_vector.o
rename $CINTDIRS/vector

execute "$CINT -w1 -zlist -n$STLDIR/G__cpp_list.cxx -D__MAKECINT__ \
         -DG__MAKECINT -I$STLDIR -c-1 -A  -Z0 $STLDIR/lst.h"
execute "$CXX $OPT $CINTCXXFLAGS $INCDIRS -o $STLDIR/G__cpp_list.o \
         -c $STLDIR/G__cpp_list.cxx"
execute "$ROOTCINT -f $STLDIR/rootcint_list.cxx -c list \
         $LINKDEFDIR/listLinkdef.h"
execute "$CXX $OPT $CINTCXXFLAGS $INCDIRS -o $STLDIR/rootcint_list.o \
         -c $STLDIR/rootcint_list.cxx"
execute "$CXX $OPT $CINTCXXFLAGS $INCDIRS -DWHAT=\"list\" \
         -o $STLDIR/stlLoader_list.o -c $LINKDEFDIR/stlLoader.cc"
$MAKELIB $PLATFORM $LD "$LDFLAGS" "$SOFLAGS" list.$SOEXT \
   $CINTDIRS/list.$SOEXT $STLDIR/G__cpp_list.o $STLDIR/stlLoader_list.o
$MAKELIB $PLATFORM $LD "$LDFLAGS" "$SOFLAGS" liblistDict.$SOEXT \
   lib/liblistDict.$SOEXT $STLDIR/rootcint_list.o
rename $CINTDIRS/list

execute "$CINT -w1 -zdeque -n$STLDIR/G__cpp_deque.cxx -D__MAKECINT__ \
         -DG__MAKECINT -I$STLDIR -c-1 -A  -Z0 $STLDIR/dqu.h"
execute "$CXX $OPT $CINTCXXFLAGS $INCDIRS -o $STLDIR/G__cpp_deque.o \
         -c $STLDIR/G__cpp_deque.cxx"
execute "$ROOTCINT -f $STLDIR/rootcint_deque.cxx -c deque \
         $LINKDEFDIR/dequeLinkdef.h"
execute "$CXX $OPT $CINTCXXFLAGS $INCDIRS -o $STLDIR/rootcint_deque.o \
         -c $STLDIR/rootcint_deque.cxx"
execute "$CXX $OPT $CINTCXXFLAGS $INCDIRS -DWHAT=\"deque\" \
         -o $STLDIR/stlLoader_deque.o -c $LINKDEFDIR/stlLoader.cc"
$MAKELIB $PLATFORM $LD "$LDFLAGS" "$SOFLAGS" deque.$SOEXT \
   $CINTDIRS/deque.$SOEXT $STLDIR/G__cpp_deque.o  $STLDIR/stlLoader_deque.o 
$MAKELIB $PLATFORM $LD "$LDFLAGS" "$SOFLAGS" libdequeDict.$SOEXT \
   lib/libdequeDict.$SOEXT $STLDIR/rootcint_deque.o
rename $CINTDIRS/deque

execute "$CINT -w1 -zmap -n$STLDIR/G__cpp_map.cxx -D__MAKECINT__ \
         -DG__MAKECINT -I$STLDIR -c-1 -A  -Z0 $STLDIR/mp.h"
execute "$CXX $OPT $CINTCXXFLAGS $INCDIRS -o $STLDIR/G__cpp_map.o \
         -c $STLDIR/G__cpp_map.cxx"
execute "$ROOTCINT -f $STLDIR/rootcint_map.cxx -c map $LINKDEFDIR/mapLinkdef.h"
execute "$CXX $OPT $CINTCXXFLAGS $INCDIRS -o $STLDIR/rootcint_map.o \
         -c $STLDIR/rootcint_map.cxx"
execute "$CXX $OPT $CINTCXXFLAGS $INCDIRS -DWHAT=\"map\" \
         -o $STLDIR/stlLoader_map.o -c $LINKDEFDIR/stlLoader.cc"
$MAKELIB $PLATFORM $LD "$LDFLAGS" "$SOFLAGS" map.$SOEXT \
   $CINTDIRS/map.$SOEXT $STLDIR/G__cpp_map.o $STLDIR/stlLoader_map.o
$MAKELIB $PLATFORM $LD "$LDFLAGS" "$SOFLAGS" libmapDict.$SOEXT \
   lib/libmapDict.$SOEXT $STLDIR/rootcint_map.o
rename $CINTDIRS/map

execute "$CINT -w1 -zmap2 -n$STLDIR/G__cpp_map2.cxx -D__MAKECINT__ \
         -DG__MAKECINT -DG__MAP2 -I$STLDIR -c-1 -A  -Z0 $STLDIR/mp.h"
execute "$CXX $OPT $CINTCXXFLAGS $INCDIRS -o $STLDIR/G__cpp_map2.o \
         -c $STLDIR/G__cpp_map2.cxx"
execute "$ROOTCINT -f $STLDIR/rootcint_map2.cxx -c -DG__MAP2 map \
         $LINKDEFDIR/mapLinkdef.h"
execute "$CXX $OPT $CINTCXXFLAGS $INCDIRS -o $STLDIR/rootcint_map2.o \
         -c $STLDIR/rootcint_map2.cxx"
execute "$CXX $OPT $CINTCXXFLAGS $INCDIRS -DWHAT=\"map2\" \
         -o $STLDIR/stlLoader_map2.o -c $LINKDEFDIR/stlLoader.cc"
$MAKELIB $PLATFORM $LD "$LDFLAGS" "$SOFLAGS" map2.$SOEXT \
   $CINTDIRS/map2.$SOEXT $STLDIR/G__cpp_map2.o $STLDIR/stlLoader_map2.o 
$MAKELIB $PLATFORM $LD "$LDFLAGS" "$SOFLAGS" libmap2Dict.$SOEXT \
   lib/libmap2Dict.$SOEXT $STLDIR/rootcint_map2.o
rename $CINTDIRS/map2

execute "$CINT -w1 -zset -n$STLDIR/G__cpp_set.cxx -D__MAKECINT__ \
         -DG__MAKECINT -I$STLDIR -c-1 -A  -Z0 $STLDIR/st.h"
execute "$CXX $OPT $CINTCXXFLAGS $INCDIRS -o $STLDIR/G__cpp_set.o \
         -c $STLDIR/G__cpp_set.cxx"
execute "$ROOTCINT -f $STLDIR/rootcint_set.cxx -c set $LINKDEFDIR/setLinkdef.h"
execute "$CXX $OPT $CINTCXXFLAGS $INCDIRS -o $STLDIR/rootcint_set.o \
         -c $STLDIR/rootcint_set.cxx"
execute "$CXX $OPT $CINTCXXFLAGS $INCDIRS -DWHAT=\"set\" \
         -o $STLDIR/stlLoader_set.o -c $LINKDEFDIR/stlLoader.cc"
$MAKELIB $PLATFORM $LD "$LDFLAGS" "$SOFLAGS" set.$SOEXT \
   $CINTDIRS/set.$SOEXT $STLDIR/G__cpp_set.o $STLDIR/stlLoader_set.o
$MAKELIB $PLATFORM $LD "$LDFLAGS" "$SOFLAGS" libsetDict.$SOEXT \
   lib/libsetDict.$SOEXT $STLDIR/rootcint_set.o
rename $CINTDIRS/set

execute "$CINT -w1 -zmultimap -n$STLDIR/G__cpp_multimap.cxx -D__MAKECINT__ \
         -DG__MAKECINT -I$STLDIR -c-1 -A  -Z0 $STLDIR/multmp.h"
execute "$CXX $OPT $CINTCXXFLAGS $INCDIRS -o $STLDIR/G__cpp_multimap.o \
         -c $STLDIR/G__cpp_multimap.cxx"
execute "$ROOTCINT -f $STLDIR/rootcint_multimap.cxx -c map \
         $LINKDEFDIR/multimapLinkdef.h"
execute "$CXX $OPT $CINTCXXFLAGS $INCDIRS -o $STLDIR/rootcint_multimap.o \
         -c $STLDIR/rootcint_multimap.cxx"
execute "$CXX $OPT $CINTCXXFLAGS $INCDIRS -DWHAT=\"multimap\" \
         -o $STLDIR/stlLoader_multimap.o -c $LINKDEFDIR/stlLoader.cc"
$MAKELIB $PLATFORM $LD "$LDFLAGS" "$SOFLAGS" multimap.$SOEXT \
   $CINTDIRS/multimap.$SOEXT $STLDIR/G__cpp_multimap.o \
   $STLDIR/stlLoader_multimap.o
$MAKELIB $PLATFORM $LD "$LDFLAGS" "$SOFLAGS" libmultimapDict.$SOEXT \
   lib/libmultimapDict.$SOEXT $STLDIR/rootcint_multimap.o
rename $CINTDIRS/multimap

execute "$CINT -w1 -zmultimap2 -n$STLDIR/G__cpp_multimap2.cxx -D__MAKECINT__ \
         -DG__MAKECINT -DG__MAP2 -I$STLDIR -c-1 -A  -Z0 $STLDIR/multmp.h"
execute "$CXX $OPT $CINTCXXFLAGS $INCDIRS -o $STLDIR/G__cpp_multimap2.o \
         -c $STLDIR/G__cpp_multimap2.cxx"
execute "$ROOTCINT -f $STLDIR/rootcint_multimap2.cxx -c -DG__MAP2 map \
         $LINKDEFDIR/multimapLinkdef.h"
execute "$CXX $OPT $CINTCXXFLAGS $INCDIRS -o $STLDIR/rootcint_multimap2.o \
         -c $STLDIR/rootcint_multimap2.cxx"
execute "$CXX $OPT $CINTCXXFLAGS $INCDIRS -DWHAT=\"multimap2\" \
         -o $STLDIR/stlLoader_multimap2.o -c $LINKDEFDIR/stlLoader.cc"
$MAKELIB $PLATFORM $LD "$LDFLAGS" "$SOFLAGS" multimap2.$SOEXT \
   $CINTDIRS/multimap2.$SOEXT $STLDIR/G__cpp_multimap2.o \
   $STLDIR/stlLoader_multimap2.o
$MAKELIB $PLATFORM $LD "$LDFLAGS" "$SOFLAGS" libmultimap2Dict.$SOEXT \
   lib/libmultimap2Dict.$SOEXT $STLDIR/rootcint_multimap2.o
rename $CINTDIRS/multimap2

execute "$CINT -w1 -zmultiset -n$STLDIR/G__cpp_multiset.cxx -D__MAKECINT__ \
         -DG__MAKECINT -I$STLDIR -c-1 -A  -Z0 $STLDIR/multst.h"
execute "$CXX $OPT $CINTCXXFLAGS $INCDIRS -o $STLDIR/G__cpp_multiset.o \
         -c $STLDIR/G__cpp_multiset.cxx"
execute "$ROOTCINT -f $STLDIR/rootcint_multiset.cxx -c set \
         $LINKDEFDIR/multisetLinkdef.h"
execute "$CXX $OPT $CINTCXXFLAGS $INCDIRS -o $STLDIR/rootcint_multiset.o \
         -c $STLDIR/rootcint_multiset.cxx"
execute "$CXX $OPT $CINTCXXFLAGS $INCDIRS -DWHAT=\"multiset\" \
         -o $STLDIR/stlLoader_multiset.o -c $LINKDEFDIR/stlLoader.cc"
$MAKELIB $PLATFORM $LD "$LDFLAGS" "$SOFLAGS" multiset.$SOEXT \
   $CINTDIRS/multiset.$SOEXT $STLDIR/G__cpp_multiset.o \
   $STLDIR/stlLoader_multiset.o
$MAKELIB $PLATFORM $LD "$LDFLAGS" "$SOFLAGS" libmultisetDict.$SOEXT \
   lib/libmultisetDict.$SOEXT $STLDIR/rootcint_multiset.o
rename $CINTDIRS/multiset

execute "$CINT -w1 -zstack -n$STLDIR/G__cpp_stack.cxx -D__MAKECINT__ \
         -DG__MAKECINT -I$STLDIR -c-1 -A  -Z0 $STLDIR/stk.h"
execute "$CXX $OPT $CINTCXXFLAGS $INCDIRS -o $STLDIR/G__cpp_stack.o \
         -c $STLDIR/G__cpp_stack.cxx"
$MAKELIB $PLATFORM $LD "$LDFLAGS" "$SOFLAGS" stack.$SOEXT \
   $CINTDIRS/stack.$SOEXT $STLDIR/G__cpp_stack.o
rename $CINTDIRS/stack

execute "$CINT -w1 -zqueue -n$STLDIR/G__cpp_queue.cxx -D__MAKECINT__ \
         -DG__MAKECINT -I$STLDIR -c-1 -A  -Z0 $STLDIR/que.h"
execute "$CXX $OPT $CINTCXXFLAGS $INCDIRS -o $STLDIR/G__cpp_queue.o \
         -c $STLDIR/G__cpp_queue.cxx"
$MAKELIB $PLATFORM $LD "$LDFLAGS" "$SOFLAGS" queue.$SOEXT \
   $CINTDIRS/queue.$SOEXT $STLDIR/G__cpp_queue.o
rename $CINTDIRS/queue

#execute "$CINT -w1 -zvalarray -n$STLDIR/G__cpp_valarray.cxx -D__MAKECINT__ \
#         -DG__MAKECINT -I$STLDIR -c-1 -A  -Z0 $STLDIR/vary.h"
#execute "$CXX $OPT $CINTCXXFLAGS $INCDIRS -o $STLDIR/G__cpp_valarray.o \
#         -c $STLDIR/G__cpp_valarray.cxx"
#$MAKELIB $PLATFORM $LD "$LDFLAGS" "$SOFLAGS" valarray.$SOEXT \
#   $CINTDIRS/valarray.$SOEXT $STLDIR/G__cpp_valarray.o
#rename $CINTDIRS/valarray

execute "$CINT -w1 -zexception -n$STLDIR/G__cpp_exception.cxx -D__MAKECINT__ \
         -DG__MAKECINT -I$STLDIR -c-1 -A  -Z0 $STLDIR/cinteh.h"
execute "$CXX $OPT $CINTCXXFLAGS $INCDIRS -o $STLDIR/G__cpp_exception.o \
         -c $STLDIR/G__cpp_exception.cxx"
$MAKELIB $PLATFORM $LD "$LDFLAGS" "$SOFLAGS" exception.$SOEXT \
   $CINTDIRS/exception.$SOEXT $STLDIR/G__cpp_exception.o
rename $CINTDIRS/exception

execute "$CINT -w1 -zcomplex -n$STLDIR/G__cpp_complex.cxx -D__MAKECINT__ \
         -DG__MAKECINT -I$STLDIR -c-1 -A  -Z0 $STLDIR/cmplx.h"
execute "$CXX $OPT $CINTCXXFLAGS $INCDIRS -o $STLDIR/G__cpp_complex.o \
         -c $STLDIR/G__cpp_complex.cxx"
$MAKELIB $PLATFORM $LD "$LDFLAGS" "$SOFLAGS" complex.$SOEXT \
   $CINTDIRS/complex.$SOEXT $STLDIR/G__cpp_complex.o
rename $CINTDIRS/complex

rm -f $STLDIR/G__* $STLDIR/rootcint_*  $STLDIR/stlLoader_*

if [ $PLATFORM = "win32" ]; then
   cpdllwin32
fi

exit 0
