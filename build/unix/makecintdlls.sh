#! /bin/sh

# Script to create auxiliary CINT dll's.
# Called by cint/Module.mk.
#
# Author: Fons Rademakers, 27/7/2000

PLATFORM=$1
CINT=$2
MAKELIB=$3
CXX=$4
CC=$5
LD=$6
OPT=$7
CINTCXXFLAGS=$8
CINTCFLAGS=$9
shift
LDFLAGS=$9
shift
SOFLAGS=$9
shift
SOEXT=$9

CINTDIRL=cint/lib
CINTDIRI=cint/include
CINTDIRS=cint/stl

rename() {
   if [ "$SOEXT" != "dll" ]; then
      mv $1.$SOEXT $1.dll;
   fi;
}

##### long.dl (note .dl not .dll) #####

LONGDIR=$CINTDIRL/longlong

$CINT -w1 -zlong -n$LONGDIR/G__cpp_long.cxx -D__MAKECINT__ \
   -DG__MAKECINT -c-1 -A -Z0 $LONGDIR/longlong.h
$CXX $OPT $CINTCXXFLAGS -I. -Icint -o $LONGDIR/G__cpp_long.o \
   -c $LONGDIR/G__cpp_long.cxx
$MAKELIB $PLATFORM $LD "$LDFLAGS" "$SOFLAGS" long.$SOEXT $CINTDIRI/long.$SOEXT \
   $LONGDIR/G__cpp_long.o
mv $CINTDIRI/long.$SOEXT $CINTDIRI/long.dl

rm -f $LONGDIR/G__cpp_long.cxx $LONGDIR/G__cpp_long.h $LONGDIR/G__cpp_long.o


##### posix.dll #####

POSIXDIR=$CINTDIRL/posix

pwd=`pwd`
cd $POSIXDIR
$CC $OPT -o mktypes mktypes.c
mktypes
rm -f mktypes
cp -f ../../include/systypes.h ../../include/sys/types.h
cd $pwd

$CINT -K -w1 -zposix -n$POSIXDIR/G__c_posix.c -D__MAKECINT__ \
   -DG__MAKECINT -c-2 -Z0 $POSIXDIR/posix.h $POSIXDIR/exten.h
$CC $OPT $CINTCFLAGS -I. -o $POSIXDIR/G__c_posix.o \
   -c $POSIXDIR/G__c_posix.c
$CC $OPT $CINTCFLAGS -I. -o $POSIXDIR/exten.o \
   -c $POSIXDIR/exten.c
$MAKELIB $PLATFORM $LD "$LDFLAGS" "$SOFLAGS" posix.$SOEXT $CINTDIRI/posix.$SOEXT \
   "$POSIXDIR/G__c_posix.o $POSIXDIR/exten.o"
rename $CINTDIRI/posix

rm -f $POSIXDIR/G__c_posix.c $POSIXDIR/G__c_posix.h $POSIXDIR/G__c_posix.o \
      $POSIXDIR/exten.o

##### STL dlls #####

STLDIR=$CINTDIRL/dll_stl

rm -f $CINTDIRS/*.$SOEXT

$CINT -w1 -zstring -n$STLDIR/G__cpp_string.cxx -D__MAKECINT__ \
   -DG__MAKECINT -c-1 -A -M0x10 -Z0 $STLDIR/str.h
$CXX $OPT $CINTCXXFLAGS -I. -I- -o $STLDIR/G__cpp_string.o \
   -c $STLDIR/G__cpp_string.cxx
$MAKELIB $PLATFORM $LD "$LDFLAGS" "$SOFLAGS" string.$SOEXT \
   $CINTDIRS/string.$SOEXT $STLDIR/G__cpp_string.o
rename $CINTDIRS/string

$CINT -w1 -zvector -n$STLDIR/G__cpp_vector.cxx -D__MAKECINT__ \
   -DG__MAKECINT -c-1 -A -M0x10 -Z0 $STLDIR/vec.h
$CXX $OPT $CINTCXXFLAGS -I. -I- -o $STLDIR/G__cpp_vector.o \
   -c $STLDIR/G__cpp_vector.cxx
$MAKELIB $PLATFORM $LD "$LDFLAGS" "$SOFLAGS" vector.$SOEXT \
   $CINTDIRS/vector.$SOEXT $STLDIR/G__cpp_vector.o
rename $CINTDIRS/vector

$CINT -w1 -zlist -n$STLDIR/G__cpp_list.cxx -D__MAKECINT__ \
   -DG__MAKECINT -c-1 -A -M0x10 -Z0 $STLDIR/lst.h
$CXX $OPT $CINTCXXFLAGS -I. -I- -o $STLDIR/G__cpp_list.o \
   -c $STLDIR/G__cpp_list.cxx
$MAKELIB $PLATFORM $LD "$LDFLAGS" "$SOFLAGS" list.$SOEXT \
   $CINTDIRS/list.$SOEXT $STLDIR/G__cpp_list.o
rename $CINTDIRS/list

$CINT -w1 -zdeque -n$STLDIR/G__cpp_deque.cxx -D__MAKECINT__ \
   -DG__MAKECINT -c-1 -A -M0x10 -Z0 $STLDIR/dqu.h
$CXX $OPT $CINTCXXFLAGS -I. -I- -o $STLDIR/G__cpp_deque.o \
   -c $STLDIR/G__cpp_deque.cxx
$MAKELIB $PLATFORM $LD "$LDFLAGS" "$SOFLAGS" deque.$SOEXT \
   $CINTDIRS/deque.$SOEXT $STLDIR/G__cpp_deque.o
rename $CINTDIRS/deque

$CINT -w1 -zmap -n$STLDIR/G__cpp_map.cxx -D__MAKECINT__ \
   -DG__MAKECINT -c-1 -A -M0x10 -Z0 $STLDIR/mp.h
$CXX $OPT $CINTCXXFLAGS -I. -I- -o $STLDIR/G__cpp_map.o \
   -c $STLDIR/G__cpp_map.cxx
$MAKELIB $PLATFORM $LD "$LDFLAGS" "$SOFLAGS" map.$SOEXT \
   $CINTDIRS/map.$SOEXT $STLDIR/G__cpp_map.o
rename $CINTDIRS/map

$CINT -w1 -zset -n$STLDIR/G__cpp_set.cxx -D__MAKECINT__ \
   -DG__MAKECINT -c-1 -A -M0x10 -Z0 $STLDIR/st.h
$CXX $OPT $CINTCXXFLAGS -I. -I- -o $STLDIR/G__cpp_set.o \
   -c $STLDIR/G__cpp_set.cxx
$MAKELIB $PLATFORM $LD "$LDFLAGS" "$SOFLAGS" set.$SOEXT \
   $CINTDIRS/set.$SOEXT $STLDIR/G__cpp_set.o
rename $CINTDIRS/set

$CINT -w1 -zmultimap -n$STLDIR/G__cpp_multimap.cxx -D__MAKECINT__ \
   -DG__MAKECINT -c-1 -A -M0x10 -Z0 $STLDIR/multmp.h
$CXX $OPT $CINTCXXFLAGS -I. -I- -o $STLDIR/G__cpp_multimap.o \
   -c $STLDIR/G__cpp_multimap.cxx
$MAKELIB $PLATFORM $LD "$LDFLAGS" "$SOFLAGS" multimap.$SOEXT \
   $CINTDIRS/multimap.$SOEXT $STLDIR/G__cpp_multimap.o
rename $CINTDIRS/multimap

$CINT -w1 -zmultiset -n$STLDIR/G__cpp_multiset.cxx -D__MAKECINT__ \
   -DG__MAKECINT -c-1 -A -M0x10 -Z0 $STLDIR/multst.h
$CXX $OPT $CINTCXXFLAGS -I. -I- -o $STLDIR/G__cpp_multiset.o \
   -c $STLDIR/G__cpp_multiset.cxx
$MAKELIB $PLATFORM $LD "$LDFLAGS" "$SOFLAGS" multiset.$SOEXT \
   $CINTDIRS/multiset.$SOEXT $STLDIR/G__cpp_multiset.o
rename $CINTDIRS/multiset

$CINT -w1 -zstack -n$STLDIR/G__cpp_stack.cxx -D__MAKECINT__ \
   -DG__MAKECINT -c-1 -A -M0x10 -Z0 $STLDIR/stk.h
$CXX $OPT $CINTCXXFLAGS -I. -I- -o $STLDIR/G__cpp_stack.o \
   -c $STLDIR/G__cpp_stack.cxx
$MAKELIB $PLATFORM $LD "$LDFLAGS" "$SOFLAGS" stack.$SOEXT \
   $CINTDIRS/stack.$SOEXT $STLDIR/G__cpp_stack.o
rename $CINTDIRS/stack

$CINT -w1 -zqueue -n$STLDIR/G__cpp_queue.cxx -D__MAKECINT__ \
   -DG__MAKECINT -c-1 -A -M0x10 -Z0 $STLDIR/que.h
$CXX $OPT $CINTCXXFLAGS -I. -I- -o $STLDIR/G__cpp_queue.o \
   -c $STLDIR/G__cpp_queue.cxx
$MAKELIB $PLATFORM $LD "$LDFLAGS" "$SOFLAGS" queue.$SOEXT \
   $CINTDIRS/queue.$SOEXT $STLDIR/G__cpp_queue.o
rename $CINTDIRS/queue

#$CINT -w1 -zvalarray -n$STLDIR/G__cpp_valarray.cxx -D__MAKECINT__ \
#   -DG__MAKECINT -c-1 -A -M0x10 -Z0 $STLDIR/vary.h
#$CXX $OPT $CINTCXXFLAGS -I. -I- -o $STLDIR/G__cpp_valarray.o \
#   -c $STLDIR/G__cpp_valarray.cxx
#$MAKELIB $PLATFORM $LD "$LDFLAGS" "$SOFLAGS" valarray.$SOEXT \
#   $CINTDIRS/valarray.$SOEXT $STLDIR/G__cpp_valarray.o
#rename $CINTDIRS/valarray

$CINT -w1 -zexception -n$STLDIR/G__cpp_exception.cxx -D__MAKECINT__ \
   -DG__MAKECINT -c-1 -A -M0x10 -Z0 $STLDIR/eh.h
$CXX $OPT $CINTCXXFLAGS -I. -I- -o $STLDIR/G__cpp_exception.o \
   -c $STLDIR/G__cpp_exception.cxx
$MAKELIB $PLATFORM $LD "$LDFLAGS" "$SOFLAGS" exception.$SOEXT \
   $CINTDIRS/exception.$SOEXT $STLDIR/G__cpp_exception.o
rename $CINTDIRS/exception

rm -f $STLDIR/G__*

exit 0

