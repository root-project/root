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

CINTDIRL=cint/lib
CINTDIRI=cint/include
CINTDIRS=cint/stl


##### long.dl (note .dl not .dll) #####

LONGDIR=$CINTDIRL/longlong

$CINT -w1 -zlong -n$LONGDIR/G__cpp_long.cxx -D__MAKECINT__ \
   -DG__MAKECINT -c-1 -A -Z0 $LONGDIR/longlong.h
$CXX $OPT $CINTCXXFLAGS -I. -Icint -o $LONGDIR/G__cpp_long.o \
   -c $LONGDIR/G__cpp_long.cxx
$MAKELIB $PLATFORM $LD "$LDFLAGS" "$SOFLAGS" long.dl $CINTDIRI/long.dl \
   $LONGDIR/G__cpp_long.o

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
$MAKELIB $PLATFORM $LD "$LDFLAGS" "$SOFLAGS" posix.dll $CINTDIRI/posix.dll \
   "$POSIXDIR/G__c_posix.o $POSIXDIR/exten.o"

rm -f $POSIXDIR/G__c_posix.c $POSIXDIR/G__c_posix.h $POSIXDIR/G__c_posix.o \
      $POSIXDIR/exten.o

##### STL dlls #####

STLDIR=$CINTDIRL/dll_stl

rm -f $CINTDIRS/*.dll

$CINT -w1 -zstring -n$STLDIR/G__cpp_string.cxx -D__MAKECINT__ \
   -DG__MAKECINT -c-1 -A -M0x10 -Z0 $STLDIR/str.h
$CXX $OPT $CINTCXXFLAGS -I. -I- -o $STLDIR/G__cpp_string.o \
   -c $STLDIR/G__cpp_string.cxx
$MAKELIB $PLATFORM $LD "$LDFLAGS" "$SOFLAGS" string.dll \
   $CINTDIRS/string.dll $STLDIR/G__cpp_string.o

$CINT -w1 -zvector -n$STLDIR/G__cpp_vector.cxx -D__MAKECINT__ \
   -DG__MAKECINT -c-1 -A -M0x10 -Z0 $STLDIR/vec.h
$CXX $OPT $CINTCXXFLAGS -I. -I- -o $STLDIR/G__cpp_vector.o \
   -c $STLDIR/G__cpp_vector.cxx
$MAKELIB $PLATFORM $LD "$LDFLAGS" "$SOFLAGS" vector.dll \
   $CINTDIRS/vector.dll $STLDIR/G__cpp_vector.o

$CINT -w1 -zlist -n$STLDIR/G__cpp_list.cxx -D__MAKECINT__ \
   -DG__MAKECINT -c-1 -A -M0x10 -Z0 $STLDIR/lst.h
$CXX $OPT $CINTCXXFLAGS -I. -I- -o $STLDIR/G__cpp_list.o \
   -c $STLDIR/G__cpp_list.cxx
$MAKELIB $PLATFORM $LD "$LDFLAGS" "$SOFLAGS" list.dll \
   $CINTDIRS/list.dll $STLDIR/G__cpp_list.o

$CINT -w1 -zdeque -n$STLDIR/G__cpp_deque.cxx -D__MAKECINT__ \
   -DG__MAKECINT -c-1 -A -M0x10 -Z0 $STLDIR/dqu.h
$CXX $OPT $CINTCXXFLAGS -I. -I- -o $STLDIR/G__cpp_deque.o \
   -c $STLDIR/G__cpp_deque.cxx
$MAKELIB $PLATFORM $LD "$LDFLAGS" "$SOFLAGS" deque.dll \
   $CINTDIRS/deque.dll $STLDIR/G__cpp_deque.o

$CINT -w1 -zmap -n$STLDIR/G__cpp_map.cxx -D__MAKECINT__ \
   -DG__MAKECINT -c-1 -A -M0x10 -Z0 $STLDIR/mp.h
$CXX $OPT $CINTCXXFLAGS -I. -I- -o $STLDIR/G__cpp_map.o \
   -c $STLDIR/G__cpp_map.cxx
$MAKELIB $PLATFORM $LD "$LDFLAGS" "$SOFLAGS" map.dll \
   $CINTDIRS/map.dll $STLDIR/G__cpp_map.o

$CINT -w1 -zset -n$STLDIR/G__cpp_set.cxx -D__MAKECINT__ \
   -DG__MAKECINT -c-1 -A -M0x10 -Z0 $STLDIR/st.h
$CXX $OPT $CINTCXXFLAGS -I. -I- -o $STLDIR/G__cpp_set.o \
   -c $STLDIR/G__cpp_set.cxx
$MAKELIB $PLATFORM $LD "$LDFLAGS" "$SOFLAGS" set.dll \
   $CINTDIRS/set.dll $STLDIR/G__cpp_set.o

$CINT -w1 -zmultimap -n$STLDIR/G__cpp_multimap.cxx -D__MAKECINT__ \
   -DG__MAKECINT -c-1 -A -M0x10 -Z0 $STLDIR/multmp.h
$CXX $OPT $CINTCXXFLAGS -I. -I- -o $STLDIR/G__cpp_multimap.o \
   -c $STLDIR/G__cpp_multimap.cxx
$MAKELIB $PLATFORM $LD "$LDFLAGS" "$SOFLAGS" multimap.dll \
   $CINTDIRS/multimap.dll $STLDIR/G__cpp_multimap.o

$CINT -w1 -zmultiset -n$STLDIR/G__cpp_multiset.cxx -D__MAKECINT__ \
   -DG__MAKECINT -c-1 -A -M0x10 -Z0 $STLDIR/multst.h
$CXX $OPT $CINTCXXFLAGS -I. -I- -o $STLDIR/G__cpp_multiset.o \
   -c $STLDIR/G__cpp_multiset.cxx
$MAKELIB $PLATFORM $LD "$LDFLAGS" "$SOFLAGS" multiset.dll \
   $CINTDIRS/multiset.dll $STLDIR/G__cpp_multiset.o

$CINT -w1 -zstack -n$STLDIR/G__cpp_stack.cxx -D__MAKECINT__ \
   -DG__MAKECINT -c-1 -A -M0x10 -Z0 $STLDIR/stk.h
$CXX $OPT $CINTCXXFLAGS -I. -I- -o $STLDIR/G__cpp_stack.o \
   -c $STLDIR/G__cpp_stack.cxx
$MAKELIB $PLATFORM $LD "$LDFLAGS" "$SOFLAGS" stack.dll \
   $CINTDIRS/stack.dll $STLDIR/G__cpp_stack.o

$CINT -w1 -zqueue -n$STLDIR/G__cpp_queue.cxx -D__MAKECINT__ \
   -DG__MAKECINT -c-1 -A -M0x10 -Z0 $STLDIR/que.h
$CXX $OPT $CINTCXXFLAGS -I. -I- -o $STLDIR/G__cpp_queue.o \
   -c $STLDIR/G__cpp_queue.cxx
$MAKELIB $PLATFORM $LD "$LDFLAGS" "$SOFLAGS" queue.dll \
   $CINTDIRS/queue.dll $STLDIR/G__cpp_queue.o

$CINT -w1 -zvalarray -n$STLDIR/G__cpp_valarray.cxx -D__MAKECINT__ \
   -DG__MAKECINT -c-1 -A -M0x10 -Z0 $STLDIR/vary.h
$CXX $OPT $CINTCXXFLAGS -I. -I- -o $STLDIR/G__cpp_valarray.o \
   -c $STLDIR/G__cpp_valarray.cxx
$MAKELIB $PLATFORM $LD "$LDFLAGS" "$SOFLAGS" valarray.dll \
   $CINTDIRS/valarray.dll $STLDIR/G__cpp_valarray.o

$CINT -w1 -zexception -n$STLDIR/G__cpp_exception.cxx -D__MAKECINT__ \
   -DG__MAKECINT -c-1 -A -M0x10 -Z0 $STLDIR/eh.h
$CXX $OPT $CINTCXXFLAGS -I. -I- -o $STLDIR/G__cpp_exception.o \
   -c $STLDIR/G__cpp_exception.cxx
$MAKELIB $PLATFORM $LD "$LDFLAGS" "$SOFLAGS" exception.dll \
   $CINTDIRS/exception.dll $STLDIR/G__cpp_exception.o

rm -f $STLDIR/G__*

exit 0

