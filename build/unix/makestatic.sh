#! /bin/sh

# Script to generate a archive library and statically linked executable.
# Called by main Makefile.
#
# Author: Fons Rademakers, 21/01/2001

PLATFORM=$1
CXX=$2
CC=$3
LD=$4
LDFLAGS=$5
XLIBS=$6
SYSLIBS=$7

ROOTALIB=lib/libRoot.a
ROOTAEXE=bin/roota

rm -f $ROOTALIB $ROOTAEXE

excl="main proofd rootd rootx pythia pythia6 venus mysql pgsql rfio sapdb \
      hbook utils srputils krb5auth dcache x11ttf"

objs=""
gobjs=""
for i in * ; do
   for j in $excl ; do
      if [ $j = $i ]; then
         continue 2
      fi
   done
   ls $i/src/*.o > /dev/null 2>&1 && objs="$objs $i/src/*.o"
   ls $i/src/G__*.o > /dev/null 2>&1 && gobjs="$gobjs $i/src/G__*.o"
done

echo "Making $ROOTALIB..."
ar rv $ROOTALIB cint/main/G__setup.o $objs > /dev/null 2>&1

arstat=$?
if [ $arstat -ne 0 ]; then
   exit $arstat
fi

dummyc=R__dummy.c
dummyo=""
if [ $PLATFORM = "alpha" ] && [ $CXX = "cxx" ]; then
   echo 'void dnet_conn() {}' > $dummyc
   $CC -c $dummyc
   dummyo=R__dummy.o
fi

echo "Making $ROOTAEXE..."
$LD $LDFLAGS -o $ROOTAEXE main/src/rmain.o $dummyo $gobjs $ROOTALIB \
   $XLIBS $SYSLIBS

linkstat=$?
if [ $linkstat -ne 0 ]; then
   exit $linkstat
fi

rm -f $dummyc $dummyo

exit 0
