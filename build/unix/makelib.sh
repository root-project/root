#! /bin/sh

# Script to generate a shared library.
# Called by main Makefile.
#
# Author: Fons Rademakers, 29/2/2000

PLATFORM=$1
LD=$2
LDFLAGS=$3
SOFLAGS=$4
SONAME=$5
LIB=$6
OBJS=$7
EXTRA=$8

rm -f $LIB

if [ $PLATFORM = "aix" ]; then
   if [ $LD = "xlC" ]; then
      if [ $LIB = "lib/libCint.a" ]; then
         echo /usr/ibmcxx/bin/makeC++SharedLib -o $LIB -p 0 $OBJS -Llib $EXTRA
         /usr/ibmcxx/bin/makeC++SharedLib -o $LIB -p 0 $OBJS -Llib $EXTRA
      elif [ $LIB = "lib/libCore.a" ]; then
         echo /usr/ibmcxx/bin/makeC++SharedLib -o $LIB -p 0 $OBJS -Llib $EXTRA -lCint
         /usr/ibmcxx/bin/makeC++SharedLib -o $LIB -p 0 $OBJS -Llib $EXTRA -lCint
      else
         echo /usr/ibmcxx/bin/makeC++SharedLib -o $LIB -p 0 $OBJS -Llib $EXTRA -lCore -lCint
         /usr/ibmcxx/bin/makeC++SharedLib -o $LIB -p 0 $OBJS -Llib $EXTRA -lCore -lCint
      fi
   fi
elif [ $PLATFORM = "alpha" ] && [ $LD = "cxx" ]; then
   echo $LD $SOFLAGS$SONAME $LDFLAGS -o $LIB $OBJS $EXTRA
   if [ $LIB = "lib/libCore.so" ]; then
      $LD $SOFLAGS$SONAME $LDFLAGS -o $LIB $OBJS hist/src/*.o graf/src/*.o \
         g3d/src/*.o $EXTRA
   elif [ $LIB = "lib/libHist.so" ] || [ $LIB = "lib/libGraf.so" ] || \
        [ $LIB = "lib/libGraf3d.so" ]; then
      $LD $SOFLAGS$SONAME $LDFLAGS -o $LIB /usr/lib/cmplrs/cc/crt0.o
   else
      $LD $SOFLAGS$SONAME $LDFLAGS -o $LIB $OBJS $EXTRA
   fi
elif [ $PLATFORM = "alphaegcs" ] || [ $PLATFORM = "hpux" ] || \
     [ $PLATFORM = "solaris" ]   || [ $PLATFORM = "sgi" ]; then
   echo $LD $SOFLAGS $LDFLAGS -o $LIB $OBJS $EXTRA
   $LD $SOFLAGS $LDFLAGS -o $LIB $OBJS $EXTRA
elif [ $PLATFORM = "lynxos" ]; then
   echo ar rv $LIB $OBJS $EXTRA
   ar rv $LIB $OBJS $EXTRA
elif [ $PLATFORM = "fbsd" ]; then
   echo $LD $SOFLAGS $LDFLAGS -o $LIB $OBJS $EXTRA
   $LD $SOFLAGS $LDFLAGS -o $LIB `lorder $OBJS | tsort -q` $EXTRA
# for elf:  echo $PLATFORM: $LD $SOFLAGS$SONAME $LDFLAGS -o $LIB $OBJS
# for elf:  $LD $SOFLAGS$SONAME $LDFLAGS -o $LIB `lorder $OBJS | tsort -q`
elif [ $LD = "KCC" ]; then
   echo $LD $LDFLAGS -o $LIB $OBJS $EXTRA
   $LD $LDFLAGS -o $LIB $OBJS $EXTRA
else
   echo $LD $SOFLAGS$SONAME $LDFLAGS -o $LIB $OBJS $EXTRA
   $LD $SOFLAGS$SONAME $LDFLAGS -o $LIB $OBJS $EXTRA
fi

linkstat=$?
if [ $linkstat -ne 0 ]; then
   exit $linkstat
fi

if [ $PLATFORM = "hpux" ]; then
   chmod 555 $LIB
fi

echo "==> $LIB done"

exit 0
