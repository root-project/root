#! /bin/sh

# Script to generate a shared library.
# Called by main Makefile.
#
# Author: Fons Rademakers, 29/2/2000

if [ "$1" = "-v" ] ; then
   MAJOR=$2
   MINOR=$3
   REVIS=$4
   shift
   shift
   shift
   shift
fi

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
      #$LD $SOFLAGS$SONAME $LDFLAGS -o $LIB $OBJS hist/src/*.o graf/src/*.o \
      #   g3d/src/*.o matrix/src/*.o $EXTRA
      ld -L/usr/lib/cmplrs/cxx -rpath /usr/lib/cmplrs/cxx \
         -expect_unresolved "*" -g0 -O1 -msym -shared \
         /usr/lib/cmplrs/cc/crt0.o /usr/lib/cmplrs/cxx/_main.o \
         -o $LIB $OBJS hist/src/*.o graf/src/*.o g3d/src/*.o matrix/src/*.o $EXTRA
   elif [ $LIB = "lib/libHist.so" ]   || [ $LIB = "lib/libGraf.so" ] || \
        [ $LIB = "lib/libGraf3d.so" ] || [ $LIB = "lib/libMatrix.so" ]; then
      #$LD $SOFLAGS$SONAME $LDFLAGS -o $LIB /usr/lib/cmplrs/cc/crt0.o
      ld -L/usr/lib/cmplrs/cxx -rpath /usr/lib/cmplrs/cxx \
         -expect_unresolved "*" -g0 -O1 -msym -shared \
         /usr/lib/cmplrs/cc/crt0.o /usr/lib/cmplrs/cxx/_main.o \
         -o $LIB
   else
      #$LD $SOFLAGS$SONAME $LDFLAGS -o $LIB $OBJS $EXTRA
      ld -L/usr/lib/cmplrs/cxx -rpath /usr/lib/cmplrs/cxx \
         -expect_unresolved "*" -g0 -O1 -msym -shared \
         /usr/lib/cmplrs/cc/crt0.o /usr/lib/cmplrs/cxx/_main.o \
         -o $LIB $OBJS $EXTRA
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
   if [ "x$MAJOR" = "x" ] ; then
      echo $LD $SOFLAGS$SONAME $LDFLAGS -o $LIB $OBJS $EXTRA
      $LD $SOFLAGS$SONAME $LDFLAGS -o $LIB $OBJS $EXTRA
   else
      echo $LD $SOFLAGS$SONAME.$MAJOR.$MINOR $LDFLAGS -o $LIB.$MAJOR.$MINOR $OBJS $EXTRA
      $LD $SOFLAGS$SONAME.$MAJOR.$MINOR $LDFLAGS \
         -o $LIB.$MAJOR.$MINOR $OBJS $EXTRA
   fi
fi

linkstat=$?
if [ $linkstat -ne 0 ]; then
   exit $linkstat
fi

if [ "x$MAJOR" != "x" ] && [ -f $LIB.$MAJOR.$MINOR ]; then
   ln -fs $SONAME.$MAJOR.$MINOR $LIB.$MAJOR
   ln -fs $SONAME.$MAJOR        $LIB
fi

if [ $PLATFORM = "hpux" ]; then
   chmod 555 $LIB
fi

echo "==> $LIB done"

exit 0
