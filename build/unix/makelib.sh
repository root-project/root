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
   makeshared="/usr/ibmcxx/bin/makeC++SharedLib"
fi
if [ $PLATFORM = "aix5" ]; then
   makeshared="/usr/vacpp/bin/makeC++SharedLib"
fi

if [ $PLATFORM = "aix" ] || [ $PLATFORM = "aix5" ]; then
   if [ $LD = "xlC" ]; then
      if [ $LIB = "lib/libCint.a" ]; then
         echo $makeshared -o $LIB -p 0 $OBJS -Llib $EXTRA
         $makeshared -o $LIB -p 0 $OBJS -Llib $EXTRA
      elif [ $LIB = "lib/libCore.a" ]; then
         echo $makeshared -o $LIB -p 0 $OBJS -Llib $EXTRA -lCint
         $makeshared -o $LIB -p 0 $OBJS -Llib $EXTRA -lCint
      elif [ $LIB = "lib/libGeomPainter.a" ]; then
         echo $makeshared -o $LIB -p 0 $OBJS -Llib $EXTRA -lCint -lGeom
         $makeshared -o $LIB -p 0 $OBJS -Llib $EXTRA -lCint -lGeom
      else
         echo $makeshared -o $LIB -p 0 $OBJS -Llib $EXTRA -lCore -lCint
         $makeshared -o $LIB -p 0 $OBJS -Llib $EXTRA -lCore -lCint
      fi
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
elif [ $PLATFORM = "macosx" ]; then
   # We need two library files: a .dylib to link to and a .so to load
   BUNDLE=`echo $LIB | sed s/.dylib/.so/`
   echo $LD $SOFLAGS $SONAME -o $LIB -ldl $OBJS $EXTRA
   $LD $SOFLAGS $SONAME -o $LIB -ldl $OBJS $EXTRA
   if [ "x`echo $SOFLAGS | grep -- '-g'`" != "x" ]; then
      opt=-g
   else
      opt=-O
   fi
   echo $LD $opt -bundle -flat_namespace -undefined suppress -install_name $BUNDLE -o $BUNDLE -ldl $OBJS $EXTRA
   $LD $opt -bundle -flat_namespace -undefined suppress -install_name $BUNDLE \
      -o $BUNDLE -ldl $OBJS $EXTRA
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
