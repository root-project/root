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

if [ "$1" = "-x" ] ; then
   EXPLICIT="yes"
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

if [ "x$EXPLICIT" = "xyes" ]; then
   if [ $LIB != "lib/libCint.so" ]; then
      if [ $LIB != "lib/libCore.so" ]; then
         EXPLLNKCORE="-Llib -lCore -lCint"
      else
         EXPLLNKCORE="-Llib -lCint"
      fi
   fi
fi

if [ $PLATFORM = "aix" ] || [ $PLATFORM = "aix5" ]; then
   if [ $LD = "xlC" ]; then
      EXPLLNKCORE=
      if [ $LIB != "lib/libCint.a" ]; then
         if [ $LIB != "lib/libCore.a" ]; then
            EXPLLNKCORE="-Llib -lCore -lCint"
         else
            EXPLLNKCORE="-Llib -lCint"
         fi
      fi

      echo $makeshared -o $LIB -p 0 $OBJS $EXTRA $EXPLLNKCORE
      $makeshared -o $LIB -p 0 $OBJS $EXTRA $EXPLLNKCORE
   fi
elif [ $PLATFORM = "alphaegcs" ] || [ $PLATFORM = "hpux" ] || \
     [ $PLATFORM = "solaris" ]   || [ $PLATFORM = "sgi" ]; then
   echo $LD $SOFLAGS $LDFLAGS -o $LIB $OBJS $EXTRA $EXPLLNKCORE
   $LD $SOFLAGS $LDFLAGS -o $LIB $OBJS $EXTRA $EXPLLNKCORE
elif [ $PLATFORM = "lynxos" ]; then
   echo ar rv $LIB $OBJS $EXTRA
   ar rv $LIB $OBJS $EXTRA
elif [ $PLATFORM = "fbsd" ]; then
    # for aout: echo $LD $SOFLAGS $LDFLAGS -o $LIB $OBJS $EXTRA $EXPLLNKCORE
    # for aout: $LD $SOFLAGS $LDFLAGS -o $LIB `lorder $OBJS | tsort -q` $EXTRA $EXPLLNKCORE
    echo $LD $SOFLAGS$SONAME.$MAJOR.$MINOR $LDFLAGS -o $LIB.$MAJOR.$MINOR `lorder $OBJS | tsort -q` $EXTRA $EXPLLNKCORE
    $LD $SOFLAGS$SONAME.$MAJOR.$MINOR $LDFLAGS -o $LIB.$MAJOR.$MINOR `lorder $OBJS | tsort -q` $EXTRA $EXPLLNKCORE
elif [ $PLATFORM = "macosx" ]; then
   macosx_minor=`sw_vers -productVersion | cut -d'.' -f2`
   if [ $LD = "xlC" ]; then
      # if we are using xlc, then the linker for shared libs is still g++
      LD=g++
   fi
   # Look for a fink installation
   FINKDIR=`which fink 2>&1 | sed -ne "s/\/bin\/fink//p"`
   if [ $macosx_minor -ge 3 ]; then
      unset LD_PREBIND
      export MACOSX_DEPLOYMENT_TARGET=10.3
   fi
   # We need two library files: a .dylib to link to and a .so to load
   BUNDLE=`echo $LIB | sed s/.dylib/.so/`
   echo $LD $SOFLAGS$SONAME -o $LIB $OBJS \
	`[ -d ${FINKDIR}/lib ] && echo -L${FINKDIR}/lib` -ldl $EXTRA
   $LD $SOFLAGS$SONAME -o $LIB $OBJS \
	`[ -d ${FINKDIR}/lib ] && echo -L${FINKDIR}/lib` -ldl $EXTRA
   if [ "x`echo $SOFLAGS | grep -- '-g'`" != "x" ]; then
      opt=-g
   else
      opt=-O
   fi
   if [ $macosx_minor -ge 3 ]; then
      echo $LD $opt -bundle -flat_namespace -undefined dynamic_lookup -o $BUNDLE \
	   $OBJS `[ -d ${FINKDIR}/lib ] && echo -L${FINKDIR}/lib` -ldl $EXTRA
      $LD $opt -bundle -flat_namespace -undefined dynamic_lookup -o $BUNDLE \
	   $OBJS `[ -d ${FINKDIR}/lib ] && echo -L${FINKDIR}/lib` -ldl $EXTRA
   else
      echo $LD $opt -bundle -flat_namespace -undefined suppress -o $BUNDLE \
	   $OBJS `[ -d ${FINKDIR}/lib ] && echo -L${FINKDIR}/lib` -ldl $EXTRA
      $LD $opt -bundle -flat_namespace -undefined suppress -o $BUNDLE \
	   $OBJS `[ -d ${FINKDIR}/lib ] && echo -L${FINKDIR}/lib` -ldl $EXTRA
   fi
elif [ $LD = "KCC" ]; then
   echo $LD $LDFLAGS -o $LIB $OBJS $EXTRA $EXPLLNKCORE
   $LD $LDFLAGS -o $LIB $OBJS $EXTRA $EXPLLNKCORE
elif [ $LD = "build/unix/wingcc_ld.sh" ]; then
   EXPLLNKCORE=
   if [ $SONAME != "libCint.dll" ]; then
      if [ $SONAME != "libCore.dll" ]; then
         EXPLLNKCORE="-Llib -lCore -lCint"
      else
         EXPLLNKCORE="-Llib -lCint"
      fi
   fi
   line="$LD $SOFLAGS$SONAME $LDFLAGS -o $LIB $OBJS \
         $EXTRA $EXPLLNKCORE"
   echo $line
   $line
else
   if [ "x$MAJOR" = "x" ] ; then
      echo $LD $SOFLAGS$SONAME $LDFLAGS -o $LIB $OBJS $EXTRA $EXPLLNKCORE
      $LD $SOFLAGS$SONAME $LDFLAGS -o $LIB $OBJS $EXTRA $EXPLLNKCORE
   else
      echo $LD $SOFLAGS$SONAME.$MAJOR.$MINOR $LDFLAGS -o $LIB.$MAJOR.$MINOR $OBJS $EXTRA $EXPLLNKCORE
      $LD $SOFLAGS$SONAME.$MAJOR.$MINOR $LDFLAGS \
         -o $LIB.$MAJOR.$MINOR $OBJS $EXTRA $EXPLLNKCORE
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
