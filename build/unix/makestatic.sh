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
EXTRALIBS=$8

ROOTALIB=lib/libRoot.a
ROOTAEXE=bin/roota
PROOFAEXE=bin/proofserva

rm -f $ROOTALIB $ROOTAEXE $PROOFAEXE

excl="main proof/proofd net/rootd net/xrootd rootx montecarlo/pythia6 \
      montecarlo/pythia8 sql/mysql sql/pgsql io/rfio sql/sapdb \
      hist/hbook core/newdelete misc/table core/utils net/srputils \
      net/krb5auth net/globusauth io/chirp io/dcache net/alien \
      graf2d/asimage net/ldap graf2d/qt gui/qtroot math/quadp \
      bindings/pyroot bindings/ruby tmva \
      io/xmlparser graf3d/gl graf3d/ftgl roofit/roofit roofit/roofitcore \
      roofit/roostats sql/oracle net/netx net/auth net/rpdutils math/mathmore \
      math/minuit2 io/gfal net/monalisa proof/proofx math/fftw gui/qtgsi \
      sql/odbc io/castor math/unuran geom/gdml cint/cint7 montecarlo/g4root \
      graf2d/gviz graf3d/gviz3d graf3d/eve net/glite misc/minicern \
      misc/memstat net/bonjour"

if test -f core/meta/src/TCint_7.o ; then
   mv core/meta/src/TCint_7.o core/meta/src/TCint_7.o-
   mv core/meta/src/G__TCint_7.o core/meta/src/G__TCint_7.o-
   mv core/utils/src/RStl7.o core/utils/src/RStl7.o-
   mv core/metautils/src/RConversionRuleParser7.o core/metautils/src/RConversionRuleParser7.o-
   mv core/metautils/src/TClassEdit7.o core/metautils/src/TClassEdit7.o-
fi

objs=""
gobjs=""
for i in * ; do
   inc=$i
   for j in $excl ; do
      if [ $j = $i ]; then
         continue 2
      fi
   done
   ls $inc/src/*.o > /dev/null 2>&1 && objs="$objs `ls $inc/src/*.o`"
   ls $inc/src/G__*.o > /dev/null 2>&1 && gobjs="$gobjs `ls $inc/src/G__*.o`"
   if [ -d $i ]; then
      for k in $i/* ; do
         inc=$k
         for j in $excl ; do
            if [ $j = $k ]; then
               continue 2
            fi
         done
         ls $inc/src/*.o > /dev/null 2>&1 && objs="$objs `ls $inc/src/*.o`"
         ls $inc/src/G__*.o > /dev/null 2>&1 && gobjs="$gobjs `ls $inc/src/G__*.o`"
      done
   fi
done

echo "Making $ROOTALIB..."
echo ar rv $ROOTALIB cint/cint/main/G__setup.o cint/cint/src/dict/*.o $objs
ar rv $ROOTALIB cint/cint/main/G__setup.o cint/cint/src/dict/*.o $objs > /dev/null 2>&1

if test -f core/meta/src/TCint_7.o- ; then
   mv core/meta/src/TCint_7.o- core/meta/src/TCint_7.o
   mv core/meta/src/G__TCint_7.o- core/meta/src/G__TCint_7.o
   mv core/utils/src/RStl7.o- core/utils/src/RStl7.o
   mv core/metautils/src/RConversionRuleParser7.o- core/metautils/src/RConversionRuleParser7.o
   mv core/metautils/src/TClassEdit7.o- core/metautils/src/TClassEdit7.o
fi

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
echo $LD $LDFLAGS -o $ROOTAEXE main/src/rmain.o $dummyo $gobjs $ROOTALIB \
   $XLIBS $SYSLIBS $EXTRALIBS
$LD $LDFLAGS -o $ROOTAEXE main/src/rmain.o $dummyo $gobjs $ROOTALIB \
   $XLIBS $SYSLIBS $EXTRALIBS

linkstat=$?
if [ $linkstat -ne 0 ]; then
   exit $linkstat
fi

echo "Making $PROOFAEXE..."
echo $LD $LDFLAGS -o $PROOFAEXE main/src/pmain.o  $dummyo $gobjs $ROOTALIB \
   $XLIBS $SYSLIBS $EXTRALIBS
$LD $LDFLAGS -o $PROOFAEXE main/src/pmain.o  $dummyo $gobjs $ROOTALIB \
   $XLIBS $SYSLIBS $EXTRALIBS

linkstat=$?
if [ $linkstat -ne 0 ]; then
   exit $linkstat
fi

rm -f $dummyc $dummyo

exit 0
