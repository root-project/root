#!/bin/sh

# This scrypt generates the PAR files required by ProofBench. It takes three arguments, all mandatory:
#
#   ./makepbenchpars.sh {ProofBenchDataSel,ProofBenchCPUSel} <SourceDir> <BuildDir>
#
# The scrypt is invoked by the CMake-based build system (see proof/proofbench/CMakeLists.txt)


PARNAME=$1
if test "x$PARNAME" = "x" ; then
   echo "Entering a PAR name is mandatory"
   exit 1
fi
if test ! "x$PARNAME" = "xProofBenchDataSel" && test ! "x$PARNAME" = "xProofBenchCPUSel" ; then
   echo "PAR name must be either xProofBenchDataSel or xProofBenchCPUSel"
   exit 1
fi
ROOT_SRCDIR=$2
if test "x$ROOT_SRCDIR" = "x" ; then
   echo "Entering the ROOT source dir is mandatory"
   exit 1
fi
ROOT_BUILDDIR=$3
if test "x$ROOT_BUILDDIR" = "x" ; then
   echo "Entering the ROOT build dir is mandatory"
   exit 1
fi


MODDIR=$ROOT_SRCDIR/proof/proofbench
MODDIRS=$MODDIR/src
MODDIRI=$MODDIR/inc
PBPARDIR=$ROOT_BUILDDIR/etc/proof/proofbench

##### ProofBenchDataSel PAR file #####
if test "x$PARNAME" = "xProofBenchDataSel"; then
   PARDIR=$MODDIRS/ProofBenchDataSel
   PARH="$ROOT_SRCDIR/test/Event.h $MODDIRI/TProofBenchTypes.h \
        $MODDIRI/TSelEventGen.h $MODDIRI/TSelEvent.h $MODDIRI/TSelHandleDataSet.h"
   PARS="$ROOT_SRCDIR/test/Event.cxx \
        $MODDIRS/TSelEventGen.cxx $MODDIRS/TSelEvent.cxx $MODDIRS/TSelHandleDataSet.cxx"
   PARF=$PBPARDIR/ProofBenchDataSel.par

##### ProofBenchCPUSel PAR file #####
elif test "x$PARNAME" = "xProofBenchCPUSel"; then

   PARDIR=$MODDIRS/ProofBenchCPUSel
   PARH="$MODDIRI/TProofBenchTypes.h $MODDIRI/TSelHist.h"
   PARS=$MODDIRS/TSelHist.cxx
   PARF=$PBPARDIR/ProofBenchCPUSel.par
fi
PARINF=$PARDIR/PROOF-INF

# The PAR top directory ...
if test -d $PARDIR; then
   rm -fr $PARDIR
fi
# ... its PROOF-INF
mkdir -p $PARINF
for f in $PARH $PARS; do
   cp -rp $f $PARDIR
done
# Its SETUP.C ...
echo "#include \"TClass.h\"" > $PARINF/SETUP.C
echo "#include \"TROOT.h\"" >> $PARINF/SETUP.C
echo "Int_t SETUP() {" >> $PARINF/SETUP.C
echo "   if (!TClass::GetClass(\"TPBReadType\")) {" >> $PARINF/SETUP.C
echo "      gROOT->ProcessLine(\".L TProofBenchTypes.h+\");" >> $PARINF/SETUP.C
echo "   }" >> $PARINF/SETUP.C
for f in $PARS; do 
   b=`basename $f`;
   echo "   gROOT->ProcessLine(\".L $b+\");" >> $PARINF/SETUP.C
done
echo "   return 0;" >> $PARINF/SETUP.C
echo "}" >> $PARINF/SETUP.C

builddir=`pwd`
cd $MODDIRS
par=`basename $PARF`
pard=`basename $PARDIR`
tar cf - $pard | gzip > $par || exit 1
mv $par $PBPARDIR || exit 1;
cd $builddir;
rm -fr $PARDIR
exit 0

