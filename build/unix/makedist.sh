#! /bin/sh

# Script to produce binary distribution of ROOT.
# Called by main Makefile.
#
# Author: Fons Rademakers, 29/2/2000

ROOTVERS=`cat build/version_number | sed -e 's/\//\./'`
MACHINE=`uname`
OSREL=`uname -r`
TYPE=$MACHINE.$OSREL
TARFILE=rootcvs_$ROOTVERS.$TYPE.tar

cp main/src/rmain.cxx include/
dir=`basename $(pwd)`
cd ..
rm -f $TARFILE.gz
tar cvf $TARFILE $dir/LICENSE $dir/README $dir/bin \
   $dir/include $dir/lib $dir/cint/MAKEINFO $dir/cint/include \
   $dir/cint/lib $dir/cint/stl $dir/tutorials/.rootrc $dir/tutorials/*.C \
   $dir/test/*.cxx $dir/test/*.h $dir/test/Makefile* \
   $dir/test/README $dir/macros $dir/icons system.rootrc
gzip $TARFILE

cd $dir
rm -f include/rmain.cxx

exit 0
