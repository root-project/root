#! /bin/sh

# Script to produce binary distribution of ROOT.
# Called by main Makefile.
#
# Author: Fons Rademakers, 29/2/2000

ROOTVERS=`cat build/version_number | sed -e 's/\//\./'`
MACHINE=`uname`
OSREL=`uname -r`
TYPE=$MACHINE.$OSREL
TARFILE=root_v$ROOTVERS.$TYPE.tar

rm -f ../${TARFILE}.gz

if [ "x`which gtar 2>/dev/null | awk '{if ($1~/gtar/) print $1;}'`" != "x" ]
then
   TAR="gtar zcvf"
   TARFILE=$TARFILE".gz"
   EXCLUDE="--exclude CVS"
else
   TAR="tar cvf"
   EXCLUDE=
   DOGZIP="y"
fi

cp -f main/src/rmain.cxx include/
pwd=`pwd`
dir=`basename $pwd`
cd ..
$TAR $TARFILE $EXCLUDE $dir/LICENSE $dir/README $dir/bin \
   $dir/include $dir/lib $dir/cint/MAKEINFO $dir/cint/include \
   $dir/cint/lib $dir/cint/stl $dir/tutorials/*.cxx $dir/tutorials/*.C \
   $dir/tutorials/*.h $dir/tutorials/*.dat $dir/tutorials/mlpHiggs.root \
   $dir/tutorials/runcatalog.sql $dir/test/*.cxx $dir/test/*.h \
   $dir/test/Makefile* $dir/test/README $dir/test/RootShower/*.h \
   $dir/test/RootShower/*.cxx $dir/test/RootShower/Makefile* \
   $dir/test/RootShower/anim $dir/test/RootShower/icons \
   $dir/macros $dir/icons $dir/fonts $dir/etc $dir/proof/etc $dir/proof/utils
if [ "x$DOGZIP" = "xy" ]; then
   gzip $TARFILE
fi

cd $dir
rm -f include/rmain.cxx

exit 0
