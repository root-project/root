#! /bin/sh

# Script to produce binary distribution of ROOT.
# Called by main Makefile.
#
# Author: Fons Rademakers, 29/2/2000

ROOTVERS=`cat build/version_number | sed -e 's/\//\./'`
MACHINE=`uname`
OSREL=`uname -r`
if [ "x$MACHINE" = "xCYGWIN_NT-5.1" ]; then
   TYPE=$MACHINE
else
   TYPE=$MACHINE.$OSREL
fi
TARFILE=root_v$ROOTVERS.$TYPE.tar

rm -f ../${TARFILE}.gz

if [ "x`which gtar 2>/dev/null | awk '{if ($1~/gtar/) print $1;}'`" != "x" ]
then
   TAR="gtar zcvf"
   TARFILE=$TARFILE".gz"
   EXCLUDE="--exclude CVS --exclude .cvsignore"
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
   $dir/tutorials/galaxy.pal.root $dir/tutorials/galaxy.root \
   $dir/tutorials/stock.root $dir/tutorials/worldmap.jpg \
   $dir/tutorials/mditestbg.xpm $dir/tutorials/fore.xpm \
   $dir/tutorials/runcatalog.sql $dir/tutorials/*.py $dir/tutorials/*.rb \
   $dir/tutorials/saxexample.xml \
   $dir/test/*.cxx $dir/test/*.h $dir/test/Makefile* $dir/test/README \
   $dir/test/RootShower/*.h $dir/test/RootShower/*.cxx \
   $dir/test/RootShower/*.rc $dir/test/RootShower/*.ico \
   $dir/test/RootShower/*.png $dir/test/RootShower/Makefile \
   $dir/test/RootShower/anim $dir/test/RootShower/icons $dir/test/ProofBench \
   $dir/macros $dir/icons $dir/fonts $dir/etc $dir/proof/etc $dir/proof/utils
if [ "x$DOGZIP" = "xy" ]; then
   gzip $TARFILE
fi

cd $dir
rm -f include/rmain.cxx

exit 0
