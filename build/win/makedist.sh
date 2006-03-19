#! /bin/sh

# Script to produce binary distribution of ROOT.
# Called by main Makefile.
#
# Author: Fons Rademakers, 29/2/2000

ROOTVERS=`cat build/version_number | sed -e 's/\//\./'`
TYPE=`bin/root-config --arch`
TARFILE=root_v$ROOTVERS.$TYPE.tar

TAR=/bin/tar
dum=`echo $TAR | grep "no gtar"`
stat=$?
if [ "$TAR" = '' ] || [ $stat = 0 ]; then
   TAR="tar cvf"
   rm -f ../$TARFILE.gz
   EXCLUDE=
else
   TAR=$TAR" zcvf"
   rm -f ../$TARFILE.gz
   TARFILE=$TARFILE".gz"
   EXCLUDE="--exclude CVS --exclude .cvsignore"
fi

for i in include/precompile.*; do
   if [ "x$i" != "xinclude/precompile.h" ]; then
      EXCLUDE="--exclude $i "$EXCLUDE
   fi
done

cp -f main/src/rmain.cxx include/
pwd=`pwd`
dir=`basename $pwd`
cd ..
$TAR $TARFILE $EXCLUDE $dir/LICENSE $dir/README $dir/bin \
   $dir/include $dir/lib $dir/cint/MAKEINFO $dir/cint/include \
   $dir/cint/lib $dir/cint/stl $dir/tutorials/*.cxx $dir/tutorials/*.C \
   $dir/tutorials/*.h $dir/tutorials/*.dat $dir/tutorials/mlpHiggs.root \
   $dir/tutorials/gallery.root $dir/tutorials/galaxy.root \
   $dir/tutorials/stock.root $dir/tutorials/worldmap.jpg \
   $dir/tutorials/mditestbg.xpm $dir/tutorials/fore.xpm \
   $dir/tutorials/runcatalog.sql $dir/tutorials/*.py $dir/tutorials/*.rb \
   $dir/tutorials/saxexample.xml $dir/tutorials/person.xml \
   $dir/tutorials/person.dtd \
   $dir/test/*.cxx $dir/test/*.h $dir/test/Makefile* $dir/test/README \
   $dir/test/RootShower/*.h $dir/test/RootShower/*.cxx \
   $dir/test/RootShower/*.rc $dir/test/RootShower/*.ico \
   $dir/test/RootShower/*.png $dir/test/RootShower/Makefile \
   $dir/test/RootShower/anim $dir/test/RootShower/icons $dir/test/ProofBench \
   $dir/macros $dir/icons $dir/fonts $dir/etc $dir/proof/etc $dir/proof/utils
if [ "$TAR" = '' ] || [ $stat = 0 ]; then
   gzip $TARFILE
fi

cd $dir
rm -f include/rmain.cxx

exit 0
