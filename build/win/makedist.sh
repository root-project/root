#! /bin/sh

# Script to produce binary distribution of ROOT.
# Called by main Makefile.
#
# Author: Fons Rademakers, 29/2/2000

ROOTVERS=`cat build/version_number | sed -e 's/\//\./'`
MACHINE=`uname`
OSREL=`uname -r`
#TYPE=$MACHINE.$OSREL
TYPE=win32
TARFILE=root_v$ROOTVERS.$TYPE.tar

TAR=/bin/tar
dum=`echo $TAR | grep "no gtar"`
stat=$?
if [ "$TAR" = '' ] || [ $stat = 0 ]; then
   TAR="tar cvf"
   rm -f ../$TARFILE.gz
else
   TAR=$TAR" zcvf"
   rm -f ../$TARFILE.gz
   TARFILE=$TARFILE".gz"
fi

cp -f main/src/rmain.cxx include/
pwd=`pwd`
dir=`basename $pwd`
cd ..
$TAR $TARFILE $dir/LICENSE $dir/README $dir/bin \
   $dir/include $dir/lib $dir/cint/MAKEINFO $dir/cint/include \
   $dir/cint/lib $dir/cint/stl $dir/tutorials/*.C \
   $dir/test/*.cxx $dir/test/*.h $dir/test/Makefile* \
   $dir/test/README $dir/macros $dir/icons $dir/system.rootrc
if [ "$TAR" = '' ] || [ $stat = 0 ]; then
   gzip $TARFILE
fi

cd $dir
rm -f include/rmain.cxx

exit 0
