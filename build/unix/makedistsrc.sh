#! /bin/sh

ROOTSRCDIR=$1

CURVERS=`cat $ROOTSRCDIR/build/version_number | sed -e "s/^/v/" -e "s/\./-/" -e "s/\//-/"`
ROOTVERS=`cat $ROOTSRCDIR/build/version_number | sed -e 's/\//\./'`
TYPE=source
TARFILE=root_v$ROOTVERS.$TYPE.tar

( cd $ROOTSRCDIR; git checkout $CURVERS )
# generate etc/gitinfo.txt
$ROOTSRCDIR/build/unix/gitinfo.sh $ROOTSRCDIR

( cd $ROOTSRCDIR; git archive -v -o ../$TARFILE --prefix=root-$ROOTVERS/ $CURVERS )

mkdir -p etc/root-$ROOTVERS/etc
cp etc/gitinfo.txt etc/root-$ROOTVERS/etc/
cd etc
tar -r -vf ../../$TARFILE root-$ROOTVERS/etc/gitinfo.txt
cd ..
rm -rf etc/root-$ROOTVERS
cd ..
gzip $TARFILE

exit 0
