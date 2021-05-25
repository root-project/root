#! /bin/sh

FILEVERS=$1
GITTAG=$2
ROOTSRCDIR=$3

TARFILE=root_v$FILEVERS.source.tar

( cd $ROOTSRCDIR; git archive -v -o ../$TARFILE --prefix=root-$FILEVERS/ $GITTAG )

mkdir -p etc/root-$FILEVERS/etc
cp etc/gitinfo.txt etc/root-$FILEVERS/etc/
cd etc
tar -r -vf ../../$TARFILE root-$FILEVERS/etc/gitinfo.txt
cd ..
rm -rf etc/root-$FILEVERS
cd ..
gzip $TARFILE

exit 0
