#! /bin/sh

CURVERS=`cat build/version_number | sed -e "s/^/v/" -e "s/\./-/" -e "s/\//-/"`
ROOTVERS=`cat build/version_number | sed -e 's/\//\./'`
TYPE=source
TARFILE=root_v$ROOTVERS.$TYPE.tar

git co $CURVERS
# generate etc/gitinfo.txt
build/unix/gitinfo.sh

git archive -v -o ../$TARFILE --prefix=root-$ROOTVERS/ $CURVERS

mkdir -p etc/root-$ROOTVERS/etc
cp etc/gitinfo.txt etc/root-$ROOTVERS/etc/
cd etc
tar -r -vf ../../$TARFILE root-$ROOTVERS/etc/gitinfo.txt
cd ..
rm -rf etc/root-$ROOTVERS
cd ..
gzip $TARFILE

exit 0
