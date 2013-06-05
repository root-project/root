#! /bin/sh

CURVERS=`cat build/version_number | sed -e "s/^/v/" -e "s/\./-/" -e "s/\//-/"`
ROOTVERS=`cat build/version_number | sed -e 's/\//\./'`
TYPE=source
TARFILE=root_v$ROOTVERS.$TYPE.tar

# generate etc/gitinfo.txt
build/unix/gitinfo.sh

#git archive -v -o ../$TARFILE --prefix=root/ master
git archive -v -o ../$TARFILE --prefix=root/ $CURVERS

mkdir -p etc/root/etc
cp etc/gitinfo.txt etc/root/etc/
cd etc
tar -r -vf ../../$TARFILE root/etc/gitinfo.txt
cd ..
rm -rf etc/root
cd ..
gzip $TARFILE

exit 0
