#! /bin/sh

CURVERS=`cat build/version_number | sed -e "s/^/v/" -e "s/\./-/" -e "s/\//-/"`
ROOTVERS=`cat build/version_number | sed -e 's/\//\./'`
TYPE=source
TARFILE=root_v$ROOTVERS.$TYPE.tar

# generate etc/gitinfo.txt
build/unix/gitinfo.sh

prefix=`basename $PWD`

git archive -v -o ../$TARFILE --prefix=$prefix/ master
#git archive -v -o ../$TARFILE --prefix=$prefix/ $CURVERS

cd ..
tar -r -f $TARFILE $prefix/etc/gitinfo.txt
gzip $TARFILE

exit 0
