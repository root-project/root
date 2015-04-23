#! /bin/sh -x

# first arguments is the source directory
if [ $# -ge 1 ]; then
   ROOT_SRCDIR=$1
   shift
else
   echo "$0: expecting at least ROOT_SRCDIR as first argument"
   exit 1
fi

CURVERS=`cat $ROOT_SRCDIR/build/version_number | sed -e "s/^/v/" -e "s/\./-/" -e "s/\//-/"`
ROOTVERS=`cat $ROOT_SRCDIR/build/version_number | sed -e 's/\//\./'`
TYPE=source
TARFILE=root_v$ROOTVERS.$TYPE.tar

# generate etc/gitinfo.txt
$ROOT_SRCDIR/build/unix/gitinfo.sh $ROOT_SRCDIR

git -C $ROOT_SRCDIR archive -v -o ../$TARFILE --prefix=root/ $CURVERS

mkdir -p etc/root/etc
cp etc/gitinfo.txt etc/root/etc/
cd etc
tar -r -vf ../../$TARFILE root/etc/gitinfo.txt
cd ..
rm -rf etc/root
cd ..
gzip $TARFILE

exit 0
