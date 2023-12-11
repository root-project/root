#! /bin/sh

FILEVERS=$1
GITTAG=$2
ROOTSRCDIR=$3

TARFILE=root_v$FILEVERS.source.tar

cd $ROOTSRCDIR \
&& git archive -v -o ../$TARFILE --prefix=root-$FILEVERS/ $GITTAG \
&& gzip $TARFILE
