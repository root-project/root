#! /bin/sh

FILEVERS=$1
ROOTSRCDIR=$2

# FILEVERS is M.mm.pp, GITTAG must be M-mm-pp
GITTAG=`echo $FILEVERS | tr '.' '-'`
TARFILE=root_v$FILEVERS.source.tar

rm -f ../$TARFILE ../${TARFILE}.gz
cd $ROOTSRCDIR \
&& git archive -v -o ../$TARFILE --prefix=root-$FILEVERS/ v$GITTAG \
&& gzip ../$TARFILE
