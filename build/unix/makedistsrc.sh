#! /bin/sh

EXPDIR=$HOME/root_export
CURVERS=`cat build/version_number | sed -e "s/^/v/" -e "s/\./-/" -e "s/\//-/"`
ROOTVERS=`cat build/version_number | sed -e 's/\//\./'`
MACHINE=`uname`
OSREL=`uname -r`
TYPE=source
TARFILE=root_v$ROOTVERS.$TYPE.tar

rm -rf $EXPDIR
mkdir $EXPDIR
cd $EXPDIR

cvs -d :pserver:cvs@root.cern.ch:/user/cvs login <<!
cvs
!

#cvs -z 3 -d :pserver:cvs@root.cern.ch:/user/cvs export -r $CURVERS root
#cvs -z 3 -d :pserver:cvs@root.cern.ch:/user/cvs export -D today root
cvs -z 3 -d :pserver:cvs@root.cern.ch:/user/cvs co root

cvs -d :pserver:cvs@root.cern.ch:/user/cvs logout

tar cvf $TARFILE root
gzip $TARFILE

mv $TARFILE.gz $HOME

cd
rm -rf $EXPDIR

exit 0
