#! /bin/sh

EXPDIR=$HOME/root_export_$$
CURVERS=`cat build/version_number | sed -e "s/^/v/" -e "s/\./-/" -e "s/\//-/"`
ROOTVERS=`cat build/version_number | sed -e 's/\//\./'`
URL=`svn info | awk '/URL:/ { print $2 }' | sed 's/https/http/'`
MACHINE=`uname`
OSREL=`uname -r`
TYPE=source
TARFILE=root_v$ROOTVERS.$TYPE.tar

rm -rf $EXPDIR
mkdir $EXPDIR
cd $EXPDIR

#svn co http://root.cern.ch/svn/root/tags/$CURVERS root
svn co $URL root

# generate etc/svninfo.txt
cd root
build/unix/svninfo.sh
cd ..

# remove .svn directories containing extra copy of the code
find root -depth -name .svn -exec rm -rf {} \;

tar cvf $TARFILE root
gzip $TARFILE

mv $TARFILE.gz $HOME

cd
rm -rf $EXPDIR

exit 0
