#! /bin/sh

# Script to produce source distribution of ROOT.
# Called by main Makefile.
#
# Author: Fons Rademakers, 29/2/2000

ROOTVERS=`cat build/version_number | sed -e 's/\//\./'`
MACHINE=`uname`
OSREL=`uname -r`
TYPE=source
TARFILE=rootcvs_$ROOTVERS.$TYPE.tar

rm -f config/Makefile.config include/config.h bin/root-config test/Makefile \
      ttf opengl
dir=`basename $(pwd)`
cd ..
rm -f $TARFILE.gz
tar cvf $TARFILE $dir
gzip $TARFILE

exit 0
