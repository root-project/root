#! /bin/sh
#
# Make the event.par file to be used to analyse Event objects with PROOF.
#
# Usage: sh make_event_par.sh
#
# Creates the PAR file "event.par" which can be used in PROOF via the
# package manager like:
#   gProof->UploadPackage("event.par")
#   gProof->EnablePackage("event")
#
# Command to check that package is active and that libEvent.so is loaded:
#   gProof->ShowPackages()
#   gProof->ShowEnabledPackages()
#   gProof->Exec("gSystem->ListLibraries()")
#

EDIR=event

if [ -d $EDIR ]; then
   rm -rf $EDIR
fi

mkdir $EDIR


SRC=$ROOTSYS/test
ETC=$ROOTSYS/etc
cp $SRC/Event.cxx $SRC/Event.h $SRC/EventLinkDef.h \
   $ETC/Makefile.arch $EDIR
cp Makefile_event $EDIR/Makefile
mkdir $EDIR/PROOF-INF
cd $EDIR/PROOF-INF

cat > BUILD.sh <<EOF
#! /bin/sh
# Build libEvent library.

if [ "$1" = "clean" ]; then
   make distclean
   exit 0
fi

make
EOF

cat > SETUP.C <<EOF
int SETUP()
{
   if (gSystem->Load("libEvent") == -1)
      return -1;
   return 0;
}
EOF

chmod 755 BUILD.sh

cd ../..

tar zcvf event.par $EDIR

# don't remove the directory as it is needed locally by PROOF-Lite
#rm -rf $EDIR

exit 0
