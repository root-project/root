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

mkdir $EDIR


SRC=$ROOTSYS/test
cp $SRC/Event.cxx $SRC/Event.h $SRC/EventLinkDef.h $SRC/Makefile \
   $SRC/Makefile.arch $EDIR
mkdir $EDIR/PROOF-INF
cd $EDIR/PROOF-INF

cat > BUILD.sh <<EOF
#! /bin/sh

make libEvent.so
EOF

cat > SETUP.C <<EOF
void SETUP()
{
   gSystem->Load("libEvent");
}
EOF

chmod 755 BUILD.sh

cd ../..

tar zcvf event.par $EDIR

rm -rf event

exit 0
