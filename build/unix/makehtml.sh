#! /bin/sh

ROOT=bin/root.exe
ROOTCONFIG=bin/root-config

VERS=`$ROOTCONFIG --prefix=. --version`

echo ""
echo "Generating doc in directory htmldoc/..."
echo ""

$ROOT -b <<makedoc
    gSystem.Load("$HOME/venus/libVenus");
    gSystem.Load("$HOME/pythia/libPythia");
    gSystem.Load("libProof");
    gSystem.Load("libHistPainter");
    gSystem.Load("libTreePlayer");
    gSystem.Load("libTreeViewer");
    gSystem.Load("libPhysics");
    gSystem.Load("libThread");
    gSystem.Load("libRFIO");
    gSystem.Load("libRGL");
    gSystem.Load("libX3d");
    gSystem.Load("libEG");
    gSystem.Load("libEGPythia");
    gSystem.Load("libEGPythia6");
    gSystem.Load("libEGVenus");
//    gSystem.Load("libTable");
//    gSystem.Load("test/libEvent.so");
    THtml html;
    html.MakeAll();
    .q
makedoc

exit 0
