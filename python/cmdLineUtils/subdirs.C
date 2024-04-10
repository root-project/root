#include "TFile.h"
#include "TCanvas.h"

void subdirs() {
    TFile* aFile = new TFile("subdirs.root", "RECREATE");
    aFile->mkdir("sub");
    aFile->mkdir("sub2");
    aFile->mkdir("sub3");
    aFile->mkdir("sub/s1");
    aFile->mkdir("sub/s2");
    aFile->mkdir("sub/s3");
    aFile->mkdir("sub/s3/t1");
    aFile->cd("sub/s3/t1");
    TCanvas* c = new TCanvas("c", "c");
    c->Write()
    aFile->Close();
    delete aFile;
}
