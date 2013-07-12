#include "TFile.h"
#include "TH1F.h"
#include "TTreeReader.h"
#include "TTreeReaderValue.h"

#ifdef __CINT__
#pragma link C++ class TTreeReaderValue<Float_t>+;
#endif

void TreeReaderSimple() {
	TH1F *myHistogram = new TH1F ("h1","ntuple",100,-4,4);

	TFile *myFile = TFile::Open("hsimple.root");
	TTreeReader myHSimpleReader ("ntuple", myFile);

	TTreeReaderValue<Float_t> myPx (myHSimpleReader, "px");
	TTreeReaderValue<Float_t> myPy (myHSimpleReader, "py");

	while (myHSimpleReader.SetNextEntry()){
		myHistogram->Fill(*myPx + *myPy);
	}

	myHistogram->Draw();
}
