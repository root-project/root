#include "TFile.h"
#include "TH1F.h"
#include "TTreeReader.h"
#include "TTreeReaderValue.h"
#include "TTreeReaderArray.h"

#ifdef __CINT__
#pragma link C++ class TTreeReaderValue<Float_t>+;
#endif

void TreeReaderSimple() {
	TH1F *myHistogram = new TH1F ("h1","ntuple",100,-4,4);

	TFile::Open("hsimple.root");
	TTreeReader myHSimpleReader ("ntuple");

	TTreeReaderValue<Float_t> myPx (myHSimpleReader, "px");
	TTreeReaderValue<Float_t> myPy (myHSimpleReader, "py");

	for (int i = 0; myHSimpleReader.SetNextEntry(); ++i){
		myHistogram->Fill(*myPx + *myPy);
	}

	myHistogram->Draw();
}
