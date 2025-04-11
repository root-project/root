#include "ClassDefinitions.C"

#ifdef __MAKECINT__
#pragma link C++ class QRawPulseR+;
#pragma link C++ class QRawTriggerPulseR+;
#pragma link C++ class QRawEventR+;
#endif 

#include "TH1.h"
#include "TTree.h"
#include "TFile.h"
#include "TTreeFormula.h"
#include "Riostream.h"
#include "TMath.h"
#include "TRandom.h"
#include "TStopwatch.h"

void createtree() {		
	TH1::AddDirectory(kFALSE);

	QRawTriggerPulseR *pulse = new QRawTriggerPulseR();
	QRawEventR *rawevent = new QRawEventR();

	TFile *f = new TFile("speedtest.root", "recreate");
	TTree *tree = new TTree("tree", "tree", 99);
	
	tree->Branch("rawevent.", "QRawEventR", &rawevent, 32000, 99);
	tree->Branch("rawtriggerpulse.", "QRawTriggerPulseR", &pulse, 32000, 99);
	
	for (int entry = 0; entry != 100; ++entry) {
		pulse->SetChannel((Int_t) (gRandom->Rndm() * 100));
		double *samples = new double[512];
      for (int i = 0; i != 512; ++i) samples[i] = gRandom->Rndm() * pow(2.0,16);
		pulse->SetDataHist(512, samples);
		rawevent->SetRawPulse(*pulse);
		tree->Fill();
		delete [] samples;
		
		if ((entry + 1) % 10000 == 0) cout << entry + 1 << " entries done" << endl;
	}

	tree->Write();
	f->Close();
	
	delete f;
}

void CheckFormula(const char *what, TTree *tree) {
   TTreeFormula *tf = new TTreeFormula("test",what,tree);
   cout << "For " << what << " found " << tf->GetLeaf(0)->GetName() << endl;
   delete tf;
}

void runtest(bool runhist = false) {
   TH1::AddDirectory(kFALSE);

   TFile *g = new TFile("speedtest.root");
   TTree *tree; g->GetObject("tree",tree);

   if (runhist) {
      TStopwatch stopwatch;

      tree->Draw("rawtriggerpulse.QRawPulseR.fChannel");
      stopwatch.Stop();
      stopwatch.Print("m");

      stopwatch.Start(kTRUE);
      tree->Draw("rawevent.fraw_pulse.fChannel");
      stopwatch.Stop();
      stopwatch.Print("m");

      stopwatch.Start(kTRUE);
      tree->Draw("rawevent.fraw_trigger_pulse.fChannel");
      stopwatch.Stop();
      stopwatch.Print("m");
   } else {
      CheckFormula("rawtriggerpulse.QRawPulseR.fChannel",tree);
      CheckFormula("rawevent.fraw_pulse.fChannel",tree);
      CheckFormula("rawevent.fraw_trigger_pulse.fChannel",tree);
      CheckFormula("rawevent.fraw_trigger_pulse.QRawPulseR.fChannel",tree);
   }
}

void runbase(bool create = false) {

   if (create) {
      createtree();
   } else {
      runtest();
   }
}
