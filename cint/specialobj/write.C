{
gROOT->ProcessLine(".L AthIndex.h+");
gROOT->ProcessLine(".L T0Result.h+");
gROOT->ProcessLine(".L calib.h+");
   RTCalib *calib = new RTCalib;
   TFile* f = TFile::Open("tmp.root", "recreate");
   calib->Write();
   f->Close();

}
