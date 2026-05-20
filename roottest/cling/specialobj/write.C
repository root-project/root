{
   gROOT->ProcessLine(".L AthIndex.h+");
   gROOT->ProcessLine(".L T0Result.h+");
   gROOT->ProcessLine(".L calib.h+");
   
#ifdef ClingWorkAroundMissingDynamicScope
   TObject *calib = 0;
   calib = (TObject*)gROOT->ProcessLine("new RTCalib;");
#else
   RTCalib *calib = new RTCalib;
#endif
   TFile* f = TFile::Open("tmp.root", "recreate");
   calib->Write();
   f->Close();

}
