{
   gROOT->ProcessLine(".L AthIndex.h+");
   gROOT->ProcessLine(".L T0Result.h+");
   gROOT->ProcessLine(".L calib.h+");
   TFile* f = TFile::Open("tmp.root", "");
   calib;
   return 0;
}
