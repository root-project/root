{
   gROOT->ProcessLine(".L AthIndex.h+");
   gROOT->ProcessLine(".L T0Result.h+");
   gROOT->ProcessLine(".L calib.h+");
#ifdef ClingWorkAroundUnnamedIncorrectInitOrder   
   TFile* f = 0;
   f = TFile::Open("tmp.root", "");
#else
   TFile* f = TFile::Open("tmp.root", "");
#endif
#ifdef ClingWorkAroundMissingDynamicScope
   f->Get("calib");
#else
   calib;
#endif
#ifdef ClingWorkAroundBrokenUnnamedReturn
   bool ret = 0;
#else
   return 0;
#endif
}
