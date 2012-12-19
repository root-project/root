{
   gROOT->ProcessLine(".L AthIndex.h+");
   gROOT->ProcessLine(".L T0Result.h+");
   gROOT->ProcessLine(".L calib.h+");
   TFile* f = TFile::Open("tmp.root", "");
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
