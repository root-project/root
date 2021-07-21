void dt_wrap(const char* from, Int_t mode = 0, Int_t verboseLevel = 0) {
   gROOT->ProcessLine(".L dt_RunDrawTest.C+");
   int status = !dt_RunDrawTest(from,mode,verboseLevel);
   if (verboseLevel==0) gSystem->Exit(status);
}
