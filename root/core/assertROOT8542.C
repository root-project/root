// Test ROOT-8542.
int assertROOT8542() {
   int interpError = 0;
   gErrorIgnoreLevel = kBreak;
   gROOT->ProcessLine(".x ThisFileDoesNotExist.C", &interpError);
   if (interpError == TInterpreter::kNoError) {
      std::cerr << "TApplication did not set error flag when .x-ing a missing file!\n";
      exit(1);
   }
   return 0;
}
