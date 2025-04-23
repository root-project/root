{
#ifdef ClingWorkAroundMissingAutoLoading
   gSystem->Load("libTreePlayer");
#endif
   // Fill out the code of the actual test
   gROOT->ProcessLine(".L userClass.C+");
#ifndef ClingWorkAroundMissingUnloading
   gROOT->ProcessLine(".x treeBranch.C");
   gROOT->ProcessLine(".U treeBranch.C");
#endif
   gROOT->ProcessLine(".x treeBranch.C+");
}
