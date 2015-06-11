{
   gErrorIgnoreLevel = kWarning;
#ifdef ClingWorkAroundNoDotInclude
   gInterpreter->AddIncludePath(".");
#else
gROOT->ProcessLine(".include .");
#endif
gROOT->ProcessLine(".L script.C+");
}

