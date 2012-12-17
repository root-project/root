{
// Fill out the code of the actual test
gROOT->ProcessLine(".L cstring.C+");
#ifdef ClingWorkAroundMissingDynamicScope
   gROOT->ProcessLine("cstring();");
#else
cstring();
#endif
}

