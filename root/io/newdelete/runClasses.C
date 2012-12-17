{
   gSystem->Load("./Classes");
#ifdef ClingWorkAroundMissingDynamicScope
   gSystem->Exit(!gROOT->ProcessLine("Classes();"));
#else
   gSystem->Exit(!Classes());
#endif
}
