{
   gSystem->Load("./Classes");
   // This is necessary with delayed header parsing until functions are fwd declared
   gROOT->ProcessLine("#include \"classesFuncFwdDecl.h\"");
#ifdef ClingWorkAroundMissingDynamicScope
   gSystem->Exit(!gROOT->ProcessLine("Classes();"));
#else
   gSystem->Exit(!Classes());
#endif
}
