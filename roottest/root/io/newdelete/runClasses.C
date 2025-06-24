{
   gSystem->Load("libRoottestIoNewdelete");
   // This is necessary with delayed header parsing until functions are fwd declared
#ifdef ClingWorkAroundMissingDynamicScope
#ifdef ClingWorkAroundFunctionForwardDeclarations
   gROOT->ProcessLine("#include \"classesFuncFwdDecl.h\"");
#endif
   gSystem->Exit(!gROOT->ProcessLine("Classes();"));
#else
#ifdef ClingWorkAroundFunctionForwardDeclarations
   #include "classesFuncFwdDecl.h";
#endif
   gSystem->Exit(!Classes());
#endif
}
