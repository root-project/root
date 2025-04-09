{
  gSystem->Load("main_C");
#ifdef ClingWorkAroundMissingDynamicScope
#ifdef ClingWorkAroundFunctionForwardDeclarations
   gROOT->ProcessLine("#include \"functionsFwdDeclarations.h\"");
#endif
   return gROOT->ProcessLine("foo::run()");
#else
#ifdef ClingWorkAroundFunctionForwardDeclarations
#include "functionsFwdDeclarations.h";
#endif
   return foo::run();
#endif
}
