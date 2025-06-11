#ifdef ClingWorkAroundMissingDynamicScope

int execTestv2()
{
   gSystem->Load("libRoottestIoEvolutionPragmaV2");
   gROOT->ProcessLine(".L v2/Test.C");
   gROOT->ProcessLine("Test(true,true);");
   return 0;
}

#else

#include "v2/Test.C"

int execTestv2()
{
   Test(true,true);
   return 0;
}

#endif
