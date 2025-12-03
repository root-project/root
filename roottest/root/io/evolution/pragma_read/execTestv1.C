#ifdef ClingWorkAroundMissingDynamicScope

int execTestv1()
{
   gSystem->Load("libRoottestIoEvolutionPragmaV1");
   gROOT->ProcessLine(".L v1/Test.C");
   gROOT->ProcessLine("Test(true,true);");
   return 0;
}

#else

#include "v1/Test.C"

int execTestv1()
{
   Test(true,true);
   return 0;
}
#endif
