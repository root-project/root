{
gROOT->ProcessLine(".L set_problem.cxx+");
#ifdef ClingWorkAroundMissingDynamicScope
   gROOT->ProcessLine("run();");
#else
   run();
#endif
}
