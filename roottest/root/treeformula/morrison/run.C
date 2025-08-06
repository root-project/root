{
  gSystem->Load("main_C");
#ifdef ClingWorkAroundMissingDynamicScope
   return gROOT->ProcessLine("foo::run()");
#else
   return foo::run();
#endif
}
