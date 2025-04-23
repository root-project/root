void runstaticFunc() {

   gROOT->ProcessLine(".L MyClass.cxx+");
#ifdef ClingWorkAroundMissingDynamicScope
   gROOT->ProcessLine(
      "MyClass * my = new MyClass();"
      "my->Init();"
      "my->Integral(0,5);");
#else
   MyClass * my = new MyClass();
   my->Init();
   my->Integral(0,5);
#endif
}
