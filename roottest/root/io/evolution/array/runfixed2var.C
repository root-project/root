{
// Fill out the code of the actual test
#ifdef ClingWorkAroundMissingDynamicScope
   gROOT->LoadMacro("MyClassOld.cxx+");      
#endif
   gROOT->ProcessLine(".x ReadOldObject.C");
}
