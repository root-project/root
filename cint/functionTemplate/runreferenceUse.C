{
#ifdef ClingWorkAroundMissingSmartInclude
   gROOT->ProcessLine(".L MyClassReferenceUse.C+");
#else
  #include "MyClassReferenceUse.C+"
#endif
#ifdef ClingWorkAroundMissingDynamicScope
   gROOT->ProcessLine("MyClass& m = GetMyClassReference();");
   gROOT->ProcessLine("m.GetScalar<Int_t>(\"Test\");");
#else
   MyClass& m = GetMyClassReference();
   m.GetScalar<Int_t>("Test");
#endif
}
