#ifdef ClingWorkAroundUnnamedInclude
#ifndef ClingWorkAroundMissingSmartInclude
#include "MyClassReferenceUse.C+"
#endif
void runreferenceUse() {
#else
{
#endif
#ifdef ClingWorkAroundMissingSmartInclude
   gROOT->ProcessLine(".L MyClassReferenceUse.C+");
#else
#ifndef ClingWorkAroundUnnamedInclude
   #include "MyClassReferenceUse.C+"
#endif
#endif
#ifdef ClingWorkAroundMissingDynamicScope
   gROOT->ProcessLine("MyClass& m = GetMyClassReference();");
   gROOT->ProcessLine("m.GetScalar<Int_t>(\"Test\");");
#else
   MyClass& m = GetMyClassReference();
   m.GetScalar<Int_t>("Test");
#endif
}
