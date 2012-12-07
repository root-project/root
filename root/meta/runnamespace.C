{
// Fill out the code of the actual test
   gROOT->ProcessLine(".L fornamespace.C+");
gROOT->GetClass("MySpace::MyClass");
#ifdef ClingWorkAroundMissingDynamicScope
   gROOT->ProcessLine("using namespace MySpace;");
#else
using namespace MySpace;
#endif
gROOT->GetClass("MyClass");
gROOT->GetClass("MySpace::MyClass")->Print();
gROOT->GetClass("MyClass")->Print();
gROOT->GetClass("MySpace::MyClass")->GetStreamerInfo()->ls();
gROOT->GetClass("MyClass")->GetStreamerInfo()->ls();
};

