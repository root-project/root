{
// Fill out the code of the actual test
   gROOT->ProcessLine(".L fornamespace.C+");
gROOT->GetClass("MySpace::MyClass");
using namespace MySpace;
gROOT->GetClass("MyClass");
gROOT->GetClass("MySpace::MyClass")->Print();
gROOT->GetClass("MyClass")->Print();
gROOT->GetClass("MySpace::MyClass")->GetStreamerInfo()->ls();
gROOT->GetClass("MyClass")->GetStreamerInfo()->ls();
};

