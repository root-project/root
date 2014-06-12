{
   gROOT->ProcessLine(".L fornamespace.C+");
gROOT->GetClass("MySpace::MyClass");
#ifdef ClingWorkAroundAutoParseUsingNamespace
   // This does not work: gInterpreter->AutoParse("MySpace");
   // so try that:
   gInterpreter->AutoParse("MySpace::MyClass");
#endif
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
gROOT->GetClass("MySpace")->GetListOfMethods()->FindObject("funcInNS")->Print();
gROOT->GetListOfGlobalFunctions()->FindObject("globalFunc")->Print();
gROOT->GetListOfGlobalFunctions()->FindObject("globalFuncInline")->Print();

 storeACl();


 // Tests for the ROOT/Meta side of ROOT-6079, ROOT-6094:
   gROOT->ProcessLine("namespace A {}");
   TClass* clA = TClass::GetClass("A");
   TList* funs = clA->GetListOfMethods(true);
   printf("Empty A: got %d funs\n", funs ? funs->GetEntries() : 0);
   gROOT->ProcessLine("namespace A {void f();}");
   funs = clA->GetListOfMethods(true);
   printf("More A: got %d funs\n", funs ? funs->GetEntries() : 0);
   gROOT->ProcessLine("namespace A {void f(int); void g(); void f_();}");
   TCollection* methods = clA->GetListOfMethods();
   TMethod* meth = dynamic_cast<TMethod*>(methods->FindObject("f"));
   if (!meth) {
      printf("Cannot find any A::f()!\n");
      exit(1);
   }
   TCollection* overloads = clA->GetListOfMethodOverloads("f");
   printf ("Full A: there are %d A::f() overloads.\n", overloads->GetSize());
   return 0;
};
