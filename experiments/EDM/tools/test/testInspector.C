#include "FOO.h"

void testInspector() {
   TString inclPath(gSystem->GetIncludePath());
   inclPath += " -I..";
   gSystem->SetIncludePath(inclPath);

   if (!gSystem->CompileMacro("FOO_rflx.cpp") ||
       !gSystem->CompileMacro("../Inspector.cxx", "..")) {
      cerr << "Cannot build libraries!" << endl;
      return;
   }
   double d=17.;
   double e=18.;
   Inspect::InspectorBase insp;

   ROOT::Reflex::Type t = ROOT::Reflex::Type::ByName("double");

   if (!insp.IsEqual(t, ROOT::Reflex::Dummy::Member(), &d, &e))
      cout << "1 OK" << endl;

   insp.Dump(cout, t, &d);
   insp.Dump(cout, t, &e);

   e=17.;
   if (insp.IsEqual(t, ROOT::Reflex::Dummy::Member(), &d, &e))
      cout << "2 OK" <<  endl;


   t=ROOT::Reflex::PointerBuilder(t);
   double* pd=&d;
   double* pe=&e;
   if (insp.IsEqual(t, ROOT::Reflex::Dummy::Member(), &pd, &pe)) cout << "3 OK" << endl;
   e=18;
   if (!insp.IsEqual(t, ROOT::Reflex::Dummy::Member(), &pd, &pe)) cout << "4 OK" << endl;


   insp.Dump(cout, t, &pd);
   insp.Dump(cout, t, &pe);

   BAR bar;
   bar.i = 1;
   bar.f = 2.;
   FOO foo;
   foo.a = 3;
   foo.b = 4.;
   foo.bar = &bar;

   ROOT::Reflex::Type typeFOO = ROOT::Reflex::Type::ByName("FOO");
   insp.Dump(cout, typeFOO, &foo);

   BAR bar2;
   bar2.i = 1;
   bar2.f = 3.;
   FOO foo2;
   foo2.a = 3;
   foo2.b = 4.;
   foo2.bar = &bar2;
   if (!insp.IsEqual(typeFOO, ROOT::Reflex::Dummy::Member(), &foo, &foo2)) cout << "5 OK" << endl;


   ROOT::Reflex::Scope scopeFOO = ROOT::Reflex::Scope::ByName("FOO");
   Inspect::InspectorGenSource gs(scopeFOO, "FOO.h");
   gs.WriteSource();
   std::string macroname(gs.GetName());
   macroname += ".hh";
   gSystem->CompileMacro(macroname.c_str());
   Inspect::InspectorBase* inspFOO = gs.GetInspector(scopeFOO, &foo);
   inspFOO->Inspect();
   inspFOO->SetExpectedResults(*inspFOO->GetActualResults());
   if (inspFOO->AsExpected()) cout << "6 OK" << endl;

   inspFOO->Inspect();
   if (!inspFOO->AsExpected()) cout << "7 OK" << endl;
   if (!gROOT->ProcessLine(Form(".x testInspectorInterpreted.C(0x%x,0x%x)", 
   		       &foo2, inspFOO->GetExpectedResults())))
     cout << "8 OK" << endl;
   if (gROOT->ProcessLine(Form(".x testInspectorInterpreted.C(0x%x,0x%x)", 
   		       &foo, inspFOO->GetExpectedResults())))
     cout << "9 OK" << endl;
}
