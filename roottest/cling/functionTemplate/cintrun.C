#include <iostream>
#ifdef ClingWorkAroundMissingDynamicScope
#include "t.h"
#endif

TObjString* methodToObjString(TMethod* m) {
   TObjString* pout = new TObjString;
   TString& out = pout->String();
   out = m->GetName();
   out += "(";
   TIter iArg(m->GetListOfMethodArgs());
   TMethodArg* arg = 0;
   bool first = true;
   while ((arg = (TMethodArg*) iArg())) {
      if (first) first = false;
      else out += ", ";
      out += arg->GetFullTypeName();
   }
   out += ")";
   if (m->Property() & kIsConstMethod)
      out += " const";
   return pout;
}

#ifdef ClingWorkAroundMissingDynamicScope
#include "printme.C"
#endif

void cintrun() {
   gSystem->Setenv("LINES","-1");
   if (!TClass::GetClass("t"))
      gSystem->Load("t_h");
   TClass *cl = TClass::GetClass("t");
   if (!cl) {
      cerr << "Cannot find TClass for t" << endl;
      return;
   }
   TList methodsSorted;
   methodsSorted.SetOwner();
   TIter iMethod(cl->GetListOfMethods());
   TMethod* m = 0;
   while ((m = (TMethod*) iMethod()))
      methodsSorted.Add(methodToObjString(m));

   methodsSorted.Sort();
   TIter iMethName(&methodsSorted);
   TObjString* name = 0;
   while ((name = (TObjString*) iMethName()))
      cout << name->String() << endl;
   cout <<endl;
#ifndef ClingWorkAroundMissingDynamicScope
   gROOT->ProcessLine(".L printme.C");
#endif
   t o; printme(o);
   o.set(12); printme(o);
   o.set(42.33); printme(o);
}
