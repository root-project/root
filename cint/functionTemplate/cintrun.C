#include <iostream>
//#include "t.h"

void cintrun() {
   gSystem->Setenv("LINES","-1");
   if (!TClass::GetClass("t"))
      //gSystem->Load("libG__t");
      gROOT->ProcessLine(".L t.h+");
   TClass *cl = TClass::GetClass("t");
   if (!cl) {
      cerr << "Cannot find TClass for t" << endl;
      return;
   }
   TList methodsSorted;
   methodsSorted.AddAll(cl->GetListOfMethods());
   methodsSorted.Sort();
   TIter iMethod(&methodsSorted);
   TMethod* m = 0;
   while ((m = (TMethod*) iMethod()))
      cout << m->GetPrototype() << endl;
   cout <<endl;
   gROOT->ProcessLine(".L printme.C");
   t o; printme(o);
   o.set(12); printme(o);
   o.set(42.33); printme(o);
}
