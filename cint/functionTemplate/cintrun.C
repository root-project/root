#include <iostream>
//#include "t.h"

void cintrun() {
   gSystem->Setenv("LINES","-1");
   if (!TClass::GetClass("t"))
      //gSystem->Load("libG__t");
      gROOT->ProcessLine(".L t.h+");
   gROOT->ProcessLine(".class t");
   gROOT->ProcessLine(".L printme.C");
   t o; printme(o);
   o.set(12); printme(o);
   o.set(42.33); printme(o);
}
