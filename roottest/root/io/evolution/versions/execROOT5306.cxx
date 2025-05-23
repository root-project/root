#include "TInterpreter.h"
#include "TFile.h"

#ifndef SUBCLASS_HH
#define SUBCLASS_HH
class MySubClass
{
public:
   int id;
};
#endif
#ifdef __ROOTCLING__
#pragma link C++ class MySubClass+;
#endif

// https://its.cern.ch/jira/browse/ROOT-5306
int execROOT5306()
{
   // Original file was generated as:
   // .L MySubClass.cxx+
   // TFile f("/tmp/mysub.root", "RECREATE");
   // MySubClass msc;
   // msc.id = 33;
   // f.WriteObjectAny(&msc, "MySubClass", "msc");
   // f.Close();
   // with MySubClass.cxx containing: class MySubClass { public: int id; ClassDef(MySubClass, 3) };
   TFile f("mysub.root", "READ");
   auto msc = f.Get<MySubClass>("msc");
   if (msc->id != 33) {
      printf("Error: id is %d vs 33\n", msc->id);
      return 3;
   }
   return 0;
}
