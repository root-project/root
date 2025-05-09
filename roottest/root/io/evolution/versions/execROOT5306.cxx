#include "TInterpreter.h"
#include "TFile.h"

// https://its.cern.ch/jira/browse/ROOT-5306
class MySubClass
{
public:
   int id;
};

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
   gInterpreter->ProcessLine(".L MySubClassUnv.cxx+"); 
   TFile f("mysub.root", "READ");
   auto msc = f.Get<MySubClass>("msc");
   if (msc->id != 33) {
      printf("Error: id is %d vs 33\n", msc->id);
      return 3;
   }
   return 0;
}
