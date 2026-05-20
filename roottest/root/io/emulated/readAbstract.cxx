#include "abstract.h"

class ConcreteAlternative : public Abstract
{
public:
   Float_t fData;
   ConcreteAlternative() : fData(-1) {}
   ~ConcreteAlternative() override { fprintf(stdout,"Running ConcreteAlternative's destructor\n"); }

   void Action() override { /* should be doing something useful */ }
};


#include "TFile.h"
#include "TTree.h"

void readAbstract(const char *filename = "abstract.root")
{
   // TClass::GetClass("ConcreteAlternative")->GetStreamerInfo(); // Force the initialization of Abstract's StreamerInfo.
   TFile *f = TFile::Open(filename,"READ");
   TTree *t; f->GetObject("t",t);
   if (t) {
      t->GetEntry(0);
      t->Scan("*");
   }
   delete f;
}
