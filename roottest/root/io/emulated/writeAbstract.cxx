#include "abstract.h"

class Concrete : public Abstract
{
public:
   Float_t fData;
   Concrete() : Abstract(3),fData(2) {}
   virtual ~Concrete() { fprintf(stdout,"Running Concrete's destructor\n"); }

   virtual void Action() { /* should be doing something useful */ }
};

#include "TFile.h"
#include "TTree.h"

void writeAbstract(const char *filename = "abstract.root")
{
   TFile *f = TFile::Open(filename,"RECREATE");
   TTree *t = new TTree("t","t");
   Concrete object;
   t->Branch("obj.",&object);
   t->Fill();
   f->Write();
   delete f;
}   
