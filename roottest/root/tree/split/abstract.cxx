#include <vector>
#include "TFile.h"
#include "TTree.h"

class absTop {
public:
   int val;

   absTop() : val(-1) {}
   absTop(int v) : val(v) {};
   virtual ~absTop() {};

   virtual int getVal() = 0;
};

class absBot : public absTop {
public:
   
   absBot() : absTop(-1) {}
   absBot(int v) : absTop(v) {};
   ~absBot() override {};

   int getVal() override { return val; }
};

class Contained {
public:
   Contained() : id(-2),ptr(-1) {};
   Contained(int i, int p) : id(i),ptr(p) {};

   int id;
   absBot ptr;
};

class Event {
public:
   std::vector<Contained> vec;
   Event() {};
};

const char *def_filename = "abstract.root";

void writefile(const char *filename = def_filename) 
{
   TFile *f = new TFile(filename,"RECREATE");
   TTree *t = new TTree("T","tree");
   Event *e = new Event;
   e->vec.push_back(Contained(1,3));
   e->vec.push_back(Contained(2,7));
   
   t->Branch("obj.",&e);
   t->Fill();
   f->Write();
}

void readfile(const char *filename = def_filename)
{
   TFile *f = new TFile(filename,"READ");
   TTree *t; f->GetObject("T",t);
   t->GetEntry(0);
}
