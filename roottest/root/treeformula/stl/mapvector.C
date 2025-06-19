#include <vector>
#include <map>
#include "TFile.h"
#include "TTree.h"

class MyEvent {
public:
   std::map<int,std::vector<int> > data;
   void fill() {
      std::vector<int> tmp;
      tmp.push_back(5);
      tmp.push_back(7);
      data[3] = tmp;
   }
};

#ifdef __MAKECINT__
#pragma link C++ class pair<int,std::vector<int> >+;
#endif

using namespace std;

void writefile(const char *filename = "mapvector.root") 
{
   TFile * f = TFile::Open(filename,"RECREATE");
   MyEvent *e = new MyEvent;
   e->fill();
   TTree * t = new TTree("T","T");
   t->Branch("event",&e);
   t->Fill();
   f->Write();
   delete f;
}

void readfile(const char  *filename = "mapvector.root") 
{
   TFile * f = TFile::Open(filename,"READ");
   TTree *tree; f->GetObject("T",tree);
   tree->Scan("data.first");
   tree->Scan("data.second");
   delete f;
}

void mapvector() {
   writefile();
   readfile();
}
