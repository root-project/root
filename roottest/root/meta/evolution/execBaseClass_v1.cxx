// Testing unversioned base classes that changes.

#include "TFile.h"
#include "TTree.h"

class TopLevel {
public:
   TopLevel() : fValues(0) {}
   int fValues;
};

class BottomLevel : public TopLevel
{
public:
   BottomLevel(int i = 0) : fMore(i) {}
   int fMore;
};

class NestedLevel {
public:
   NestedLevel() : fNestValues(0) {}
   int fNestValues;
};

class OuterLevel {
public:
   OuterLevel(int i = 0) : fNestMore(i) {}
   int fNestMore;
   NestedLevel fNestContent;
};


class Holder {
public:
   Holder() {}
   Holder(int n) {
      for(int i = 0 ; i < n; ++i) fTracks.push_back(BottomLevel(i));
      for(int i = 0 ; i < n; ++i) fVertex.push_back(OuterLevel(i));
   }
   std::vector<BottomLevel> fTracks;
   std::vector<OuterLevel>  fVertex;
};

void write(const char *filename = "baseClass_v1.root")
{
   TFile * f = new TFile(filename,"RECREATE");
   Holder obj(2);
   f->WriteObject(&obj,"obj");
   TTree *tree = new TTree("tree","unversioning test");
   tree->Branch("obj.",&obj);
   tree->Fill();

   f->Write();
   delete f;
}

void read(const char *filename = "baseClass_v1.root")
{
   TFile * f = new TFile(filename,"READ");
   Holder *obj;
   f->GetObject("obj",obj);
   TTree *tree;
   f->GetObject("tree",tree);
   if (tree) tree->GetEntry(0);
   else fprintf(stderr,"Error: could not read the TTree.\n");
   delete f;
}

void execBaseClass_v1() {
   write();
   read();
}
