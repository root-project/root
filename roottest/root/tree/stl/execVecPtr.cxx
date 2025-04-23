#include <iostream>

class Top {
public:
  Top() : fTopValue(-1) {}
  Top(int seed) : fTopValue(seed) {}
  int fTopValue;
  virtual void Print() {
     std::cout << " top::fTopValue= " << fTopValue; 
  }
  virtual ~Top() {}
};

class One : public Top {
public:
   One() : fOneValue(-1) {}
   One(int seed) : Top(seed), fOneValue(seed) {}
   int fOneValue;
   void Print() override {
     Top::Print();
     std::cout << " One::fOneValue= " << fOneValue;
   }
   ~One() override {}
};

class Two : public Top {
public:
   Two() : fTwoValue(-1) {}
   Two(int seed) : Top(seed),fTwoValue(seed) {}
   int fTwoValue;
   void Print() override {
     Top::Print();
     std::cout << " Two::fTwoValue= " << fTwoValue;
   }
   ~Two() override {}
};

#include <vector>
#include <iostream>

class Holder {
public:
   std::vector<Top*> fValues;
   
   void Fill(int seed) {
      fValues.clear();
      for(int i = 0; i < (1+seed) ; ++i) {
         fValues.push_back(new Top((100*(seed+1))+i));
         if (i>0) fValues.push_back(new One((200*(seed+1))+i));
         if (i>1) fValues.push_back(new Two((300*(seed+1))+i));
      }
   }
   void Print() {
      std::cout << "Printing holder:\n";
      for(unsigned int i = 0; i < fValues.size(); ++i) {
         std::cout << "i=" << i;
         fValues[i]->Print();
         std::cout << '\n';
      }
   }
};

#include "TFile.h"
#include "TTree.h"

void write() {
  TFile *f = new TFile("vecptr.root","RECREATE");
  TTree *t = new TTree("T","T");
  Holder out;
  t->Branch("holder.",&out,32000,25599);
  for(int i=0; i<4; ++i) {
     out.Fill(i);
     out.Print();
     t->Fill();
  }
  f->Write();
  delete f;
}

void read() {
  TFile *f = new TFile("vecptr.root");
  TTree *t; f->GetObject("T",t);
  Holder *out = 0;
  t->SetBranchAddress("holder.",&out);
  for(int i=0; i<t->GetEntries(); ++i) {
    t->GetEntry(i);
    if (out) out->Print();
  }
}

void execVecPtr()
{
   std::cout << "Writing\n";
   write();
   std::cout << "Reading\n";
   read();
}
