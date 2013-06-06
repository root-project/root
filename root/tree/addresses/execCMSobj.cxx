#include <iostream>
#include "TString.h"

class Inside {
public:
   TString name;
   int object;  // will match the name of a containing things.
   
   Inside() : name("n/a/"),object(-1) {}
   Inside(const TString &n, int i): name(n),object(i) {}
   void Print() {
      std::cout << "name= " << name << " object=" << object;
   }
};

class Inter {
public:
   vector<Inside> vec;
   void Fill(int seed) {
      vec.clear();
      for(int i = 0; i < (2+seed) ; ++i) 
        vec.push_back(Inside(TString::Format("%d",seed),seed*100+i));
   }
   void Print() {
      std::cout << "Printing Outside\n";
      for(unsigned int i = 0 ; i < vec.size(); ++i) {
         std::cout << "Element " << i << ' ';
         vec[i].Print();
         std::cout << '\n';
      }
   }
};

class Outside {
public:
   bool present;
   Inter obj;
};

#include "TFile.h"
#include "TTree.h"

void write() {
  TFile *f = new TFile("cmsobj.root","RECREATE");
  TTree *t = new TTree("T","T");
  Outside out;
  t->Branch("out.",&out);
  for(int i=0; i<2; ++i) {
     out.obj.Fill(i);
     out.obj.Print();
     t->Fill();
  }
  f->Write();
  delete f;
}

void read() {
  TFile *f = new TFile("cmsobj.root");
  TTree *t; f->GetObject("T",t);
  Outside *out = 0;
  t->SetBranchAddress("out.",&out);
  for(int i=0; i<2; ++i) {
    t->GetEntry(i);
    if (out) out->obj.Print();
  }
}

void execCMSobj()
{
   std::cout << "Writing\n";
   write();
   std::cout << "Reading\n";
   read();
}
