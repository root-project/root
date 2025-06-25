//#define OLD_FILE

#ifndef BASE_HH
#define BASE_HH

#include <cstdint>

class Base
{
  public:
    Base() : fI(0) {}
    virtual ~Base() {}

    void SetInt(uint64_t i) { fI = i; }

  protected:
#ifdef OLD_FILE
   unsigned long fI;
#else
   unsigned long long fI;
#endif
};

#endif
#ifndef DERIVED_HH
#define DERIVED_HH

#include "TObject.h"
//#include "Base.hh"
#include <vector>

class Derived : public TObject, public Base
{
  public:
    Derived() {}
    ~Derived() override {}

  ClassDefOverride(Derived,1)
};

#endif

#ifdef __ROOTCLING__

#pragma link C++ class Base+;
#pragma link C++ class Derived+;
#pragma link C++ class vector<Derived>+;

#endif

#include "TFile.h"
#include "TTree.h"
#include <cstdio>
#include "TClonesArray.h"

void writeROOT7500()
{
   TFile* file = TFile::Open("file7500.root", "RECREATE");
   Derived* dd = new Derived;
   //Base* dd = new Base; // replace previous line with this one and it will work!
   TTree* tree = new TTree("tree", "tree");
   tree->Branch("dd", &dd);
   dd->SetInt(13);
   TClonesArray *arr = new TClonesArray("Derived");
   Derived *dd2 = (Derived*)arr->ConstructedAt(0);
   dd2->SetInt(14);
   tree->Branch("arr",&arr,32000,0);

   vector<Derived> dd3;
   dd3.push_back(Derived());
   dd3[0].SetInt(15);
   tree->Branch("vec",&dd3);

   tree->Branch("vec0",&dd3,32000,0);

   tree->Fill();
   tree->Write();
   file->Close();
}

int execROOT7500()
{
   TFile* file = TFile::Open("file7500.root", "READ");
   if (!file || file->IsZombie()) {
      printf("Error: Can not open file7500.root\n");
      return 1;
   }
   TTree *tree(nullptr);
   file->GetObject("tree",tree);
   if (!tree) {
      printf("Error: Can not retrieve tree from %s\n",file->GetName());
      return 2;
   }
   auto n = tree->GetEntry(0);
   if (n != 141) {
      printf("Error: Read too few bytes (%d vs 411)\n",(int)n);
      return 3;
   }
   delete file;
   return 0;
}
