#include "TFile.h"
#include "TProcessID.h"
#include "TRef.h"
#include "TClonesArray.h"
#include "TTree.h"
#include "TObject.h"

#if defined(__ROOTCLING__) || !defined(__CLING__)
#include "A.C"
#else
#ifdef ClingWorkAroundMissingSmartInclude
#include "A.C"
#else
#include "A.C+"
#endif
#endif

#include <iostream>

void Check(TObject* obj) {

   if (obj)
      std::cout << "Found the referenced object\n";
   else
      std::cout << "Error: Could not find the referenced object\n";
}


void AddTree() {
   TTree* tree = new TTree("tree", "");

   TClonesArray* c1 = new TClonesArray("A", 1);
   TClonesArray* c2 = new TClonesArray("A", 1);

   tree->Branch("C1", "TClonesArray", &c1, 32000,3);
   tree->Branch("C2", "TClonesArray", &c2, 32000,3);

   A* a1 = new((*c1)[0]) A;
   A* a2 = new((*c2)[0]) A(a1);

   tree->Fill();

   c1->Clear();
   c2->Clear();

   a1 = new((*c1)[0]) A;
   a2 = new((*c2)[0]) A(a1);

   tree->Fill();

   tree->Write();
}

void ReadTree() {
   TTree* tree; gFile->GetObject("tree",tree);

   TClonesArray* c1 = new TClonesArray("A", 1);
   TClonesArray* c2 = new TClonesArray("A", 1);
   tree->SetBranchAddress("C1", &c1);
   tree->SetBranchAddress("C2", &c2);

   tree->GetEntry(0);
   tree->GetEntry(1);
}

const char *filename = "lotsRef.root";
void lotsRef(int what) {
   if (what > 2) {
      Int_t size = what;
      TFile *_file0 = TFile::Open(filename,"RECREATE");
      for(int i=0;i<size;++i) {
         TProcessID *id = TProcessID::AddProcessID();
         _file0->WriteProcessID(id);
      }
      AddTree();
      _file0->Write();
      delete _file0;
   } else if (what == 2) {
      TFile *_file0 = TFile::Open(filename,"UPDATE");

      TNamed *n = new TNamed("mine", "title");
      TRef *r = new TRef(n);
      n->Write();
      r->Write();
      AddTree();
      _file0->Write();
      delete _file0;
   } else if (what == 1) {
      TFile *_file0 = TFile::Open(filename,"UPDATE");

      int i = 0;
      while( _file0->ReadProcessID(++i) ) {};

      TNamed *n = nullptr;
      _file0->GetObject("mine",n);
      TRef *r = nullptr;
      _file0->GetObject("TRef",r);
      if (!r) { std::cerr << "Could not find the TRef on file \n"; return; }
      if (!n) { std::cerr << "Could not find the TNamed on file \n"; return;}
      ReadTree();
      Check( r->GetObject() );
      AddTree();
      n->Write();
      r->Write();
      _file0->Write();
      delete _file0;
   } else {
      TFile *_file0 = TFile::Open(filename);
      TRef *r = nullptr;
      _file0->GetObject("TRef", r);
      TNamed *n = nullptr;
      _file0->GetObject("mine", n);
      Check( r ? r->GetObject() : nullptr );
      ReadTree();
      delete _file0;
   }
}

