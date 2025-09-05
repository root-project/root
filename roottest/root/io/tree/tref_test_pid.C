/*
  create(filename)

  clone(filename in, filename out)

  addref(filename in, filename out,
         branch name to add, branch name to reference)

  readall(filename)

  readauto(filename, branch to read)

 */


#include "TRef.h"
#include "TNamed.h"
#include "TFile.h"
#include "TTree.h"
#include "TBranchElement.h"
#include "TClonesArray.h"
#include "TObjString.h"
#include "TObjArray.h"
#include "TProcessID.h"
#include <list>

class tref_test_pid: public TNamed {
public:
   tref_test_pid() {}
   ~tref_test_pid() override {}

   tref_test_pid(const char* name, TObject* ref) {
      fName.Form("%s_%d", name, GetID());
      fRef = ref;
   }
   static Int_t GetID() {
      static Int_t currentMax=0;
      return currentMax++;
   }
   void Dump() const override {
      TNamed* r=(TNamed*)fRef.GetObject();
      printf("TTP %s ref: %s\n", GetName(), r ? r->GetName() : "(!NULL!)");
   }

   TRef  fRef; // a reference
   ClassDefOverride(tref_test_pid,1) // a roottest TRef autoloading object
};

void create(const char *filename) {
   TFile* file = new TFile(filename, "RECREATE");
   TTree* T = new TTree("T","T");
   TClonesArray* caN = new TClonesArray("TNamed");
   T->Branch("N", &caN);

   UInt_t objCount = TProcessID::GetObjectCount();
   for (int ev=0; ev<10; ++ev) {
      for (int ent=0; ent<9; ++ent) {
         TString n;
         n.Form("N_%d_%d",ev,ent);
         new ((*caN)[ent]) TNamed(n.Data(),n.Data());
      }
      T->Fill();
      caN->Clear();
      TProcessID::SetObjectCount(objCount);
   }
   T->Write();
   delete file;
}

void clone(const char* filein, const char* fileout) {
   TFile* fIn = new TFile(filein);
   TTree* tIn = 0;
   fIn->GetObject("T", tIn);

   TFile* fOut = new TFile(fileout, "RECREATE");
   TTree* tOut = tIn->CloneTree();
   tOut->Write();
   delete fIn;
   delete fOut;
}

void addref(const char* filenameIn, const char* filenameOut,
            const char* addbranch, const char* refbranch) {
   TFile* fIn = new TFile(filenameIn);
   TTree* tIn = 0;
   fIn->GetObject("T", tIn);

   TFile* fOut = new TFile(filenameOut, "RECREATE");
   TTree* tOut = tIn->CloneTree(0);
   tOut->BranchRef();
   TClonesArray* caR = new TClonesArray("tref_test_pid");
   tOut->Branch(addbranch, &caR);

   TBranchElement* brRef = dynamic_cast<TBranchElement*>(tIn->GetBranch(refbranch));
   if (!brRef) {
      printf("ERROR in addref: referenced branch %s is not a TBranchElement but a %s!\n",
             refbranch, tIn->GetBranch(refbranch)->IsA()->GetName());
      return;
   }

   TClonesArray* caN = new TClonesArray(brRef->GetClonesName());
   tIn->SetBranchAddress(refbranch, &caN);

   for (int ev=0; ev<tIn->GetEntries(); ++ev) {
      tIn->GetEntry(ev);
      UInt_t objCount = TProcessID::GetObjectCount();
      for (int ent=0; ent<9; ++ent) {
         TNamed *ref = (TNamed*)(*caN)[8-ent];
         TString name;
         name.Form("%s_%s_%d_%d", addbranch, refbranch, ev, ent);
         new ((*caR)[ent]) tref_test_pid(name.Data(), ref);
      }
      tOut->Fill();
      caR->Clear();
      TProcessID::SetObjectCount(objCount);
   }
   tOut->Write();
   delete fOut;
   delete fIn;
}

void readall(const char* filename) {
   TFile file(filename);
   TTree* T=0;
   file.GetObject("T", T);
   for (int ev=0; ev<T->GetEntries(); ++ev)
      T->Show(ev);
}

void readauto(const char* filename, const char* branch) {
   TFile file(filename);
   TTree* T=0;
   file.GetObject("T", T);
   std::list<std::pair<TBranch*,TClonesArray*> > branches;
   TString strBranches(branch);
   TObjArray* oaBranches = strBranches.Tokenize(":");
   TIter iBranch(oaBranches);
   TObjString* osBranch = 0;
   while ((osBranch = (TObjString*)iBranch())) {
      T->SetBranchStatus((osBranch->String()+"*").Data(), 1);
      TBranch* b=T->GetBranch(osBranch->String());
      TBranchElement* brRef = dynamic_cast<TBranchElement*>(b);
      TClonesArray* caN = new TClonesArray(brRef->GetClonesName());
      b->SetAddress(&caN);
      branches.push_back(std::make_pair(b, caN));
   }

   for (int ev=0; ev<T->GetEntries(); ++ev) {
      std::list<std::pair<TBranch*,TClonesArray*> >::iterator iBranch;
      for (iBranch=branches.begin(); iBranch!=branches.end(); iBranch++) {
         iBranch->first->GetEntry(ev);
         for (int e=0; e<=iBranch->second->GetLast(); ++e) {
            tref_test_pid* o=(tref_test_pid*)(*(iBranch->second))[e];
            o->Dump();
            TNamed* n=(TNamed*)o->fRef.GetObject();
            if (n) {
               TString name(n->GetName());
               // check that auto-loading doesn't reload
               // branches, overwriting our modified object
               name+=" GOOD!";
               n->SetName(name);
            }
            o->Dump();
         }
      }
   }
   delete oaBranches;
}

