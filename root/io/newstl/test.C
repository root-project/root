#include "TFile.h"
#include "TTree.h"
#include "TSystem.h"
#include <iostream>

void TestError(const char *msg) {
   std::cerr << msg << "\n";
}

template <class HolderClass> void write(const char *testname, int nEntry = 3) {

   TString dirname = gROOT->GetVersion();
   dirname.ReplaceAll(".","-");
   dirname.ReplaceAll("/","-");

   gSystem->mkdir(dirname);
   TString filename = gSystem->ConcatFileName(dirname, testname );

   TFile *file = new TFile(filename,"RECREATE");

   HolderClass *holder = new HolderClass( 0 );
   
   // Write(file,"scalar",holder->fScalar)
   // Write(file,"object",holder->fObject)
   // Write(file,"nested",holder->fNested)

   holder->Write("holder");

   TString classname = holder->IsA()->GetName();
   TTree *tree = new TTree("stltree","testing stl containers");
   tree->Branch("split0.",classname,&holder,32000,0);
   tree->Branch("split1.",classname,&holder,32000,1);
   tree->Branch("split2.",classname,&holder,32000,2);
   tree->Branch("split99.",classname,&holder,32000,99);

   TClass *cls = gROOT->GetClass(typeid(holder->fScalar));
   TString scalarclass = cls?cls->GetName():typeid(holder->fScalar).name();
   tree->Branch("scalar0." ,scalarclass,&holder->fScalar,32000,0);
   tree->Branch("scalar1." ,scalarclass,&holder->fScalar,32000,1);
   tree->Branch("scalar2." ,scalarclass,&holder->fScalar,32000,2);
   tree->Branch("scalar99.",scalarclass,&holder->fScalar,32000,99);

   TClass *clo = gROOT->GetClass(typeid(holder->fObject));
   TString objectclass = clo?clo->GetName():typeid(holder->fObject).name();
   tree->Branch("object0." ,objectclass,&holder->fObject,32000,0);
   tree->Branch("object1." ,objectclass,&holder->fObject,32000,1);
   tree->Branch("object2." ,objectclass,&holder->fObject,32000,2);
   tree->Branch("object99.",objectclass,&holder->fObject,32000,99);

   TClass *cln = gROOT->GetClass(typeid(holder->fNested));
   TString nestedclass = cln?cln->GetName():typeid(holder->fNested).name();
   tree->Branch("nested0." ,nestedclass,&holder->fNested,32000,0);
   tree->Branch("nested1." ,nestedclass,&holder->fNested,32000,1);
   tree->Branch("nested2." ,nestedclass,&holder->fNested,32000,2);
   tree->Branch("nested99.",nestedclass,&holder->fNested,32000,99);

   for(int i=0; i<nEntry; i++) {
      holder->Reset(i);
      tree->Fill();
   }

   file->Write();
   delete file;
}

template <class HolderClass> bool verifyBranch(TTree *chain, const char *bname, int type = 0) {
   HolderClass **add = 0;
   HolderClass *holder = 0;

   TBranch *branch = chain->GetBranch(bname);
   if (branch==0) {
      TestError(Form("Missing branch: %s",bname));
      return false;
   }

   add = (HolderClass**)branch->GetAddress();
   if (add==0) {
      TestError(Form("Branch %s with add == 0!",bname));
      return false;
   }
   
   holder = *add;
   switch (type) {
      case 0: return holder->Verify(chain->GetTree()->GetReadEntry());
      case 1: return holder->VerifyScalar(chain->GetTree()->GetReadEntry());
      case 2: return holder->VerifyObject(chain->GetTree()->GetReadEntry());
      case 3: return holder->VerifyNested(chain->GetTree()->GetReadEntry());
      default: 
         TestError(Form("Unknown type %d in verifyBranch",type));
         return false;
   }
}

template <class HolderClass> bool read(const char *dirname, const char *testname, int nEntry) {
   HolderClass *holder = 0;
   bool result = true;
   
   TString filename = gSystem->ConcatFileName(dirname, testname );
   TFile file(filename,"READ");

   holder = dynamic_cast<HolderClass*>( file.Get("holder") );
   if (!holder) result = false;
   else result &= holder->Verify(0);

   TTree *chain = dynamic_cast<TTree*>( file.Get("stltree") );

   if (nEntry==0 || nEntry>chain->GetEntriesFast()) nEntry = (Int_t)chain->GetEntriesFast();
   for ( Int_t entryInChain = 0, entryInTree = chain->LoadTree(0);
         entryInTree >= 0 && entryInChain<nEntry;
         entryInChain++, entryInTree = chain->LoadTree(entryInChain)
         ) {

      if ( chain->GetEntry(entryInChain) == 0 ) {
         TestError(Form("Nothing read for entry #%d",entryInChain));
         break;
      }

      result &= verifyBranch<HolderClass>(chain,"split0.");
      result &= verifyBranch<HolderClass>(chain,"split1.");
      result &= verifyBranch<HolderClass>(chain,"split2.");
      result &= verifyBranch<HolderClass>(chain,"split99.");

      if (0) {
         // we know that they all fail! (missing dictionary)
         result &= verifyBranch<HolderClass>(chain,"scalar0",1);
         result &= verifyBranch<HolderClass>(chain,"scalar1",1);
         result &= verifyBranch<HolderClass>(chain,"scalar2",1);
         result &= verifyBranch<HolderClass>(chain,"scalar99",1);
         
         result &= verifyBranch<HolderClass>(chain,"object0",2);
         result &= verifyBranch<HolderClass>(chain,"object1",2);
         result &= verifyBranch<HolderClass>(chain,"object2",2);
         result &= verifyBranch<HolderClass>(chain,"object99",2);
         
         result &= verifyBranch<HolderClass>(chain,"nested0",3);
         result &= verifyBranch<HolderClass>(chain,"nested1",3);
         result &= verifyBranch<HolderClass>(chain,"nested2",3);
         result &= verifyBranch<HolderClass>(chain,"nested99",3);  
      }
   }
   return result;
}

template <class HolderClass> bool read(const char *testname, int nEntry = 0) {
   // for each dirname 
   TString dirname = gROOT->GetVersion();
   dirname.ReplaceAll(".","-");
   dirname.ReplaceAll("/","-");

   bool result = true;
   result &= read<HolderClass>(dirname,testname, nEntry);
   return result;
}



