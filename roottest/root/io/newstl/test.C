#ifndef TEST_C
#define TEST_C

#include "TFile.h"
#include "TTree.h"
#include "TSystem.h"
#include "TLeafObject.h"
#include "TRegexp.h"
#include "TObjString.h"
#include "TBranchElement.h"

void TestError(const std::string &test, const char *msg);

template <class HolderClass> Bool_t checkHolder(const char *testname = "") {
   HolderClass *holder = new HolderClass( 0 );
   Bool_t result = holder->Verify(0,Form("%s checkHolder",testname),0);
   delete holder;
   holder = new HolderClass( 2 );
   result &= holder->Verify(2,Form("%s checkHolder",testname),0);
   return result;
}

template <class HolderClass> void write(const char *testname, int nEntry = 3) {
   bool testingTopLevelVectors = true;

   TString dirname = gROOT->GetVersion();
   dirname.ReplaceAll(".","-");
   dirname.ReplaceAll("/","-");

   gSystem->mkdir(dirname);
   gSystem->Unlink("latest");
   gSystem->Symlink(dirname,"latest");

   auto _filename = gSystem->ConcatFileName(dirname, testname );
   TString filename = _filename;
   delete [] _filename;
   filename += ".root";

   TFile *file = new TFile(filename,"RECREATE","stl test file",0);

   HolderClass *holder = new HolderClass( 0 );

   // Write(file,"scalar",holder->fScalar)
   // Write(file,"object",holder->fObject)
   // Write(file,"nested",holder->fNested)

   holder->Write("holder");

   TString classname = holder->IsA()->GetName();
   TTree *tree = new TTree("stltree","testing stl containers");
   tree->Branch("split_2.",classname,&holder,32000,-2);
   tree->Branch("split_1.",classname,&holder,32000,-1);
   tree->Branch("split0.",classname,&holder,32000,0);
   tree->Branch("split1.",classname,&holder,32000,1);
   tree->Branch("split2.",classname,&holder,32000,2);
   tree->Branch("split99.",classname,&holder,32000,99);

   if (testingTopLevelVectors) {
     TClass *cls = gROOT->GetClass(typeid(holder->fScalar));
     if (!cls) {
        TestError("TreeBuilding", Form("Writing holder class: Missing class for %s",
                                       typeid(holder->fScalar).name()));
     } else {
        tree->Branch("scalar0.",&(holder->fScalarPtr),32000,0);
        tree->Branch("scalar1.",&(holder->fScalarPtr),32000,1);
        tree->Branch("scalar2.",&(holder->fScalarPtr),32000,2);
        tree->Branch("scalar99.",&(holder->fScalarPtr),32000,99);
     }

     TClass *clo = gROOT->GetClass(typeid(holder->fObject));
     if (!clo) {
        TestError("TreeBuilding", Form("Writing holder class: Missing class for %s",
                  typeid(holder->fObject).name()));
     } else {
       tree->Branch("object0." ,&(holder->fObjectPtr),32000,0);
       tree->Branch("object1." ,&(holder->fObjectPtr),32000,1);
       tree->Branch("object2." ,&(holder->fObjectPtr),32000,2);
       tree->Branch("object99.",&(holder->fObjectPtr),32000,99);
     }

     TClass *cln = gROOT->GetClass(typeid(holder->fNested));
     if (!cln) {
        TestError("TreeBuilding", Form("Writing holder class: Missing class for %s",
                  typeid(holder->fNested).name()));
     } else {
        tree->Branch("nested0." ,&(holder->fNestedPtr),32000,0);
        tree->Branch("nested1." ,&(holder->fNestedPtr),32000,1);
        tree->Branch("nested2." ,&(holder->fNestedPtr),32000,2);
        tree->Branch("nested99.",&(holder->fNestedPtr),32000,99);
     }
   }
   for(int i=0; i<nEntry; i++) {
      holder->Reset(i);
      tree->Fill();
   }
   file->Write();
   delete file;
}

template <class HolderClass> bool verifyBranch(const char *testname, TTree *chain, const char *bname, int type = 0) {
   static HolderClass *gHolder = new HolderClass;
   HolderClass **add = 0;
   HolderClass *holder = 0;

   TBranch *branch = chain->GetBranch(bname);
   if (branch==0) {
      TestError("treeReading",Form("Missing branch: %s",bname));
      return false;
   }

   if (branch->InheritsFrom("TBranchObject")) {
      TLeafObject *tbo = dynamic_cast<TLeafObject*>(branch->GetListOfLeaves()->At(0));
      holder = (HolderClass*)(tbo->GetObject());

      if (holder==0) {
         TestError("treeReading",Form("BranchObject %s with holder == 0!",bname));
         return false;
      }
   } else {
      add = (HolderClass**)branch->GetAddress();
      if (add==0) {
         TestError("treeReading",Form("Branch %s with add == 0!",bname));
         return false;
      }
      void **p;
      switch (type) {
         case 0: holder = *add; break;
         case 1: p = (void**) &(gHolder->fScalarPtr); *p = ((TBranchElement*)branch)->GetObject(); break;
         case 2: p = (void**) &(gHolder->fObjectPtr); *p = ((TBranchElement*)branch)->GetObject(); break;
         case 3: p = (void**) &(gHolder->fNestedPtr); *p = ((TBranchElement*)branch)->GetObject(); break;
      }
   }

   int splitlevel = branch->GetSplitLevel();

   switch (type) {
      case 0: return holder->Verify(chain->GetTree()->GetReadEntry(),Form("%s %s",testname,bname),splitlevel);
      case 1: return gHolder->VerifyScalarPtr(chain->GetTree()->GetReadEntry(),Form("%s %s",testname,bname),splitlevel);
      case 2: return gHolder->VerifyObjectPtr(chain->GetTree()->GetReadEntry(),Form("%s %s",testname,bname),splitlevel);
      case 3: return gHolder->VerifyNestedPtr(chain->GetTree()->GetReadEntry(),Form("%s %s",testname,bname),splitlevel);
      default:
         TestError("treeReading",Form("Unknown type %d in verifyBranch",type));
         return false;
   }
}

template <class HolderClass> bool read(const char *dirname, const char *testname, int nEntry, bool current) {
   HolderClass *holder = 0;
   bool result = true;
   bool testingTopLevelVectors = true;


   auto _filename = gSystem->ConcatFileName(dirname, testname );
   TString filename = _filename;
   filename += ".root";
   delete [] _filename;

   if (!current && gSystem->AccessPathName(filename, kFileExists)) {
      // when reading old directory a missing files is not an error.
      // For example this happens if the run of roottest at the time was done
      // with FAST=yes
      return true;
   }

   TFile file(filename,"READ");

   if (file.IsZombie()) return false;

   holder = dynamic_cast<HolderClass*>( file.Get("holder") );
   if (!holder) {
      TestError("Reading",Form("Missing object: holder"));
      result = false;
   } else {
      result &= holder->Verify(0,Form("%s: write in dir",testname),0);
   }

   TTree *chain = dynamic_cast<TTree*>( file.Get("stltree") );

   if (!chain) {
      TestError("treeReading",Form("Missing TTree: stltree"));
      return false;
   }

   if (nEntry==0 || nEntry>chain->GetEntriesFast()) nEntry = (Int_t)chain->GetEntriesFast();
   for ( Int_t entryInChain = 0, entryInTree = chain->LoadTree(0);
         entryInTree >= 0 && entryInChain<nEntry;
         entryInChain++, entryInTree = chain->LoadTree(entryInChain)
         ) {

      if ( chain->GetEntry(entryInChain) == 0 ) {
         TestError("treeReading",Form("Nothing read for entry #%d",entryInChain));
         break;
      }

      result &= verifyBranch<HolderClass>(testname,chain,"split_2.");
      result &= verifyBranch<HolderClass>(testname,chain,"split_1.");
      result &= verifyBranch<HolderClass>(testname,chain,"split0.");
      result &= verifyBranch<HolderClass>(testname,chain,"split1.");
      result &= verifyBranch<HolderClass>(testname,chain,"split2.");
      result &= verifyBranch<HolderClass>(testname,chain,"split99.");

      if (testingTopLevelVectors) {
         // we know that they all fail! (missing dictionary)
         result &= verifyBranch<HolderClass>(testname,chain,"scalar0",1);
         result &= verifyBranch<HolderClass>(testname,chain,"scalar1",1);
         result &= verifyBranch<HolderClass>(testname,chain,"scalar2",1);
         result &= verifyBranch<HolderClass>(testname,chain,"scalar99",1);

         result &= verifyBranch<HolderClass>(testname,chain,"object0",2);
         result &= verifyBranch<HolderClass>(testname,chain,"object1",2);
         result &= verifyBranch<HolderClass>(testname,chain,"object2",2);
         result &= verifyBranch<HolderClass>(testname,chain,"object99",2);

         result &= verifyBranch<HolderClass>(testname,chain,"nested0",3);
         result &= verifyBranch<HolderClass>(testname,chain,"nested1",3);
         result &= verifyBranch<HolderClass>(testname,chain,"nested2",3);
         result &= verifyBranch<HolderClass>(testname,chain,"nested99",3);
      }
   }
   return result;
}

template <class HolderClass> bool read(const char *testname, int nEntry = 0, bool readother = false) {

   // for each dirname
   TString dirname = gROOT->GetVersion();
   dirname.ReplaceAll(".","-");
   dirname.ReplaceAll("/","-");

   bool result = true;
   result &= read<HolderClass>(dirname,testname, nEntry, /*current=*/true);

   if (readother) {
      TList listOfDirs;
      listOfDirs.SetOwner(kTRUE);
      fillListOfDir(listOfDirs);

      TIter next(&listOfDirs);
      while (TObjString *dir = (TObjString*)next()) {
         if (dirname != dir->GetName()) {
            std::cout << "Testing older file format from: " << dir->GetName() << std::endl;
            result &= read<HolderClass>(dir->GetName(),testname, nEntry, /*current=*/ false);
         }
      }
   }

   return result;
}
#endif

