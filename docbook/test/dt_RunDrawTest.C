#include "TCanvas.h"
#include "TClassTable.h"
#include "TFile.h"
#include "TH1.h"
#include "TKey.h"
#include "TChain.h"
#include "TSystem.h"

#include "TBranchElement.h"

#include "Riostream.h"

#include "dt_DrawTest.C"

Bool_t gInteractiveTest = kTRUE;
Int_t gQuietLevel = 0;

//_______________________________________________________________
Int_t HistCompare(TH1 *ref, TH1 *comp)
{
// Compare histograms h1 and h2
// Check number of entries, mean and rms
// if means differ by more than 1/1000 of the range return -1
// if means differ by more than 1/100 of the original mean return -2
// if rms differs in percent by more than 1/1000 return -3
// Otherwise return difference of number of entries

   Int_t n1       = (Int_t)ref->GetEntries();
   Double_t mean1 = ref->GetMean();
   Double_t rms1  = ref->GetRMS();
   Int_t n2       = (Int_t)comp->GetEntries();
   Double_t mean2 = comp->GetMean();
   Double_t rms2  = comp->GetRMS();

   Int_t factor = 1;
   if (n2==2*n1) {
     // we have a chain.
     factor = 2;
   }

   Float_t xrange = ref->GetXaxis()->GetXmax() - ref->GetXaxis()->GetXmin();
   if (xrange==0) { fprintf(stderr,"no range for %s\n",ref->GetName()); return -4; }
   if (xrange>0.0001 && TMath::Abs((mean1-mean2)/xrange) > 0.001) {
      printf("xrange=%g, mean1=%g, mean2=%g, abs=%g\n",xrange,mean1,mean2,TMath::Abs((mean1-mean2)/xrange));
      return -1;
   }
   if (mean2> 0.0001 && TMath::Abs((mean1-mean2)/mean2) > 0.01) {
      printf("mean1=%g, mean2=%g, abs=%g\n",mean1,mean2,TMath::Abs((mean1-mean2)/mean2));
      return -2;
   }
   if (rms1 > 0.0001 && TMath::Abs((rms1-rms2)/rms1) > 0.0003) {
      printf("rms1=%g, rms2=%g, abs=%g\n",rms1,rms2,TMath::Abs((rms1-rms2)/rms1));
      return -3;
   }
   return n1*factor-n2;
}

Int_t Compare(TDirectory* from) {
   TFile * reffile = new TFile("dt_reference.root");
   
   TIter next(reffile->GetListOfKeys());
   TH1 *ref, *draw;
   const char* name;
   Int_t comp;
   Int_t fail = 0;
   TKey* key;

   while ((key=(TKey*)next())) {
      if (strcmp(key->GetClassName(),"TH1F")
          && strcmp(key->GetClassName(),"TH2F") ) 
        continue; //may be a TList of TStreamerInfo
      ref = (TH1*)reffile->Get(key->GetName());
      name = ref->GetName();
      if (strncmp(name,"ref",3)) continue;
      name += 3;
      draw = (TH1*)from->Get(name);
      if (!draw) {
         if (!gSkipped.FindObject(name)) {
            cerr << "Miss: " << name << endl;
            fail++;
         }
         continue;
      }
      comp = HistCompare(ref,draw);
      if (comp!=0) {
         cerr << "Fail: " << name << ":" << comp << " " << ref->GetTitle() << endl;
         fail++;
         if (gInteractiveTest) {
            TCanvas * canv = new TCanvas();
            canv->Divide(2,1);
            canv->cd(1); 
            TString reftitle = "Ref: ";
            reftitle.Append(ref->GetTitle());
            ref->SetTitle(reftitle);
            ref->Draw();
            canv->cd(2); draw->Draw();
            return 1;
         }
      } else {
         if (gQuietLevel<1) cerr << "Succ: " << name << ":" << comp << endl;
      }
   }
   delete reffile;
   return fail;
}

void SetVerboseLevel(Int_t verboseLevel) {
   switch (verboseLevel) {
   case 0: gInteractiveTest = kFALSE;
     gQuietLevel = 2;
     break;
   case 1: gInteractiveTest = kFALSE;
     gQuietLevel = 1;
     break;
   case 2: gInteractiveTest = kFALSE;
     gQuietLevel = 0;
     break;
   case 3: gInteractiveTest = kTRUE;
     gQuietLevel = 0;
     break;
   }
}

bool dt_RunDrawTest(const char* from, Int_t mode = 0, Int_t verboseLevel = 0) {
  // This launch a test a TTree::Draw.
  // The mode currently available are:
  //    0: Do not load the shared library
  //    1: Load the shared library before opening the file
  //    2: Load the shared library after opening the file
  //    3: Simple TChain test with shared library
  //    4: Simple Friend test with shared library
  // The verboseLeve currently available:
  //    0: As silent as possible, only report errors and overall speed results.
  //    1: Output 0 + label for the start of each phase
  //    2: Output 1 + more details on the different phase being done
  //    3: Output 2 + stop at the first and draw a canvas showing the differences

//gDebug = 5;
   SetVerboseLevel(verboseLevel);

   if (mode == 1) {
      if (!TClassTable::GetDict("Event")) {
         gSystem->Load("libEvent");
     }     
      gHasLibrary = kTRUE;
   }

   TFile *hfile = 0;
   TTree *tree = 0;
   if (mode <3) {
      hfile = new TFile(from);
      tree = (TTree*)hfile->Get("T");
   }

   if (mode >= 2 && mode <= 4) {
      if (!TClassTable::GetDict("Event")) {
         gSystem->Load("libEvent");
      } else {
         cerr << "Since libEvent.so has already been loaded, mode 2 can not be tested!";
         cerr << endl;
      }
      gHasLibrary = kTRUE;
   }

   if (mode == 3) {
      // Test Chains.
      TChain * chain = new TChain("T");
      chain->Add(from);
      chain->Add(from);
      tree = chain;
   }

   if (mode == 4) {
      // Test friends.
      tree = new TTree("T","Base of friendship");
      tree->AddFriend("T",from);
   }

   TBranch *eb = tree->GetBranch("event");
   gBranchStyle = (int) eb->InheritsFrom(TBranchElement::Class());
   // cerr << "Branch style is " << gBranchStyle << endl;

   if (gQuietLevel<2) cout << "Generating histograms from TTree::Draw" << endl;
   TDirectory* where = GenerateDrawHist(tree,gQuietLevel);
 
   if (gQuietLevel<2) cout << "Comparing histograms" << endl;
   if (Compare(where)>0) {
     cout << "DrawTest: Comparison failed" << endl;
     return false;
   }
   DrawMarks();

   if (gQuietLevel<2) cout << "DrawTest: Comparison was successfull" << endl;
   if (hfile) delete hfile;
   else delete tree;
   gROOT->GetList()->Delete();

   return true;
}   
