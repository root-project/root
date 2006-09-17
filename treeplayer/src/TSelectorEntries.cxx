#define TSelectorEntries_cxx
// The class definition in TSelectorEntries.h has been generated automatically
// by the ROOT utility TTree::MakeSelector(). This class is derived
// from the ROOT class TSelector. For more information on the TSelector
// framework see $ROOTSYS/README/README.SELECTOR or the ROOT User Manual.

// The following methods are defined in this file:
//    Begin():        called everytime a loop on the tree starts,
//                    a convenient place to create your histograms.
//    SlaveBegin():   called after Begin(), when on PROOF called only on the
//                    slave servers.
//    Process():      called for each event, in this function you decide what
//                    to read and fill your histograms.
//    SlaveTerminate: called at the end of the loop on the tree, when on PROOF
//                    called only on the slave servers.
//    Terminate():    called at the end of the loop on the tree,
//                    a convenient place to draw/fit your histograms.
//
// To use this file, try the following session on your Tree T:
//
// Root > T->Process("TSelectorEntries.C")
// Root > T->Process("TSelectorEntries.C","some options")
// Root > T->Process("TSelectorEntries.C+")
//

#include "TSelectorEntries.h"
#include "TTreeFormula.h"
#include "TSelectorScalar.h"

TSelectorEntries::TSelectorEntries(TTree *tree, const char *selection) : 
   fChain(tree), fSelect(0), fSelectedRows(0), fSelectMultiple(kFALSE) 
{ 
   // Default, constructor.   

   if (selection && selection[0]) {
      TSelectorEntries::SetSelection(selection);
   }
}

TSelectorEntries::TSelectorEntries(const char *selection) : 
   fSelect(0), fSelectedRows(0), fSelectMultiple(kFALSE) 
{ 
   // Constructor.

   TSelectorEntries::SetSelection(selection);
}

TSelectorEntries::~TSelectorEntries() 
{
   // Destructor.

   delete fSelect; fSelect = 0;
}

void TSelectorEntries::Begin(TTree *tree)
{
   // The Begin() function is called at the start of the query.
   // When running with PROOF Begin() is only called on the client.
   // The tree argument is deprecated (on PROOF 0 is passed).

   TString option = GetOption();
   fChain = tree;
}

void TSelectorEntries::SlaveBegin(TTree *tree)
{
   // The SlaveBegin() function is called after the Begin() function.
   // When running with PROOF SlaveBegin() is called on each slave server.
   // The tree argument is deprecated (on PROOF 0 is passed).

   fChain = tree;
   TString option = GetOption();

   SetStatus(0);
   fSelectedRows   = 0;
   const char *selection = fInput->FindObject("selection")->GetTitle();

   if (strlen(selection)) {
      fSelect = new TTreeFormula("Selection",selection,fChain);
      fSelect->SetQuickLoad(kTRUE);
      if (!fSelect->GetNdim()) {delete fSelect; fSelect = 0; return; }
   }
   if (fSelect && fSelect->GetMultiplicity()) fSelectMultiple = kTRUE;

   fChain->ResetBit(TTree::kForceRead);
}

//______________________________________________________________________________
void TSelectorEntries::Init(TTree *tree)
{
   // The Init() function is called when the selector needs to initialize
   // a new tree or chain. Typically here the branch addresses and branch
   // pointers of the tree will be set.
   // It is normaly not necessary to make changes to the generated
   // code, but the routine can be extended by the user if needed.
   // Init() will be called many times when running on PROOF
   // (once per file to be processed).

}

//______________________________________________________________________________
Bool_t TSelectorEntries::Notify()
{
   // This function is called at the first entry of a new tree in a chain.

   if (fSelect) fSelect->UpdateFormulaLeaves();
   return kTRUE;
}

Bool_t TSelectorEntries::Process(Long64_t entry)
{
   // The Process() function is called for each entry in the tree (or possibly
   // keyed object in the case of PROOF) to be processed. The entry argument
   // specifies which entry in the currently loaded tree is to be processed.
   // It can be passed to either TSelectorEntries::GetEntry() or TBranch::GetEntry()
   // to read either all or the required parts of the data. When processing
   // keyed objects with PROOF, the object is already loaded and is available
   // via the fObject pointer.
   //
   // This function should contain the "body" of the analysis. It can contain
   // simple or elaborate selection criteria, run algorithms on the data
   // of the event and typically fill histograms.
   //
   // The processing can be stopped by calling Abort().
   //
   // Use fStatus to set the return value of TTree::Process().
   //
   // The return value is currently not used.

   if (!fSelectMultiple) {
      if (fSelect) {
         if ( fSelect->EvalInstance(0) ) {
            ++fSelectedRows;
         }
      } else {
         ++fSelectedRows;
      }
   } else {
      // Grab the array size of the formulas for this entry
      Int_t ndata = fSelect->GetNdata();

      // No data at all, let's move on to the next entry.
      if (!ndata) return kTRUE;

      // Calculate the first values
      if (fSelect) {
         // Always call EvalInstance(0) to insure the loading
         // of the branches.
         if (fSelect->EvalInstance(0)) {
            ++fSelectedRows;
         } else {
            for (Int_t i=1;i<ndata;i++) {
               if (fSelect->EvalInstance(i)) {
                  ++fSelectedRows;
                  break;
               }
            }
         }
      }
   }
   return kTRUE;
}

void TSelectorEntries::SetSelection(const char *selection)
{
   if (!fInput) {
      fInput = new TList;
   }
   TNamed *cselection = (TNamed*)fInput->FindObject("selection");
   if (!cselection) {
      cselection = new TNamed("selection","");
      fInput->Add(cselection);
   }
   cselection->SetTitle(selection);
}

void TSelectorEntries::SlaveTerminate()
{
   // The SlaveTerminate() function is called after all entries or objects
   // have been processed. When running with PROOF SlaveTerminate() is called
   // on each slave server.

   fOutput->Add(new TSelectorScalar("fSelectedRows",fSelectedRows));
}

void TSelectorEntries::Terminate()
{
   // The Terminate() function is the last function to be called during
   // a query. It always runs on the client, it can be used to present
   // the results graphically or save the results to file.

   TSelectorScalar* rows = (TSelectorScalar*)fOutput->FindObject("fSelectedRows");
   if (rows)
   {
      fSelectedRows = rows->GetVal();
   } else {
      Error("Terminate","fSelectedRows is missing in fOutput");
   }
}
