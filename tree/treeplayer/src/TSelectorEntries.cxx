// @(#)root/treeplayer:$Id$
// Author: Philippe Canal 09/06/2006

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TSelectorEntries
The class is derived from the ROOT class TSelector. For more
information on the TSelector framework see
$ROOTSYS/README/README.SELECTOR or the ROOT User Manual.

The following methods are defined in this file:

  - Begin():        called every time a loop on the tree starts,
                    a convenient place to create your histograms.
  - SlaveBegin():   called after Begin(), when on PROOF called only on the
                    slave servers.
  - Process():      called for each event, in this function you decide what
                    to read and fill your histograms.
  - SlaveTerminate: called at the end of the loop on the tree, when on PROOF
                    called only on the slave servers.
  - Terminate():    called at the end of the loop on the tree,
                    a convenient place to draw/fit your histograms.

To use this file, try the following session on your Tree T:
~~~{.cpp}
     Root > T->Process("TSelectorEntries.C")
     Root > T->Process("TSelectorEntries.C","some options")
     Root > T->Process("TSelectorEntries.C+")
~~~
*/

#include "TSelectorEntries.h"
#include "TTree.h"
#include "TTreeFormula.h"
#include "TSelectorScalar.h"

////////////////////////////////////////////////////////////////////////////////
/// Default, constructor.

TSelectorEntries::TSelectorEntries(TTree *tree, const char *selection) :
   fOwnInput(kFALSE), fChain(tree), fSelect(0), fSelectedRows(0), fSelectMultiple(kFALSE)
{
   if (selection && selection[0]) {
      TSelectorEntries::SetSelection(selection);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

TSelectorEntries::TSelectorEntries(const char *selection) :
   fOwnInput(kFALSE), fChain(0), fSelect(0), fSelectedRows(0), fSelectMultiple(kFALSE)
{
   TSelectorEntries::SetSelection(selection);
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor.

TSelectorEntries::~TSelectorEntries()
{
   delete fSelect; fSelect = nullptr;
   if (fOwnInput) {
      fInput->Delete();
      delete fInput;
      fInput = nullptr;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// The Begin() function is called at the start of the query.
/// When running with PROOF Begin() is only called on the client.
/// The tree argument is deprecated (on PROOF 0 is passed).

void TSelectorEntries::Begin(TTree *tree)
{
   TString option = GetOption();
   fChain = tree;
}

////////////////////////////////////////////////////////////////////////////////
/// The SlaveBegin() function is called after the Begin() function.
/// When running with PROOF SlaveBegin() is called on each slave server.
/// The tree argument is deprecated (on PROOF 0 is passed).

void TSelectorEntries::SlaveBegin(TTree *tree)
{
   fChain = tree;
   TString option = GetOption();

   SetStatus(0);
   fSelectedRows   = 0;
   TObject *selectObj = fInput->FindObject("selection");
   const char *selection = selectObj ? selectObj->GetTitle() : "";

   if (strlen(selection)) {
      fSelect = new TTreeFormula("Selection",selection,fChain);
      fSelect->SetQuickLoad(kTRUE);
      if (!fSelect->GetNdim()) {delete fSelect; fSelect = 0; return; }
   }
   if (fSelect && fSelect->GetMultiplicity()) fSelectMultiple = kTRUE;

   fChain->ResetBit(TTree::kForceRead);
}

////////////////////////////////////////////////////////////////////////////////
/// Read entry.

Int_t TSelectorEntries::GetEntry(Long64_t entry, Int_t getall)
{
   return fChain ? fChain->GetTree()->GetEntry(entry, getall) : 0;
}

////////////////////////////////////////////////////////////////////////////////
/// The Init() function is called when the selector needs to initialize
/// a new tree or chain. Typically here the branch addresses and branch
/// pointers of the tree will be set.
/// It is normally not necessary to make changes to the generated
/// code, but the routine can be extended by the user if needed.
/// Init() will be called many times when running on PROOF
/// (once per file to be processed).

void TSelectorEntries::Init(TTree * /* tree */)
{
}

////////////////////////////////////////////////////////////////////////////////
/// This function is called at the first entry of a new tree in a chain.

Bool_t TSelectorEntries::Notify()
{
   if (fSelect) fSelect->UpdateFormulaLeaves();
   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// The Process() function is called for each entry in the tree (or possibly
/// keyed object in the case of PROOF) to be processed. The entry argument
/// specifies which entry in the currently loaded tree is to be processed.
/// It can be passed to either TSelectorEntries::GetEntry() or TBranch::GetEntry()
/// to read either all or the required parts of the data. When processing
/// keyed objects with PROOF, the object is already loaded and is available
/// via the fObject pointer.
///
/// This function should contain the "body" of the analysis. It can contain
/// simple or elaborate selection criteria, run algorithms on the data
/// of the event and typically fill histograms.
///
/// The processing can be stopped by calling Abort().
///
/// Use fStatus to set the return value of TTree::Process().
///
/// The return value is currently not used.

Bool_t TSelectorEntries::Process(Long64_t /* entry */)
{
   if (!fSelectMultiple) {
      if (fSelect) {
         if ( fSelect->EvalInstance(0) ) {
            ++fSelectedRows;
         }
      } else {
         ++fSelectedRows;
      }
   } else if (fSelect) {
      // Grab the array size of the formulas for this entry
      Int_t ndata = fSelect->GetNdata();

      // No data at all, let's move on to the next entry.
      if (!ndata) return kTRUE;

      // Calculate the first values
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
   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Set the selection expression.

void TSelectorEntries::SetSelection(const char *selection)
{
   if (!fInput) {
      fOwnInput = kTRUE;
      fInput = new TList;
   }
   TNamed *cselection = (TNamed*)fInput->FindObject("selection");
   if (!cselection) {
      cselection = new TNamed("selection","");
      fInput->Add(cselection);
   }
   cselection->SetTitle(selection);
}

////////////////////////////////////////////////////////////////////////////////
/// The SlaveTerminate() function is called after all entries or objects
/// have been processed. When running with PROOF SlaveTerminate() is called
/// on each slave server.

void TSelectorEntries::SlaveTerminate()
{
   fOutput->Add(new TSelectorScalar("fSelectedRows",fSelectedRows));
}

////////////////////////////////////////////////////////////////////////////////
/// The Terminate() function is the last function to be called during
/// a query. It always runs on the client, it can be used to present
/// the results graphically or save the results to file.

void TSelectorEntries::Terminate()
{
   TSelectorScalar* rows = (TSelectorScalar*)fOutput->FindObject("fSelectedRows");
   if (rows)
   {
      fSelectedRows = rows->GetVal();
   } else {
      Error("Terminate","fSelectedRows is missing in fOutput");
   }
}
