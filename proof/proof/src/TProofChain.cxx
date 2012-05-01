
// @(#)root/proof:$Id$
// Author: G. Ganis  Nov 2006

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TProofChain                                                          //
//                                                                      //
// A TChain proxy on PROOF.                                             //
// Uses an internal TDSet to handle processing.                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TProofChain.h"
#include "TDSet.h"
#include "TList.h"
#include "TProof.h"
#include "TROOT.h"
#include "TEventList.h"
#include "TEntryList.h"

ClassImp(TProofChain)

//______________________________________________________________________________
TProofChain::TProofChain() : TChain()
{
   // Crates a new PROOF chain proxy.

   fChain        = 0;
   fTree         = 0;
   fSet          = 0;
   fDirectory    = gDirectory;
   ResetBit(kOwnsChain);
}

//______________________________________________________________________________
TProofChain::TProofChain(TChain *chain, Bool_t gettreeheader) : TChain()
{
   // Crates a new PROOF chain proxy containing the files from the chain.

   fChain        = chain;
   fTree         = 0;
   fSet          = chain ? new TDSet((const TChain &)(*chain)) : 0;
   fDirectory    = gDirectory;
   if (gProof) {
      gProof->AddChain(chain);
      ConnectProof();
      if (gProof->IsLite()) {
         SetBit(kProofLite);
         fTree = fChain;
      } else {
         if (gettreeheader && fSet)
            fTree = gProof->GetTreeHeader(fSet);
      }
   }
   ResetBit(kOwnsChain);
}

//______________________________________________________________________________
TProofChain::TProofChain(TDSet *dset, Bool_t gettreeheader) : TChain()
{
   // Creates a new PROOF chain proxy containing the files from the dset.

   fChain        = 0;
   fTree         = 0;
   fSet          = dset;
   fDirectory    = gDirectory;
   if (gProof) {
      ConnectProof();
      if (gettreeheader && dset)
         fTree = gProof->GetTreeHeader(dset);
      if (gProof->IsLite())
         SetBit(kProofLite);
   }
   if (fTree && fSet) {
      fChain = new TChain(fTree->GetName());
      TIter nxe(fSet->GetListOfElements());
      TDSetElement *e = 0;
      while ((e = (TDSetElement *) nxe())) {
         fChain->AddFile(e->GetName());
      }
      SetBit(kOwnsChain);
      if (TestBit(kProofLite))
         fTree = fChain;
   }
}

//______________________________________________________________________________
TProofChain::~TProofChain()
{
   // Destructor.

   if (fChain) {
      SafeDelete(fSet);
      // Remove the chain from the private lists in the TProof objects
      TIter nxp(gROOT->GetListOfSockets());
      TObject *o = 0;
      TProof *p = 0;
      while ((o = nxp()))
         if ((p = dynamic_cast<TProof *>(o)))
            p->RemoveChain(fChain);
      if (fTree == fChain) fTree = 0;
      if (TestBit(kOwnsChain)) {
         SafeDelete(fChain);
      } else {
         fChain = 0;
      }
   } else {
      // Not owner
      fSet = 0;
   }
   SafeDelete(fTree);
   fDirectory    = 0;

}

//______________________________________________________________________________
void TProofChain::Browse(TBrowser *b)
{
   // Forwards the execution to the dummy tree header.
   // See TTree::Browse().

   fSet->Browse(b);
}

//______________________________________________________________________________
Long64_t TProofChain::Draw(const char *varexp, const TCut &selection,
                           Option_t *option, Long64_t nentries, Long64_t firstentry)
{
   // Forwards the execution to the TDSet.
   // Returns -1 in case of error or number of selected events in case of success.
   // See TDSet::Browse().

   if (!gProof) {
      Error("Draw", "no active PROOF session");
      return -1;
   }
   ConnectProof();

   fReadEntry = firstentry;

   // Set either the entry-list (priority) or the event-list
   if (fEntryList) {
      fSet->SetEntryList(fEntryList);
   } else if (fEventList) {
      fSet->SetEntryList(fEventList);
   }

   // Fill drawing attributes
   FillDrawAttributes(gProof);

   Long64_t rv = fSet->Draw(varexp, selection, option, nentries, firstentry);
   return rv;
}

//______________________________________________________________________________
Long64_t TProofChain::Draw(const char *varexp, const char *selection,
                           Option_t *option,Long64_t nentries, Long64_t firstentry)
{
   // Forwards the execution to the TDSet.
   // Returns -1 in case of error or number of selected events in case of success.
   // See TDSet::Browse().

   if (!gProof) {
      Error("Draw", "no active PROOF session");
      return -1;
   }
   ConnectProof();

   fReadEntry = firstentry;

   // Set either the entry-list (priority) or the event-list
   if (fEntryList) {
      fSet->SetEntryList(fEntryList);
   } else if (fEventList) {
      fSet->SetEntryList(fEventList);
   }

   // Fill drawing attributes
   FillDrawAttributes(gProof);

   Long64_t rv = fSet->Draw(varexp, selection, option, nentries, firstentry);
   return rv;
}

//______________________________________________________________________________
void TProofChain::FillDrawAttributes(TProof *p)
{
   // Communicate the drawing attributes for this chain to the input list
   // so that the draw selectors can use them, in case of need.
   // The drawing attributes are:
   //
   //    LineColor          Line color
   //    LineStyle          Line style
   //    LineWidth          Line width
   //    MarkerColor        Marker color index
   //    MarkerSize         Marker size
   //    MarkerStyle        Marker style
   //    FillColor          Area fill color
   //    FillStyle          Area fill style

   if (!p || !fChain) {
      Error("FillDrawAttributes", "invalid PROOF or mother chain pointers!");
      return;
   }
   
   // Weight
   p->SetParameter("PROOF_ChainWeight", fChain->GetWeight());

   // Line Attributes
   p->SetParameter("PROOF_LineColor", (Int_t) fChain->GetLineColor());
   p->SetParameter("PROOF_LineStyle", (Int_t) fChain->GetLineStyle());
   p->SetParameter("PROOF_LineWidth", (Int_t) fChain->GetLineWidth());

   // Marker Attributes
   p->SetParameter("PROOF_MarkerColor", (Int_t) fChain->GetMarkerColor());
   p->SetParameter("PROOF_MarkerSize", (Int_t) fChain->GetMarkerSize()*1000);
   p->SetParameter("PROOF_MarkerStyle", (Int_t) fChain->GetMarkerStyle());

   // Area fill attributes
   p->SetParameter("PROOF_FillColor", (Int_t) fChain->GetFillColor());
   p->SetParameter("PROOF_FillStyle", (Int_t) fChain->GetFillStyle());

   if (gDebug > 0) {
      Info("FillDrawAttributes","line:   color:%d, style:%d, width:%d",
           fChain->GetLineColor(), fChain->GetLineStyle(), fChain->GetLineWidth());
      Info("FillDrawAttributes","marker: color:%d, style:%d, size:%f",
           fChain->GetMarkerColor(), fChain->GetMarkerStyle(), fChain->GetMarkerSize());
      Info("FillDrawAttributes","area:   color:%d, style:%d",
           fChain->GetFillColor(), fChain->GetFillStyle());
   }
}

//______________________________________________________________________________
TBranch *TProofChain::FindBranch(const char* branchname)
{
   // Forwards the execution to the dummy tree header.
   // See TTree::FindBranch().

   return (fTree ? fTree->FindBranch(branchname) : (TBranch *)0);
}

//______________________________________________________________________________
TLeaf *TProofChain::FindLeaf(const char* searchname)
{
   // Forwards the execution to the dummy tree header.
   // See TTree::FindLeaf().

   return (fTree ? fTree->FindLeaf(searchname) : (TLeaf *)0);
}

//______________________________________________________________________________
TBranch *TProofChain::GetBranch(const char *name)
{
   // Forwards the execution to the dummy tree header.
   // See TTree::GetBranch().

   return (fTree ? fTree->GetBranch(name) : (TBranch *)0);
}

//______________________________________________________________________________
Bool_t TProofChain::GetBranchStatus(const char *branchname) const
{
   // Forwards the execution to the dummy tree header.
   // See TTree::GetBranchStatus().

   return (fTree ? fTree->GetBranchStatus(branchname) : kFALSE);
}

//______________________________________________________________________________
TVirtualTreePlayer *TProofChain::GetPlayer()
{
   // Forwards the execution to the dummy tree header.
   // See TTree::GetPlayer().

   return (fTree ? fTree->GetPlayer() : (TVirtualTreePlayer *)0);
}

//______________________________________________________________________________
Long64_t TProofChain::Process(const char *filename, Option_t *option,
                              Long64_t nentries, Long64_t firstentry)
{
   // Forwards the execution to the TDSet.
   // The return value is -1 in case of error and TSelector::GetStatus() in
   // in case of success.
   // See TDSet::Process().

   // Set either the entry-list (priority) or the event-list
   if (fEntryList) {
      fSet->SetEntryList(fEntryList);
   } else if (fEventList) {
      fSet->SetEntryList(fEventList);
   }

   return fSet->Process(filename, option, nentries, firstentry);
}

//______________________________________________________________________________
Long64_t TProofChain::Process(TSelector *selector, Option_t *option,
                              Long64_t nentries, Long64_t firstentry)
{
   // The return value is -1 in case of error and TSelector::GetStatus() in
   // in case of success.

   // Set either the entry-list (priority) or the event-list
   if (fEntryList) {
      fSet->SetEntryList(fEntryList);
   } else if (fEventList) {
      fSet->SetEntryList(fEventList);
   }

   return fSet->Process(selector, option, nentries, firstentry);
}

//______________________________________________________________________________
void TProofChain::SetDebug(Int_t level, Long64_t min, Long64_t max)
{
   // See TTree::SetDebug

   TTree::SetDebug(level, min, max);
}

//______________________________________________________________________________
void TProofChain::SetName(const char *name)
{
   // See TTree::GetName.

   TTree::SetName(name);
}

//______________________________________________________________________________
Long64_t TProofChain::GetEntries() const
{
   // Returns the total number of entries in the TProofChain, which is
   // the number of entries in the TDSet that it holds.

   // this was used for holding the total number of entries
   if (TestBit(kProofLite)) {
      return (fTree ? fTree->GetEntries() : (Long64_t)(-1));
   } else {
      return (fTree ? fTree->GetMaxEntryLoop() : (Long64_t)(-1));
   }
}

//______________________________________________________________________________
Long64_t TProofChain::GetEntries(const char *selection)
{
   // See TTree::GetEntries(const char *selection)
   // Not implemented in TProofChain. Shouldn't be used.

   if (TestBit(kProofLite)) {
      return (fTree ? fTree->GetEntries(selection) : (Long64_t)(-1));
   } else {
      Warning("GetEntries", "GetEntries(selection) not yet implemented");
      return ((Long64_t)(-1));
   }
}

//______________________________________________________________________________
void TProofChain::Progress(Long64_t total, Long64_t processed)
{
   // Changes the number of processed entries.

   if (gROOT->IsInterrupted() && gProof)
      gProof->StopProcess(kTRUE);
   if (total) { }

   fReadEntry = processed;
}

//______________________________________________________________________________
Long64_t TProofChain::GetReadEntry() const
{
   // Returns the number of processed entries.

   return fReadEntry;
}

//______________________________________________________________________________
void TProofChain::ReleaseProof()
{
   // Releases PROOF. Disconnect the "progress" signal.

   if (!gProof)
      return;
   gProof->Disconnect("Progress(Long64_t,Long64_t)",
                      this, "Progress(Long64_t,Long64_t)");
}

//______________________________________________________________________________
void TProofChain::ConnectProof()
{
   // Connects the proof "Progress" signal.

   if (gProof)
      gProof->Connect("Progress(Long64_t,Long64_t)", "TProofChain",
                       this, "Progress(Long64_t,Long64_t)");
}
