// @(#)root/proof:$Name:  $:$Id: TProofPlayer.cxx,v 1.1 2002/01/15 00:45:20 rdm Exp $
// Author: Maarten Ballintijn   07/01/02

/*************************************************************************
 * Copyright (C) 1995-2001, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TProofPlayer                                                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TProofPlayer.h"

#include "THashList.h"
#include "TEventIter.h"
#include "TSelector.h"
#include "TProof.h"
#include "TROOT.h"
#include "TError.h"
#include "MessageTypes.h"
#include "TMessage.h"
#include "TDSet.h"
#include "TString.h"



//------------------------------------------------------------------------

ClassImp(TProofPlayer)

//______________________________________________________________________________
TProofPlayer::TProofPlayer()
{
   // Default ctor.

   fInput  = new THashList;
   fOutput = 0;
}

//______________________________________________________________________________
TProofPlayer::~TProofPlayer()
{
   delete fInput;
}

//______________________________________________________________________________
void TProofPlayer::AddInput(TObject *inp)
{
   fInput->Add(inp);
}

//______________________________________________________________________________
void TProofPlayer::ClearInput()
{
   fInput->Clear();
}

//______________________________________________________________________________
TObject *TProofPlayer::GetOutput(const char *name) const
{
   return fOutput->FindObject(name);
}

//______________________________________________________________________________
TList *TProofPlayer::GetOutputList() const
{
   return fOutput;
}


//------------------------------------------------------------------------

ClassImp(TProofPlayerLocal)

//______________________________________________________________________________
Int_t TProofPlayerLocal::Process(TDSet *dset, const char *selector_file,
                                 Int_t nentries, Int_t first,
                                 TEventList *evl)
{
   ::Info("TProofPlayerLocal::Process","Voila!");

   // create TSelector
   TSelector *selector = TSelector::GetSelector(selector_file);

   if ( !selector ) {
      Error("Process", "Cannot load: %s", selector_file );
      return -1;
   }

   // create TEventIter
   TEventIter *evIter = new TEventIter(dset);

   // Init
   selector->SetInputList(fInput);
   selector->Begin(0);

   // Loop
   while (evIter->GetNextEvent(selector)) {

      Bool_t stop = selector->Process();
      if (stop) {}  // remove unused warning

      if (gROOT->IsInterrupted()) break;

   }

   // Finalize
   selector->Terminate();

   fOutput = selector->GetOutputList();

   // add: retrieve output


   delete selector;  // deletes output list

   return 0;
}


//------------------------------------------------------------------------

ClassImp(TProofPlayerRemote)

//______________________________________________________________________________
TProofPlayerRemote::TProofPlayerRemote(TProof *proof) : fProof(proof)
{
}

//______________________________________________________________________________
Int_t TProofPlayerRemote::Process(TDSet *set, const char *selector_file,
                                  Int_t nentries, Int_t first,
                                  TEventList *evl)
{

   ::Info("TProofPlayerRemote::Process","Voila!");

   TString filename = selector_file;
   filename = filename.Strip(TString::kTrailing,'+');
   fProof->SendFile(filename);

   TMessage mesg(kPROOF_PROCESS);
   TString fn(selector_file);

   mesg << set << fn << fInput << nentries << first; // no evl yet

   fProof->Broadcast(mesg);

   fProof->Collect();

   // extract output ...

   return 0;
}

//------------------------------------------------------------------------

ClassImp(TProofPlayerSlave)

//______________________________________________________________________________
TProofPlayerSlave::TProofPlayerSlave()
{
}


//______________________________________________________________________________
Int_t TProofPlayerSlave::Process(TDSet *dset, const char *selector_file,
                                 Int_t nentries, Int_t first,
                                 TEventList *evl)
{
   ::Info("TProofPlayerSlave::Process","Voila!");

   // create TSelector
   TSelector *selector = TSelector::GetSelector(selector_file);

   if ( !selector ) {
      Error("Process", "Cannot load: %s", selector_file );
      return -1;
   }

   // create TEventIter
   TEventIter *evIter = new TEventIter(dset);

   // Init
   selector->SetInputList(fInput);
   selector->Begin(0);

   // Loop
   while (evIter->GetNextEvent(selector)) {

      Bool_t stop = selector->Process();
      if (stop) {}  // remove unused warning

      if (gROOT->IsInterrupted()) break;

   }

   // Finalize
   selector->Terminate();

   fOutput = selector->GetOutputList();

   // add: retrieve output


   delete selector;     // deletes output list

   return 0;
}

