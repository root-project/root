// @(#)root/treeplayer:$Name:$:$Id:$
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
// TPlayer                                                              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TPlayer.h"

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

ClassImp(TPlayer)

//______________________________________________________________________________
TPlayer::TPlayer()
{
   // Default ctor.

   fInput  = new THashList;
   fOutput = new THashList;
   fOutput->IsOwner();
}

//______________________________________________________________________________
TPlayer::~TPlayer()
{
   delete fInput;
   delete fOutput;
}

//______________________________________________________________________________
void TPlayer::AddInput(TObject *inp)
{
   fInput->Add(inp);
}

//______________________________________________________________________________
void TPlayer::ClearInput()
{
   fInput->Clear();
}

//______________________________________________________________________________
TObject *TPlayer::GetOutput(const char *name) const
{
   return fOutput->FindObject( name );
}

//______________________________________________________________________________
TList *TPlayer::GetOutputList() const
{
   return fOutput;
}


//------------------------------------------------------------------------

ClassImp(TPlayerLocal)

//______________________________________________________________________________
Int_t TPlayerLocal::Process(TDSet *dset, const char *selector_file,
                            Int_t nentries, Int_t first,
                            TEventList *evl)
{
   ::Info("TPlayerLocal::Process","Voila!");

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
   selector->Begin();

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

   return 0;
}


//------------------------------------------------------------------------

ClassImp(TPlayerRemote)

//______________________________________________________________________________
TPlayerRemote::TPlayerRemote(TProof *proof) : fProof(proof)
{
}

//______________________________________________________________________________
Int_t TPlayerRemote::Process(TDSet *set, const char *selector_file,
                             Int_t nentries, Int_t first,
                             TEventList *evl)
{

   ::Info("TPlayerRemote::Process","Voila!");

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

ClassImp(TPlayerSlave)

//______________________________________________________________________________
TPlayerSlave::TPlayerSlave()
{
}


//______________________________________________________________________________
Int_t TPlayerSlave::Process(TDSet *dset, const char *selector_file,
                            Int_t nentries, Int_t first,
                            TEventList *evl)
{
   ::Info("TPlayerSlave::Process","Voila!");

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
   selector->Begin();

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

   return 0;
}

