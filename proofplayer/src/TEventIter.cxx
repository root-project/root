// @(#)root/proof:$Name:  $:$Id: TEventIter.cxx,v 1.1 2002/01/18 14:24:09 rdm Exp $
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
// TEventIter                                                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TEventIter.h"

#include "TDSet.h"
#include "TSelector.h"
#include "TKey.h"
#include "TFile.h"
#include "TCollection.h"
#include "TError.h"
#include "TTree.h"


//------------------------------------------------------------------------

ClassImp(TEventIter)


//______________________________________________________________________________
TEventIter::TEventIter()
{
}


//______________________________________________________________________________
TEventIter::~TEventIter()
{
}


//------------------------------------------------------------------------

ClassImp(TEventIterLocal)


//______________________________________________________________________________
TEventIterLocal::TEventIterLocal()
{
   // Default ctor.

   fIsTree   = kFALSE;

   fFiles    = 0;
   fNextFile = 0;
   fFile     = 0;

   fKeys     = 0;
   fNextKey  = 0;
   fObj      = 0;

   fTree     = 0;
}

//______________________________________________________________________________
TEventIterLocal::TEventIterLocal(TDSet *dset)
{

   fIsTree = dset->IsTree();

   fFiles = dset->GetListOfElements();
   fNextFile = new TIter(fFiles);
   fFile     = 0;

   if ( fIsTree ) {
      // ...
   } else {
      fClassName = dset->GetType();
   }

   fKeys     = 0;
   fNextKey  = 0;
   fObj      = 0;

   fTree     = 0;
   fMaxEntry = -1;
}


//______________________________________________________________________________
TEventIterLocal::~TEventIterLocal()
{
   // TODO: check cleanup

   delete fNextFile;
   delete fFile;

   // delete fKeys; ? who owns it?
   delete fNextKey;
   delete fObj;

   // delete fTree; ? who owns it
}


//______________________________________________________________________________
Bool_t TEventIterLocal::Init(TSelector *selector)
{
   if ( fIsTree ) {
      LoadNextTree();
   }

   selector->Begin(fTree);

   return kTRUE;
}


//______________________________________________________________________________
Bool_t TEventIterLocal::GetNextEvent(TSelector *selector)
{
   Bool_t   ok;

   if ( fIsTree ) {
         ok = GetNextEventTree(selector);
   } else {
         ok = GetNextEventObj(selector);
   }

   return ok;
}


//______________________________________________________________________________
Bool_t TEventIterLocal::GetNextEventTree(TSelector *selector)
{

   if ( fFiles == 0 ) {
      ::Warning("TEventIterLocal::GetNextEvent", "Not initialized");
      return kFALSE;
   }

   while ( kTRUE ) {
      if ( fMaxEntry == -1 ) {

         Bool_t ok = LoadNextTree();

         if ( !ok ) return kFALSE;

         selector->Notify();
      }

      if ( ++fEntry < fMaxEntry ) {

         // selector->SetEvent( fIevent );

         fTree->GetEntry(fEntry);
         return kTRUE;

      }

      // Done with this file
      fMaxEntry = -1;

   }

}

//______________________________________________________________________________
Bool_t TEventIterLocal::LoadNextTree()
{
   delete fFile; fFile = 0;
   TDSetElement *elem;
   if ( (elem = (TDSetElement*) fNextFile->Next()) == 0 ) {
         // Done with all files
         return kFALSE;
   }

   Info("TEventIterLocal::LoadNextTree","Next File: %s", elem->GetFileName());

   // need to do possible grid translation

   fFile = new TFile(elem->GetFileName());
   if ( fFile->IsZombie() ) {
      Warning("TEventIterLocal::LoadNextTree","Error opening %s",
               elem->GetFileName());
      return kFALSE;
   }

   if ( elem->GetDirectory() != 0 )
      fFile->Cd(elem->GetDirectory());

   fTree = (TTree *) fFile->Get(elem->GetObjName());

   fMaxEntry = fTree->GetEntries();
   fEntry = -1;

   return kTRUE;
}


//______________________________________________________________________________
Bool_t TEventIterLocal::GetNextEventObj(TSelector *selector)
{
   TKey  *next;

   if ( fFiles == 0 ) {
      ::Warning("TEventIterLocal::GetNextEvent", "Not initialized");
      return kFALSE;
   }

   Info("TEventIterLocal::GetNextEventObj","fkeys %p", fKeys );
   while ( kTRUE ) {
      if ( fKeys == 0 ) {
         delete fFile; fFile = 0;
         TDSetElement *elem;
         if ( (elem = (TDSetElement*) fNextFile->Next()) == 0 ) {
               // Done with all files
               return kFALSE;
         }

         Info("TEventIterLocal::GetNextEventObj","Next File: %s", elem->GetFileName());

         // need to do possible grid translation

         fFile = new TFile(elem->GetFileName());   // TODO: check result

         // cd to dir

         fKeys = fFile->GetListOfKeys();
         fNextKey = new TIter(fKeys);
      }

      if ((next=(TKey*)(*fNextKey)()) ) {
         next->Print();
         if ( fClassName != next->GetClassName() )
            continue;

         delete fObj;
         fObj = next->ReadObj();
         selector->SetObject( fObj );

         return kTRUE;

      }

      // Done with this file
      delete fNextKey; fNextKey = 0;
      fKeys = 0;

   }

}


//------------------------------------------------------------------------

ClassImp(TEventIterSlave)


//______________________________________________________________________________
TEventIterSlave::TEventIterSlave()
{
   // Default ctor.

   fSocket = 0;
}

//______________________________________________________________________________
TEventIterSlave::TEventIterSlave(TSocket *socket)
{
   fSocket = socket;
}


//______________________________________________________________________________
TEventIterSlave::~TEventIterSlave()
{
}


//______________________________________________________________________________
Bool_t TEventIterSlave::Init(TSelector *selector)
{
   return kFALSE;
}


//______________________________________________________________________________
Bool_t TEventIterSlave::GetNextEvent(TSelector *selector)
{
   return kFALSE;
}
