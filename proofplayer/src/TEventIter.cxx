// @(#)root/proof:$Name:  $:$Id: TEventIter.cxx,v 1.2 2002/02/12 17:53:18 rdm Exp $
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
#include "TKey.h"
#include "TFile.h"
#include "TCollection.h"
#include "TError.h"
#include "TTree.h"
#include "TSelector.h"


//------------------------------------------------------------------------

ClassImp(TEventIter)


//______________________________________________________________________________
TEventIter::TEventIter()
{
   fDSet = 0;
   fDir = 0;
   fSel = 0;
   fNum = 0;
}


//______________________________________________________________________________
TEventIter::TEventIter(TDSet *dset, TDirectory *dir, TSelector *sel)
   : fDSet(dset), fDir(dir), fSel(sel)
{
   fNum = 0;
}


//______________________________________________________________________________
TEventIter::~TEventIter()
{
}


//______________________________________________________________________________
TEventIter *TEventIter::Create(TDSet *dset, TDirectory *dir, TSelector *sel)
{
   if ( dset->IsTree() ) {
      return new TEventIterTree(dset, dir, sel);
   } else {
      return new TEventIterObj(dset, dir, sel);
   }
}


//------------------------------------------------------------------------

ClassImp(TEventIterObj)


//______________________________________________________________________________
TEventIterObj::TEventIterObj()
{
   // Default ctor.

   fKeys     = 0;
   fNextKey  = 0;
   fObj      = 0;

}

//______________________________________________________________________________
TEventIterObj::TEventIterObj(TDSet *dset, TDirectory *dir, TSelector *sel)
   : TEventIter(dset,dir,sel)
{
   fClassName = dset->GetType();
   fKeys     = 0;
   fNextKey  = 0;
   fObj      = 0;
}


//______________________________________________________________________________
TEventIterObj::~TEventIterObj()
{
   // delete fKeys ?
   delete fNextKey;
   delete fObj;
}


//______________________________________________________________________________
Bool_t TEventIterObj::GetNextEvent()
{
   delete fObj; fObj = 0;

   if ( fNum > 0 ) {
         --fNum;
         ++fCur;
         TKey *key = (TKey*) fNextKey->Next();
         fObj = key->ReadObj();
         fSel->SetObject( fObj );
         return kTRUE;
   }

   return kFALSE;
}


//______________________________________________________________________________
Bool_t TEventIterObj::InitRange(Double_t first, Double_t num)
{
   // new file / directory?

   if ( fKeys == 0 ) {
      fKeys = fDir->GetListOfKeys();
      fNextKey = new TIter(fKeys);
   }

   fFirst = first;
   fNum = num;
   fCur = first-1;

   if ( fFirst >= fKeys->GetSize() ) {
      Error("TEventIterObj::InitRange","First larger the number of keys");
      return kFALSE;
   }

   if ( fFirst + fNum  > fKeys->GetSize() ) {
      Warning("TEventIterObj::InitRange","Num larger the number of keys");
      fNum = fKeys->GetSize() - fFirst;
   }

   // Position the iterator FIXME: should be more efficient?
   fNextKey->Reset();
   for( fCur = 0; fCur < fFirst ; fCur++, fNextKey->Next() );

   return kTRUE;
}


//------------------------------------------------------------------------

ClassImp(TEventIterTree)


//______________________________________________________________________________
TEventIterTree::TEventIterTree()
{
   // Default ctor.

   fTree = 0;
   fNum = 999999999; // TODO: proper max event
   fFirst = 0;
   fCur = -1;
}

//______________________________________________________________________________
TEventIterTree::TEventIterTree(TDSet *dset, TDirectory *dir, TSelector *sel)
   : TEventIter(dset,dir,sel)
{
   fTreeName = dset->GetObjName();
   fTree = 0;
   fNum = 999999999; // TODO: proper max event
   fFirst = 0;
   fCur = -1;
}


//______________________________________________________________________________
TEventIterTree::~TEventIterTree()
{
   // delete fTree ?
}


//______________________________________________________________________________
Bool_t TEventIterTree::GetNextEvent()
{
   if ( fNum > 0 ) {
         --fNum;
         ++fCur;
         return kTRUE;
   }
   return kFALSE;
}


//______________________________________________________________________________
Bool_t TEventIterTree::InitRange(Double_t first, Double_t num)
{
   // New Tree?

   if ( fTree == 0 ) {

      TKey *key;
      if ( (key = fDir->GetKey(fTreeName)) == 0 ) {
         Error("InitRange","Cannot find tree \"%s\"",
               fTreeName.Data());
         return kFALSE;
      }
      Info("TEventIterTree::InitRange","Reading: %s", fTreeName.Data() );
      fTree = (TTree *) key->ReadObj(); // TODO: check result and type?
      fSel->Notify( /* fTree */ );  // TODO: change API
   }

   // TODO: add checks for first and num vs. the tree
   fFirst = first;
   fNum = num;
   fCur = first-1;

   return kTRUE;
}

