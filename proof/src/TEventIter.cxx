// @(#)root/proof:$Name:  $:$Id: TEventIter.cxx,v 1.1 2002/01/15 00:45:20 rdm Exp $
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


ClassImp(TEventIter)

//______________________________________________________________________________
TEventIter::TEventIter()
{
   // Default ctor.

   fIsTree = kFALSE;
   fFile   = 0;
   fKeys   = 0;
   fNext   = 0;
   fObj    = 0;
}

//______________________________________________________________________________
TEventIter::TEventIter(TDSet *set)
{
   // dummy setup

   fIsTree = kTRUE;

   fFile = new TFile("dummies.root");
   fKeys = fFile->GetListOfKeys();
   fNext = new TIter(fKeys);
   fClassName = TString("MyDummy");

   fObj = 0;
}

//______________________________________________________________________________
Bool_t TEventIter::GetNextEvent(TSelector *selector)
{
   TKey  *next;

   while ( (next=(TKey*)(*fNext)()) ) {
      next->Print();
      if ( fClassName != next->GetClassName() )
         continue;

      delete fObj;
      fObj = next->ReadObj();
      selector->SetObject( fObj );

      return kTRUE;
   }

   return kFALSE;
}

