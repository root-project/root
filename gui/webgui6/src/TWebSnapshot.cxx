/// Author:  Sergey Linev, GSI  6/04/2017

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TWebSnapshot.h"

#include "TString.h"

///////////////////////////////////////////////////////////////////////////////////////////
/// destructor

TWebSnapshot::~TWebSnapshot()
{
   SetSnapshot(kNone, nullptr);
}

///////////////////////////////////////////////////////////////////////////////////////////
/// SetUse pointer to assign object id - TString::Hash

void TWebSnapshot::SetSnapshot(Int_t kind, TObject *shot, Bool_t owner)
{
   if (fSnapshot && fOwner) delete fSnapshot;
   fKind = kind;
   fSnapshot = shot;
   fOwner = owner;
}

///////////////////////////////////////////////////////////////////////////////////////////
/// Use pointer to assign object id - TString::Hash

void TWebSnapshot::SetObjectIDAsPtr(void *ptr)
{
   UInt_t hash = TString::Hash(&ptr, sizeof(ptr));
   SetObjectID(TString::UItoa(hash,10));
}


///////////////////////////////////////////////////////////////////////////////////////////
/// destructor

TPadWebSnapshot::~TPadWebSnapshot()
{
   for (auto &&item: fPrimitives)
      delete item;
   fPrimitives.clear();
}

