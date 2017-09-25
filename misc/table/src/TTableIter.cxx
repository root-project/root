// @(#)root/table:$Id$
// Author: Valery Fine(fine@bnl.gov)   03/12/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TTableIter - class iterator to loop over sorted TTable's             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TTableIter.h"
#include "TTableSorter.h"

ClassImp(TTableIter);

////////////////////////////////////////////////////////////////////////////////
///to be documented

TTableIter::TTableIter(const TTableSorter *table, Float_t &keyvalue)
           : fTableSorter(table), fIndx(0), fFirstIndx(0)
{
   CountKey(keyvalue);
}

////////////////////////////////////////////////////////////////////////////////
///to be documented

TTableIter::TTableIter(const TTableSorter *table, Long_t &keyvalue)
           : fTableSorter(table), fIndx(0), fFirstIndx(0)
{
   CountKey(keyvalue);
}

////////////////////////////////////////////////////////////////////////////////
///to be documented

TTableIter::TTableIter(const TTableSorter *table, Int_t &keyvalue)
           : fTableSorter(table), fIndx(0), fFirstIndx(0)
{
   CountKey(keyvalue);
}

////////////////////////////////////////////////////////////////////////////////
///to be documented

TTableIter::TTableIter(const TTableSorter *table, Short_t &keyvalue)
           : fTableSorter(table), fIndx(0), fFirstIndx(0)
{
   CountKey(keyvalue);
}

////////////////////////////////////////////////////////////////////////////////
///to be documented

TTableIter::TTableIter(const TTableSorter *table, Double_t &keyvalue)
           : fTableSorter(table), fIndx(0), fFirstIndx(0)
{
   CountKey(keyvalue);
}

////////////////////////////////////////////////////////////////////////////////
///to be documented

Int_t TTableIter::CountKey(Float_t &keyvalue)
{
   fTotalKeys = fTableSorter->CountKey(&keyvalue,0,kTRUE,&fFirstIndx);
   return GetNRows();
}
////////////////////////////////////////////////////////////////////////////////
///to be documented

Int_t TTableIter::CountKey(Long_t &keyvalue)
{
   fTotalKeys = fTableSorter->CountKey(&keyvalue,0,kTRUE,&fFirstIndx);
   return GetNRows();
}

////////////////////////////////////////////////////////////////////////////////
///to be documented

Int_t TTableIter::CountKey(Int_t &keyvalue)
{
   fTotalKeys = fTableSorter->CountKey(&keyvalue,0,kTRUE,&fFirstIndx);
   return GetNRows();
}

////////////////////////////////////////////////////////////////////////////////
///to be documented

Int_t TTableIter::CountKey(Short_t &keyvalue)
{
   fTotalKeys = fTableSorter->CountKey(&keyvalue,0,kTRUE,&fFirstIndx);
   return GetNRows();
}

////////////////////////////////////////////////////////////////////////////////
///to be documented

Int_t TTableIter::CountKey(Double_t &keyvalue)
{
   fTotalKeys = fTableSorter->CountKey(&keyvalue,0,kTRUE,&fFirstIndx);
   return GetNRows();
}

////////////////////////////////////////////////////////////////////////////////
///to be documented

Int_t TTableIter::Next()
{
   Int_t rowIndx = -1;
   if (fIndx < fTotalKeys) {
      rowIndx = fTableSorter->GetIndex(UInt_t(fFirstIndx+fIndx));
      fIndx++;
   }
   return rowIndx;
}

////////////////////////////////////////////////////////////////////////////////
///to be documented

Int_t TTableIter::Next(Int_t idx)
{
   Int_t rowIndx = -1;
   if (idx < fTotalKeys)
      rowIndx = fTableSorter->GetIndex(UInt_t(fFirstIndx+idx));
   return rowIndx;
}

////////////////////////////////////////////////////////////////////////////////
///to be documented

Int_t TTableIter::Reset(Int_t indx)
{
   Int_t oldIdx = fIndx;
   fIndx = TMath::Min(indx,fTotalKeys);
   return oldIdx;
}
