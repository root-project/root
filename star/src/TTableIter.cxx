// @(#)root/star:$Name$:$Id$
// Author: Valery Fine(fine@bnl.gov)   03/12/99
// Copyright(c) 1997~1999  [BNL] Brookhaven National Laboratory, STAR, All rights reserved
// Author                  Valerie Fine  (fine@bnl.gov)
// Copyright(c) 1997~1999  Valerie Fine  (fine@bnl.gov)
// $Id: TTableIter.cxx,v 1.4 1999/12/29 18:43:03 fine Exp $
//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TTableIter - class iterator to loop over sorted TTable's             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TTableIter.h"
#include "TTableSorter.h"

ClassImp(TTableIter)

//_____________________________________________________________________
TTableIter::TTableIter(const TTableSorter *table, Float_t &keyvalue)
           : fTableSorter(table), fIndx(0), fFirstIndx(0)
{
  CountKey(keyvalue);
}

//_____________________________________________________________________
TTableIter::TTableIter(const TTableSorter *table, Long_t &keyvalue)
           : fTableSorter(table), fIndx(0), fFirstIndx(0)
{
  CountKey(keyvalue);
}

//_____________________________________________________________________
TTableIter::TTableIter(const TTableSorter *table, Int_t &keyvalue)
           : fTableSorter(table), fIndx(0), fFirstIndx(0)
{
  CountKey(keyvalue);
}

//_____________________________________________________________________
TTableIter::TTableIter(const TTableSorter *table, Short_t &keyvalue)
           : fTableSorter(table), fIndx(0), fFirstIndx(0)
{
  CountKey(keyvalue);
}

//_____________________________________________________________________
TTableIter::TTableIter(const TTableSorter *table, Double_t &keyvalue)
           : fTableSorter(table), fIndx(0), fFirstIndx(0)
{
  CountKey(keyvalue);
}

//_____________________________________________________________________
Int_t TTableIter::CountKey(Float_t &keyvalue)
{
   fTotalKeys = fTableSorter->CountKey(&keyvalue,0,kTRUE,&fFirstIndx);
   return GetNRows();
}
//_____________________________________________________________________
Int_t TTableIter::CountKey(Long_t &keyvalue)
{
   fTotalKeys = fTableSorter->CountKey(&keyvalue,0,kTRUE,&fFirstIndx);
   return GetNRows();
}

//_____________________________________________________________________
Int_t TTableIter::CountKey(Int_t &keyvalue)
{
   fTotalKeys = fTableSorter->CountKey(&keyvalue,0,kTRUE,&fFirstIndx);
   return GetNRows();
}

//_____________________________________________________________________
Int_t TTableIter::CountKey(Short_t &keyvalue)
{
   fTotalKeys = fTableSorter->CountKey(&keyvalue,0,kTRUE,&fFirstIndx);
   return GetNRows();
}

//_____________________________________________________________________
Int_t TTableIter::CountKey(Double_t &keyvalue)
{
   fTotalKeys = fTableSorter->CountKey(&keyvalue,0,kTRUE,&fFirstIndx);
   return GetNRows();
}

//_____________________________________________________________________
Int_t TTableIter::Next()
{
   Int_t rowIndx = -1;
   if (fIndx < fTotalKeys) {
     rowIndx = fTableSorter->GetIndex(UInt_t(fFirstIndx+fIndx));
     fIndx++;
   }
   return rowIndx;
}

//_____________________________________________________________________
Int_t TTableIter::Next(Int_t idx)
{
   Int_t rowIndx = -1;
   if (idx < fTotalKeys)
     rowIndx = fTableSorter->GetIndex(UInt_t(fFirstIndx+idx));
   return rowIndx;
}

//_____________________________________________________________________
Int_t TTableIter::Reset(Int_t indx)
{
   Int_t oldIdx = fIndx;
   fIndx = TMath::Min(indx,fTotalKeys);
   return oldIdx;
}
