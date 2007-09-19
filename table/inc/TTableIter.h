// @(#)root/table:$Id$
// Author: Valery Fine(fine@bnl.gov)   03/12/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TTableIter
#define ROOT_TTableIter

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TTableIter - class iterator to loop over sorted TTable's             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TObject.h"

class TTableSorter;

class TTableIter : public TObject {
private:
   const TTableSorter *fTableSorter;
   Int_t  fIndx;
   Int_t  fTotalKeys;
   Int_t  fFirstIndx;
protected:
   TTableIter(){;}
   TTableIter(const TTableIter &org) : TObject(org) {;}

public:
   TTableIter(const TTableSorter *table, Float_t  &keyvalue);
   TTableIter(const TTableSorter *table, Double_t &keyvalue);
   TTableIter(const TTableSorter *table, Int_t    &keyvalue);
   TTableIter(const TTableSorter *table, Long_t   &keyvalue);
   TTableIter(const TTableSorter *table, Short_t  &keyvalue);

   virtual ~TTableIter(){;}

   Int_t CountKey(Float_t &keyvalue);
   Int_t CountKey(Long_t &keyvalue);
   Int_t CountKey(Int_t &keyvalue);
   Int_t CountKey(Short_t &keyvalue);
   Int_t CountKey(Double_t &keyvalue);

   Int_t GetNRows() const;
   Int_t Next();
   Int_t Next(Int_t idx);
   Int_t Reset(Int_t indx=0);
   Int_t operator()();
   Int_t operator[](Int_t idx);
   ClassDef(TTableIter,0) // Iterator over "sorted" TTable objects
};

//_____________________________________________________________________
inline Int_t TTableIter::GetNRows() const { return fTotalKeys;}
//_____________________________________________________________________
inline Int_t TTableIter::operator()() { return Next();}

//_____________________________________________________________________
inline Int_t TTableIter::operator[](Int_t idx) { return Next(idx);}

#endif
