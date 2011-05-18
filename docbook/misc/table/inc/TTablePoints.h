// @(#)root/table:$Id$
// Author: Valery Fine   14/05/99  (E-mail: fine@bnl.gov)

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TTablePoints
#define ROOT_TTablePoints

#include "TPoints3DABC.h"
#include "TTableSorter.h"
#include "TTable.h"

class TTablePoints : public TPoints3DABC
{
protected:
   TTableSorter   *fTableSorter;
   const void     *fKey;            // pointer to key value to select rows
   Int_t           fFirstRow;       // The first row to take in account
   Int_t           fSize;
   void           *fRows;           // Pointer the first row of the STAF table

   virtual void SetTablePointer(void *table);
   TTablePoints();
public:
   TTablePoints(TTableSorter *sorter,const void *key,Option_t *opt="");
   TTablePoints(TTableSorter *sorter, Int_t keyIndex,Option_t *opt="");
   ~TTablePoints(){}
   virtual Int_t     DistancetoPrimitive(Int_t px, Int_t py);
   virtual Int_t     GetLastPosition()const;
   virtual Float_t   GetX(Int_t idx)  const = 0;
   virtual Float_t   GetY(Int_t idx)  const = 0;
   virtual Float_t   GetZ(Int_t idx)  const = 0;
   virtual void     *GetTable();
   virtual Option_t *GetOption()      const { return 0;}
   virtual Int_t     Indx(Int_t sortedIndx) const;
   virtual Int_t     SetLastPosition(Int_t idx);
   virtual void      SetOption(Option_t *){;}
   virtual Int_t     SetPoint(Int_t, Float_t, Float_t, Float_t ){return -1;}
   virtual Int_t     SetPoints(Int_t , Float_t *, Option_t *){return -1;}
   virtual Int_t     Size() const;
   ClassDef(TTablePoints,0)  // Defines the TTable as an element of "event" geometry
};

//____________________________________________________________________________
inline void TTablePoints::SetTablePointer(void *table){ fRows = table;}

//____________________________________________________________________________
// return the index of the origial row by its index from the sorted table
inline Int_t TTablePoints::Indx(Int_t sortedIndx) const
{return fTableSorter?fTableSorter->GetIndex(fFirstRow+sortedIndx):-1;}
//____________________________________________________________________________
// return the pointer to the original table object
inline void *TTablePoints::GetTable(){
   void *ret = 0;
   if (fTableSorter) {
      TTable *t = fTableSorter->GetTable();
      if (t) ret = t->GetArray();
   }
   return ret;
}
//____________________________________________________________________________
inline Int_t TTablePoints::Size() const { return fSize;}
//____________________________________________________________________________
inline Int_t TTablePoints::GetLastPosition() const {return Size()-1;}

//____________________________________________________________________________
inline Int_t TTablePoints::SetLastPosition(Int_t idx)
{
   Int_t pos = GetLastPosition();
   fSize = TMath::Min(pos,idx)+1;
   return pos;
}

#endif

