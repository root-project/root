// @(#)root/proof:$Name:  $:$Id: TProof.cxx,v 1.78 2005/03/08 09:19:18 rdm Exp $
// Author: Marek Biskup   28/01/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TProofNTuple
#define ROOT_TProofNTuple

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TProofNTuple                                                         //
//                                                                      //
// This class represents a set of points of a dimension specified in    //
// the constructor the points are counted from 0 and the coordinates    //
// from 1, see TProofNTuple::Get().                                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TNamed.h"
#endif

#ifndef ROOT_TArrayD
#include "TArrayD.h"
#endif


class TProofNTuple : public TNamed {

protected:
   Int_t         fDimension;                 // dimension of the NTuple
   TArrayD       fArray;                     // an array of stored values
   Int_t         fEntries;                   // number of rows

   Bool_t        AssertSize(int newMinSize);

public:
   TProofNTuple(Int_t dimension = 1);
   ~TProofNTuple();
   virtual Bool_t      Fill(Double_t x);
   virtual Bool_t      Fill(Double_t x, Double_t y);
   virtual Bool_t      Fill(Double_t x, Double_t y, Double_t z);
   virtual Bool_t      Fill(Double_t x, Double_t y, Double_t z, Double_t t);

   virtual Double_t    GetX(Int_t entry) {
      // returns the 1st coordinate of the entry *entry*. Assumes that the dimension >= 1
      // in case of an error returns 0
       return Get(entry, 1);
   }
   virtual Double_t    GetY(Int_t entry) {
      // returns the 2nd coordinate of the entry *entry*. Assumes that the dimension >= 2
      // in case of an error returns 0
      return Get(entry, 2);
   }
   virtual Double_t    GetZ(Int_t entry) {
      // returns the 3nd coordinate of the entry *entry*. Assumes that the dimension >= 3
      // in case of an error returns 0
      return Get(entry, 3);
   }
   virtual Double_t    GetT(Int_t entry) {
      // returns the 4nd coordinate of the entry *entry*. Assumes that the dimension >= 4
      // in case of an error returns 0
      return Get(entry, 4);
   }
   virtual Double_t    Get(Int_t entry, Int_t coord);

   virtual void        DrawCopy(const Option_t* option);
   virtual Int_t       Merge(TCollection* list);
   virtual Bool_t      Add(TProofNTuple* ntuple);

   virtual Int_t GetDimension() {
      // Returns the dimension
      return fDimension;
   }
   virtual Int_t GetEntries() {
      // returns the number of entries
      return fEntries;
   }
   virtual Double_t Min(int coord);
   virtual Double_t Max(int coord);

   ClassDef(TProofNTuple,1)
};

#endif

