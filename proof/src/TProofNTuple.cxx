// @(#)root/proof:$Name:  $:$Id: TProofNTuple.cxx,v 1.1 2005/03/10 17:57:04 rdm Exp $
// Author: Marek Biskup   28/01/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TProofNTuple                                                         //
//                                                                      //
// This class represents a set of points of a dimension specified in    //
// the constructor the points are counted from 0 and the coordinates    //
// from 1, see TProofNTuple::Get().                                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TProofNTuple.h"
#include "TGraph.h"
#include "TVirtualPad.h"
#include "TGraph2D.h"
#include "TPolyMarker3D.h"


ClassImp(TProofNTuple)

//______________________________________________________________________________
TProofNTuple::TProofNTuple(Int_t dimension) : fDimension(dimension),
   fArray(10*dimension),  fEntries(0)
{
   // Creates a new TProofNTuple of dimension *dimension*.
   // dimension - dimension of the new TProofNTuple.
}


//______________________________________________________________________________
TProofNTuple::~TProofNTuple()
{
   // destructor
}


//______________________________________________________________________________
Double_t TProofNTuple::Get(Int_t entry, Int_t coord)
{
   // The *coord*-th coordinate of the entry *entry*.
   // entry - entry number (counted from 0)
   // coord - coordinate of the point. Must be between 1 and the
   //         dimension of the plot
   // Returns apropriate value or 0 in case of an error (wrong parameters).

   if (coord < 1 || coord > fDimension) {
      Error("Get", "Wrong coord: %d", coord);
      return 0;
   }
   if (entry < 0 || entry >= fEntries) {
      Error("Get", "Trying to access coordinate %d while the dimension is %d",
            coord, entry);
      return 0;
   }
   coord--;
   return fArray[entry*fDimension + coord];
}

//______________________________________________________________________________
Bool_t TProofNTuple::AssertSize(Int_t newMinSize)
{
   // Enlarges (if necessary) the array where the poins are kept to the size
   // of minimum *newMinSize.
   // newMinSize - new minimum size of the
   // Returns kTRUE if success, kFALSE in case of an error.

   if (fArray.GetSize() < newMinSize) {
      Int_t size = fArray.GetSize();
      while (size < newMinSize)
         size *= 2;
      fArray.Set(size);
      return kTRUE;
   }
   else
      return kTRUE;
}

//______________________________________________________________________________
Bool_t TProofNTuple::Fill(Double_t x)
{
   // Adds a new point, assumes that the dimension equals 1.
   // x - 1st coordinate of the poind
   // Returns kTRUE if success, kFALSE if failed.

   if (fDimension != 1) {
      Error("Fill(x)", "the dimension of this array is %d", fDimension);
      return 0;
   }
   if (!AssertSize((fEntries+1)*fDimension))
      return kFALSE;
   fArray[fEntries++] = x;
   return kTRUE;
}


//______________________________________________________________________________
Bool_t TProofNTuple::Fill(Double_t x, Double_t y)
{
   // Adds a new point, assumes that the dimension equals 2.
   // x - 1st coordinate of the point
   // y - 2nd coordinate of the point
   // Returns kTRUE if success, kFALSE if failed.

   if (fDimension != 2) {
      Error("Fill(x, y)", "the dimension of this array is %d", fDimension);
      return 0;
   }
   if (!AssertSize((fEntries+1)*fDimension))
      return kFALSE;
   fArray[2*fEntries] = x;
   fArray[2*fEntries+1] = y;
   fEntries++;

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TProofNTuple::Fill(Double_t x, Double_t y, Double_t z)
{
   // Adds a new point, assumes that the dimension equals 3.
   // x - 1st coordinate of the point
   // y - 2nd coordinate of the point
   // z - 3rd coordinate of the point
   // Returns kTRUE if success, kFALSE if failed.

   if (fDimension != 3) {
      Error("Fill(x, y, z)", "the dimension of this array is %d", fDimension);
      return 0;
   }
   if (!AssertSize((fEntries+1)*fDimension))
      return kFALSE;
   fArray[3*fEntries] = x;
   fArray[3*fEntries+1] = y;
   fArray[3*fEntries+2] = z;
   fEntries++;

   return kTRUE;
}


//______________________________________________________________________________
Bool_t TProofNTuple::Fill(Double_t x, Double_t y, Double_t z, Double_t t)
{
   // Adds a new point, assumes that the dimension equals 4.
   // x - 1st coordinate of the point
   // y - 2nd coordinate of the point
   // z - 3rd coordinate of the point
   // z - 4th coordinate of the point
   // Returns kTRUE if success, kFALSE if failed.

   if (fDimension != 4) {
      Error("Fill(x, y, z, t)", "The dimension of this array is %d", fDimension);
      return 0;
   }
   if (!AssertSize((fEntries+1)*fDimension))
      return kFALSE;
   fArray[4*fEntries] = x;
   fArray[4*fEntries+1] = y;
   fArray[4*fEntries+2] = z;
   fArray[4*fEntries+3] = t;
   fEntries++;

   return kTRUE;
}


//______________________________________________________________________________
Bool_t TProofNTuple::Add(TProofNTuple* ntuple)
{
   // Adds another ntuple.
   // ntuple - another ntuple plot. It will be added to this one.
   // failes if the dimension are different
   // Returns kTRUE if success, kFALSE if failure.

   if (fDimension != ntuple->GetDimension()) {
      Error("Add", "Wrond dimensions");
      return kFALSE;
   }
   Int_t to = (fEntries + ntuple->GetEntries())*fDimension;
   AssertSize(to);

   for (int i = fEntries*fDimension, j = 0; i < to; i++, j++)
      fArray[i] = ntuple->fArray[j];
   fEntries = fEntries + ntuple->GetEntries();
   return kTRUE;
}

//______________________________________________________________________________
Int_t TProofNTuple::Merge(TCollection* list)
{
   // Adds all thei TNtuples in the collection to this one.
   // fails if there is an object on the list which is not a TProofNTuple
   // or if the dimension of that object != dimension of this one
   // if the function fails the state of this object is unspecified.
   // After the successfull function call the NTuple contains
   // all the points from every ntuple on the list and from
   // its previous state.
   // list - a collection of objects that are to be added to this one.
   // Returns the merged number of entries if the merge is
   // successfull, -1 otherwise.

   int totalEntries = fEntries;
   TIter next(list);
   while (TObject * o = next()) {
      TProofNTuple *p = dynamic_cast<TProofNTuple*> (o);
      if (!p)
         return -1;
      if (fDimension != p->GetDimension())
         return kFALSE;
      totalEntries += p->GetEntries();
   }
   next.Reset();
   AssertSize(totalEntries*fDimension);

   while (TObject * o = next()) {
      TProofNTuple *p = dynamic_cast<TProofNTuple*> (o);
      if (!p)
         return -1;
      if (!Add(p))
         return -1;
   }
   return totalEntries;
}

//______________________________________________________________________________
Double_t TProofNTuple::Min(int coord)
{
   // Returns minimum value for the specified coordinate.
   // coord - the number of the coordinate for which we want the minimum. Must be between
   // 1 and this->GetDimension()
   // Returns minimum of 0 in case of w wrong parameter of empty TProofNTuple.

   if (coord < 1 || coord > GetDimension()) {
      Error("Min", "Wrong parameter %d, the dimension of the TProofNTuple is %d",
         coord, GetDimension());
      return 0.0;
   }
   if (fEntries == 0)
      return 0.0;
   coord--;
   Double_t m = fArray[coord];
   for (int entry = 1; entry < fEntries; entry++)
      if (fArray[entry*fDimension + coord] < m)
         m = fArray[entry*fDimension + coord];
   return m;
}

//______________________________________________________________________________
Double_t TProofNTuple::Max(int coord)
{
   // Returns maximum value for the specified coordinate.
   // coord - the number of the coordinate for which we want the maximum. Must be between
   // 1 and this->GetDimension()
   // Returns minimum of 0 in case of w wrong parameter of empty TProofNTuple.

   if (coord < 1 || coord > GetDimension()) {
      Error("Min", "Wrong parameter %d, the dimension of the TProofNTuple is %d",
         coord, GetDimension());
      return 0.0;
   }
   if (fEntries == 0)
      return 0.0;
   coord--;
   Double_t m = fArray[coord];
   for (int entry = 1; entry < fEntries; entry++)
      if (fArray[entry*fDimension + coord] > m)
         m = fArray[entry*fDimension + coord];
   return m;
}
