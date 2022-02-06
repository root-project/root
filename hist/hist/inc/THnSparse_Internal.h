// @(#)root/hist:$Id$
// Author: Axel Naumann (2007-09-11)

/*************************************************************************
 * Copyright (C) 1995-2012, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_THnSparse_Internal
#define ROOT_THnSparse_Internal

/*************************************************************************
 * Non-API classes for THnSparse.                                        *
 * I.e. interesting to look at if you want to know how it works, but     *
 * don't use directly.                                                   *
 * Implementation in THnSparse.cxx.                                      *
 *************************************************************************/

#include "TArrayD.h"

#include "TObject.h"

class TBrowser;
class TH1;
class THnSparse;

class THnSparseArrayChunk: public TObject {
 private:

   THnSparseArrayChunk(const THnSparseArrayChunk&) = delete;
   THnSparseArrayChunk& operator=(const THnSparseArrayChunk&) = delete;

 public:
   THnSparseArrayChunk():
      fCoordinateAllocationSize(-1), fSingleCoordinateSize(0), fCoordinatesSize(0), fCoordinates(0),
      fContent(0), fSumw2(0) {}

   THnSparseArrayChunk(Int_t coordsize, bool errors, TArray* cont);
   ~THnSparseArrayChunk() override;

   Int_t    fCoordinateAllocationSize; ///<! Size of the allocated coordinate buffer; -1 means none or fCoordinatesSize
   Int_t    fSingleCoordinateSize;     ///<  Size of a single bin coordinate
   Int_t    fCoordinatesSize;          ///<  Size of the bin coordinate buffer
   Char_t  *fCoordinates;              ///<[fCoordinatesSize] compact bin coordinate buffer
   TArray  *fContent;                  ///<  Bin content
   TArrayD *fSumw2;                    ///<  Bin errors

   void AddBin(Int_t idx, const Char_t* idxbuf);
   void AddBinContent(Int_t idx, Double_t v = 1.) {
      fContent->SetAt(v + fContent->GetAt(idx), idx);
      if (fSumw2)
         fSumw2->SetAt(v * v+ fSumw2->GetAt(idx), idx);
   }
   void Sumw2();
   Int_t GetEntries() const { return fCoordinatesSize / fSingleCoordinateSize; }

   /// Check whether bin at idx batches idxbuf.
   /// If we don't store indexes we trust the caller that it does match,
   /// see comment in THnSparseCompactBinCoord::GetHash().
   Bool_t Matches(Int_t idx, const Char_t* idxbuf) const {
      return fSingleCoordinateSize <= 8 ||
         !memcmp(fCoordinates + idx * fSingleCoordinateSize, idxbuf, fSingleCoordinateSize); }

   ClassDefOverride(THnSparseArrayChunk, 1); // chunks of linearized bins
};
#endif // ROOT_THnSparse_Internal

