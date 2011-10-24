// @(#)root/hist:$Id$
// Author: Axel Naumann (2007-09-11)

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
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

#ifndef ROOT_TArrayD
#include "TArrayD.h"
#endif

class TBrowser;
class TH1;
class THnSparse;

class THnSparseArrayChunk: public TObject {
 private:

   THnSparseArrayChunk(const THnSparseArrayChunk&); // Not implemented
   THnSparseArrayChunk& operator=(const THnSparseArrayChunk&); // Not implemented

 public:
   THnSparseArrayChunk():
      fCoordinateAllocationSize(-1), fSingleCoordinateSize(0), fCoordinatesSize(0), fCoordinates(0),
      fContent(0), fSumw2(0) {}

   THnSparseArrayChunk(Int_t coordsize, bool errors, TArray* cont);
   virtual ~THnSparseArrayChunk();

   Int_t    fCoordinateAllocationSize; //! size of the allocated coordinate buffer; -1 means none or fCoordinatesSize
   Int_t    fSingleCoordinateSize; // size of a single bin coordinate
   Int_t    fCoordinatesSize;      // size of the bin coordinate buffer
   Char_t  *fCoordinates;          //[fCoordinatesSize] compact bin coordinate buffer
   TArray  *fContent;              // bin content
   TArrayD *fSumw2;                // bin errors

   void AddBin(Int_t idx, const Char_t* idxbuf);
   void AddBinContent(Int_t idx, Double_t v = 1.) {
      fContent->SetAt(v + fContent->GetAt(idx), idx);
      if (fSumw2)
         fSumw2->SetAt(v * v+ fSumw2->GetAt(idx), idx);
   }
   void Sumw2();
   Int_t GetEntries() const { return fCoordinatesSize / fSingleCoordinateSize; }
   Bool_t Matches(Int_t idx, const Char_t* idxbuf) const {
      // Check whether bin at idx batches idxbuf.
      // If we don't store indexes we trust the caller that it does match,
      // see comment in THnSparseCompactBinCoord::GetHash().
      return fSingleCoordinateSize <= 8 ||
         !memcmp(fCoordinates + idx * fSingleCoordinateSize, idxbuf, fSingleCoordinateSize); }

   ClassDef(THnSparseArrayChunk, 1); // chunks of linearized bins
};

namespace ROOT {
   class THnSparseBrowsable: public TNamed {
   public:
      THnSparseBrowsable(THnSparse* hist, Int_t axis);
      ~THnSparseBrowsable();
      void Browse(TBrowser *b);
      Bool_t IsFolder() const { return kFALSE; }

   private:
      THnSparse* fHist; // Original histogram
      Int_t      fAxis; // Axis to visualize
      TH1*       fProj; // Projection result
      ClassDef(THnSparseBrowsable, 0); // Browser-helper for THnSparse
   };
};
#endif // ROOT_THnSparse_Internal

