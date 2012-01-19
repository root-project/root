// @(#)root/hist:$Id$
// Author: Axel Naumann (2011-12-13)

/*************************************************************************
 * Copyright (C) 1995-2012, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "THn.h"

#include "TClass.h"

namespace {
   struct CounterRange_t {
      Int_t i;
      Int_t first;
      Int_t last;
      Int_t len;
      Long64_t cellSize;
   };

   class THnBinIter: public ROOT::THnBaseBinIter {
   public:
      THnBinIter(Int_t dim, const TObjArray* axes, const TNDArray* arr,
                 Bool_t respectAxisRange);
      ~THnBinIter() { delete [] fCounter; }

      Long64_t Next(Int_t* coord = 0);
      Int_t GetCoord(Int_t dim) const { return fCounter[dim].i; }
   private:
      THnBinIter(const THnBinIter&); // intentionally unimplemented
      THnBinIter& operator=(const THnBinIter&); // intentionally unimplemented

   public:
      Int_t fNdimensions;
      Long64_t fIndex;
      const TNDArray* fArray;
      CounterRange_t* fCounter;
   };


   //______________________________________________________________________________
   THnBinIter::THnBinIter(Int_t dim, const TObjArray* axes,
                              const TNDArray* arr, Bool_t respectAxisRange):
      ROOT::THnBaseBinIter(respectAxisRange),
      fNdimensions(dim), fIndex(-1), fArray(arr) {
      fCounter = new CounterRange_t[dim]();
      for (Int_t i = 0; i < dim; ++i) {
         TAxis *axis = (TAxis*) axes->At(i);
         fCounter[i].len  = axis->GetNbins() + 2;
         fCounter[i].cellSize  = arr->GetCellSize(i);
         if (!respectAxisRange || !axis->TestBit(TAxis::kAxisRange)) {
            fCounter[i].first = 0;
            fCounter[i].last  = fCounter[i].len - 1;
            fCounter[i].i     = 0;
            continue;
         }
         fHaveSkippedBin = kTRUE;
         Int_t min = axis->GetFirst();
         Int_t max = axis->GetLast();
         if (min == 0 && max == 0) {
            // special case where TAxis::SetBit(kAxisRange) and
            // over- and underflow bins are de-selected.
            // first and last are == 0 due to axis12->SetRange(1, axis12->GetNbins());
            min = 1;
            max = axis->GetNbins();
         }
         fCounter[i].first = min;
         fCounter[i].last  = max;
         fCounter[i].i     = min;
         fIndex += fCounter[i].first * fCounter[i].cellSize;
      }
      // First Next() will increment it:
      --fCounter[dim - 1].i;
   }

   //______________________________________________________________________________
   Long64_t THnBinIter::Next(Int_t* coord /*= 0*/) {
      if (fNdimensions < 0) return -1; // end
      ++fCounter[fNdimensions - 1].i;
      ++fIndex;
      // Wrap around if needed
      for (Int_t d = fNdimensions - 1; d > 0 && fCounter[d].i > fCounter[d].last; --d) {
         // We skip last + 1..size and 0..first - 1, adjust fIndex
         Int_t skippedCells = fCounter[d].len - (fCounter[d].last + 1);
         skippedCells += fCounter[d].first;
         fIndex += skippedCells * fCounter[d].cellSize;
         fCounter[d].i = fCounter[d].first;
         ++fCounter[d - 1].i;
      }
      if (fCounter[0].i > fCounter[0].last) {
         fNdimensions = -1;
         return -1;
      }
      if (coord) {
         for (Int_t d = 0; d < fNdimensions; ++d) {
            coord[d] = fCounter[d].i;
         }
      }
      return fIndex;
   }
} // unnamed namespce



//______________________________________________________________________________
THn::THn(const char* name, const char* title,
         Int_t dim, const Int_t* nbins,
         const Double_t* xmin, const Double_t* xmax):
   THnBase(name, title, dim, nbins, xmin, xmax),
   fSumw2(dim, nbins, kTRUE /*overflow*/) {
   // Construct a THn.
   fCoordBuf = new Int_t[dim];
}

//______________________________________________________________________________
THn::~THn()
{
   // Destruct a THn
   delete [] fCoordBuf;
}


//______________________________________________________________________________
ROOT::THnBaseBinIter* THn::CreateIter(Bool_t respectAxisRange) const
{
   // Create an iterator over all bins. Public interface is THnIter.
   return new THnBinIter(GetNdimensions(), GetListOfAxes(), &GetArray(),
                         respectAxisRange);
}

//______________________________________________________________________________
void THn::Sumw2() {
   // Enable calculation of errors
   if (!GetCalculateErrors()) {
      fTsumw2 = 0.;
   }
}

 
//______________________________________________________________________________
THnBase* THn::CloneEmpty(const char* name, const char* title,
                         const TObjArray* axes, Bool_t keepTargetAxis) const
{
   // Create a new THn object that is of the same type as *this,
   // but with dimensions and bins given by axes.
   // If keepTargetAxis is true, the axes will keep their original xmin / xmax,
   // else they will be restricted to the range selected (first / last).

   THn* ret = (THn*)IsA()->New();
   ret->SetNameTitle(name, title);

   TIter iAxis(axes);
   const TAxis* axis = 0;
   Int_t pos = 0;
   Int_t *nbins = new Int_t[axes->GetEntriesFast()];
   while ((axis = (TAxis*)iAxis())) {
      TAxis* reqaxis = (TAxis*)axis->Clone();
      if (!keepTargetAxis && axis->TestBit(TAxis::kAxisRange)) {
         Int_t binFirst = axis->GetFirst();
         Int_t binLast = axis->GetLast();
         Int_t nBins = binLast - binFirst + 1;
         if (axis->GetXbins()->GetSize()) {
            // non-uniform bins:
            reqaxis->Set(nBins, axis->GetXbins()->GetArray() + binFirst - 1);
         } else {
            // uniform bins:
            reqaxis->Set(nBins, axis->GetBinLowEdge(binFirst), axis->GetBinUpEdge(binLast));
         }
         reqaxis->ResetBit(TAxis::kAxisRange);
      }

      nbins[pos] = reqaxis->GetNbins();
      ret->fAxes.AddAtAndExpand(reqaxis->Clone(), pos++);
   }
   ret->fAxes.SetOwner();

   ret->fNdimensions = axes->GetEntriesFast();
   ret->fCoordBuf = new Int_t[ret->fNdimensions];
   ret->GetArray().Init(ret->fNdimensions, nbins, true /*addOverflow*/);

   delete [] nbins;

   return ret;
   
}

//______________________________________________________________________________
void THn::Reset(Option_t* option /*= ""*/)
{
   // Reset the contents of a THn.
   GetArray().Reset(option);
   fSumw2.Reset(option);
}
