// @(#)root/hist:$Name:  $:$Id: THnSparse.h,v 1.2 2007/09/13 18:24:10 brun Exp $
// Author: Axel Naumann (2007-09-11)

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_THnSparse
#define ROOT_THnSparse

/*************************************************************************

 THnSparse: histogramming multi-dimensional, sparse distributions in 
 a memory-efficient way.

*************************************************************************/


#ifndef ROOT_TExMap
#include "TExMap.h"
#endif
#ifndef ROOT_TNamed
#include "TNamed.h"
#endif
#ifndef ROOT_TObjArray
#include "TObjArray.h"
#endif
#ifndef ROOT_TArrayD
#include "TArrayD.h"
#endif

// needed only for template instantiations of THnSparseT:
#ifndef ROOT_TArrayF
#include "TArrayF.h"
#endif
#ifndef ROOT_TArrayL
#include "TArrayL.h"
#endif
#ifndef ROOT_TArrayI
#include "TArrayI.h"
#endif
#ifndef ROOT_TArrayS
#include "TArrayS.h"
#endif
#ifndef ROOT_TArrayC
#include "TArrayC.h"
#endif

class TAxis;
class TH1D;
class TH2D;
class TH3D;

class THnSparseArrayChunk: public TObject {
 public:
   THnSparseArrayChunk():
      fContent(0), fSingleCoordinateSize(0), fCoordinatesSize(0), fCoordinates(0), 
      fSumw2(0) {}

   THnSparseArrayChunk(UInt_t coordsize, bool errors, TArray* cont);
   virtual ~THnSparseArrayChunk();
   TArray   *fContent; // bin content
   Int_t    fSingleCoordinateSize; // size of a single bin coordinate
   Int_t    fCoordinatesSize; // size of the bin coordinate buffer
   Char_t   *fCoordinates;   //[fCoordinatesSize] compact bin coordinate buffer
   TArrayD  *fSumw2;  // bin errors

   void AddBin(ULong_t idx, Char_t* idxbuf);
   void AddBinContent(ULong_t idx, Double_t v = 1.) {
      fContent->SetAt(v + fContent->GetAt(idx), idx);
      if (fSumw2) fSumw2->AddAt(v * v, idx);
   }
   void Sumw2();
   ULong64_t GetEntries() const { return fCoordinatesSize / fSingleCoordinateSize; }
   Bool_t Matches(UInt_t idx, const Char_t* idxbuf) const {
      // Check whether bin at idx batches idxbuf.
      // If we don't store indexes we trust the caller that it does match,
      // see comment in THnSparseCompactBinCoord::GetHash().
      return fCoordinatesSize <= 4 || 
         !memcmp(fCoordinates + idx * fSingleCoordinateSize, idxbuf, fSingleCoordinateSize); }
   ClassDef(THnSparseArrayChunk, 1); // chunks of linearized bins
};

class THnSparseCompactBinCoord;

class THnSparse: public TNamed {
 private:
   UInt_t     fNdimensions;  // number of dimensions
   Long64_t   fFilledBins;   // number of filled bins
   Double_t   fEntries;      // number of entries, spread over chunks
   Double_t   fWeightSum;      // sum of weights, spread over chunks
   TObjArray  fAxes;         // axes of the histogram
   TExMap     fBins;         // filled bins
   TExMap     fBinsContinued;// filled bins for non-unique hashes, containing pairs of (bin index 0, bin index 1)
   UInt_t     fChunkSize;    // number of entries for each chunk
   TObjArray  fBinContent;   // array of THnSparseArrayChunk
   THnSparseCompactBinCoord *fCompactCoord; //! compact coordinate
   Double_t   *fIntegral;    //! array with bin weight sums
   enum {kNoInt, kValidInt, kInvalidInt} fIntegralStatus; //! status of integral

 protected:
   UInt_t GetChunkSize() const { return fChunkSize; }
   THnSparseCompactBinCoord* GetCompactCoord() const;
   THnSparseArrayChunk* GetChunk(UInt_t idx) const {
      return (THnSparseArrayChunk*) fBinContent[idx]; }

   THnSparseArrayChunk* AddChunk();
   virtual TArray* GenerateArray() const = 0;
   Long_t GetBinIndexForCurrentBin(Bool_t allocate);
   Long_t Fill(Long_t bin, Double_t w) {
      // Increment the bin content of "bin" by "w",
      // return the bin index.
      fEntries += 1;
      fWeightSum += w;
      if (fIntegralStatus == kValidInt)
         fIntegralStatus = kInvalidInt;
      THnSparseArrayChunk* chunk = GetChunk(bin / fChunkSize);
      chunk->AddBinContent(bin % fChunkSize, w);
      return bin;
   }

 public:
   THnSparse(const char* name, const char* title, UInt_t dim,
             UInt_t* nbins, Double_t* xmin, Double_t* xmax,
             UInt_t chunksize);
   THnSparse();
   virtual ~THnSparse();

   TObjArray* GetListOfAxes() { return &fAxes; }
   TAxis* GetAxis(UInt_t dim) const { return (TAxis*)fAxes[dim]; }

   Long_t Fill(Double_t *x, Double_t w = 1.) { return Fill(GetBin(x), w); }
   Long_t Fill(const char* name[], Double_t w = 1.) { return Fill(GetBin(name), w); }

   Double_t GetEntries() const { return fEntries; }
   Double_t GetWeightSum() const { return fWeightSum; }
   Long64_t GetNbins() const { return fFilledBins; }
   UInt_t   GetNdimensions() const { return fNdimensions; }

   Long_t GetBin(UInt_t* idx, Bool_t allocate = kTRUE);
   Long_t GetBin(Double_t* x, Bool_t allocate = kTRUE);
   Long_t GetBin(const char* name[], Bool_t allocate = kTRUE);

   void SetBinContent(UInt_t* x, Double_t v);
   void SetBinError(UInt_t* x, Double_t e);
   void AddBinContent(UInt_t* x, Double_t v = 1.);


   Double_t GetBinContent(UInt_t *idx) const;
   Double_t GetBinContent(Long64_t bin, UInt_t* idx = 0) const;
   Double_t GetBinError(UInt_t *idx) const;
   Double_t GetBinError(Long64_t linidx) const;

   Double_t GetSparseFractionBins() const;
   Double_t GetSparseFractionMem() const;

   TH1D*      Projection(UInt_t xDim, Option_t* option = "") const;
   TH2D*      Projection(UInt_t xDim, UInt_t yDim,
                         Option_t* option = "") const;
   TH3D*      Projection(UInt_t xDim, UInt_t yDim, UInt_t zDim,
                         Option_t* option = "") const;
   THnSparse* Projection(UInt_t ndim, UInt_t* dim,
                         Option_t* option = "") const;

   void Reset(Option_t* option = "");
   void Sumw2();

   Double_t ComputeIntegral();
   void GetRandom(Double_t *rand, Bool_t subBinRandom = kTRUE);

   //void Draw(Option_t* option = "");

   ClassDef(THnSparse, 1); // Interfaces of sparse n-dimensional histogram
};

template <class CONT>
class THnSparseT: public THnSparse {
 public:
   THnSparseT() {}
   THnSparseT(const char* name, const char* title, UInt_t dim,
              UInt_t* nbins, Double_t* xmin, Double_t* xmax,
              UInt_t chunksize = 1024 * 16):
      THnSparse(name, title, dim, nbins, xmin, xmax, chunksize) {}

   TArray* GenerateArray() const { return new CONT(GetChunkSize()); }
 private:
   ClassDef(THnSparseT, 1); // Sparse n-dimensional histogram with templated content
};

typedef THnSparseT<TArrayD> THnSparseD;
typedef THnSparseT<TArrayF> THnSparseF;
typedef THnSparseT<TArrayL> THnSparseL;
typedef THnSparseT<TArrayI> THnSparseI;
typedef THnSparseT<TArrayS> THnSparseS;
typedef THnSparseT<TArrayC> THnSparseC;

#endif //  ROOT_THnSparse
