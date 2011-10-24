// @(#)root/hist:$Id$
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
#ifndef ROOT_TFitResultPtr
#include "TFitResultPtr.h"
#endif
#ifndef ROOT_THnSparse_Internal
#include "THnSparse_Internal.h"
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
class TCollection;
class TH1;
class TH1D;
class TH2D;
class TH3D;
class TF1;



class THnSparseCompactBinCoord;

class THnSparse: public TNamed {
 private:
   Int_t      fNdimensions;  // number of dimensions
   Int_t      fChunkSize;    // number of entries for each chunk
   Long64_t   fFilledBins;   // number of filled bins
   TObjArray  fAxes;         // axes of the histogram
   TObjArray  fBrowsables;   //! browser-helpers for each axis
   TObjArray  fBinContent;   // array of THnSparseArrayChunk
   TExMap     fBins;         //! filled bins
   TExMap     fBinsContinued;//! filled bins for non-unique hashes, containing pairs of (bin index 0, bin index 1)
   Double_t   fEntries;      // number of entries, spread over chunks
   Double_t   fTsumw;        // total sum of weights
   Double_t   fTsumw2;       // total sum of weights squared; -1 if no errors are calculated
   TArrayD    fTsumwx;       // total sum of weight*X for each dimension
   TArrayD    fTsumwx2;      // total sum of weight*X*X for each dimension
   THnSparseCompactBinCoord *fCompactCoord; //! compact coordinate
   Double_t  *fIntegral;     //! array with bin weight sums
   enum {
      kNoInt,
      kValidInt,
      kInvalidInt
   } fIntegralStatus;        //! status of integral

   THnSparse(const THnSparse&); // Not implemented
   THnSparse& operator=(const THnSparse&); // Not implemented

 protected:

   THnSparse();
   THnSparse(const char* name, const char* title, Int_t dim,
             const Int_t* nbins, const Double_t* xmin, const Double_t* xmax,
             Int_t chunksize);
   Int_t GetChunkSize() const { return fChunkSize; }
   THnSparseCompactBinCoord* GetCompactCoord() const;
   THnSparseArrayChunk* GetChunk(Int_t idx) const {
      return (THnSparseArrayChunk*) fBinContent[idx]; }

   THnSparseArrayChunk* AddChunk();
   void FillExMap();
   virtual TArray* GenerateArray() const = 0;
   Long64_t GetBinIndexForCurrentBin(Bool_t allocate);
   Long64_t Fill(Long64_t bin, Double_t w) {
      // Increment the bin content of "bin" by "w",
      // return the bin index.
      fEntries += 1;
      if (GetCalculateErrors()) {
         fTsumw += w;
         fTsumw2 += w*w;
      }
      fIntegralStatus = kInvalidInt;
      THnSparseArrayChunk* chunk = GetChunk(bin / fChunkSize);
      chunk->AddBinContent(bin % fChunkSize, w);
      return bin;
   }
   THnSparse* CloneEmpty(const char* name, const char* title,
                         const TObjArray* axes, Int_t chunksize,
                         Bool_t keepTargetAxis) const;

   Bool_t CheckConsistency(const THnSparse *h, const char *tag) const;
   Bool_t IsInRange(Int_t *coord) const;
   TH1* CreateHist(const char* name, const char* title,
                   const TObjArray* axes, Bool_t keepTargetAxis) const;
   TObject* ProjectionAny(Int_t ndim, const Int_t* dim,
                          Bool_t wantSparse, Option_t* option = "") const;
   Bool_t PrintBin(Long64_t idx, Int_t* coord, Option_t* options) const;
   void AddInternal(const THnSparse* h, Double_t c, Bool_t rebinned);

 public:
   virtual ~THnSparse();

   static THnSparse* CreateSparse(const char* name, const char* title,
                                  const TH1* h1, Int_t ChunkSize = 1024 * 16);

   Int_t  GetNChunks() const { return fBinContent.GetEntriesFast(); }
   TObjArray* GetListOfAxes() { return &fAxes; }
   TAxis* GetAxis(Int_t dim) const { return (TAxis*)fAxes[dim]; }

   Long64_t Fill(const Double_t *x, Double_t w = 1.) {
      if (GetCalculateErrors()) {
         for (Int_t d = 0; d < fNdimensions; ++d) {
            const Double_t xd = x[d];
            fTsumwx[d]  += w * xd;
            fTsumwx2[d] += w * xd * xd;
         }
      }
      return Fill(GetBin(x), w);
   }
   Long64_t Fill(const char* name[], Double_t w = 1.) {
      return Fill(GetBin(name), w);
   }

   TFitResultPtr Fit(TF1 *f1 ,Option_t *option = "", Option_t *goption = "");
   TList* GetListOfFunctions() { return 0; }

   Double_t GetEntries() const { return fEntries; }
   Double_t GetWeightSum() const { return fTsumw; }
   Long64_t GetNbins() const { return fFilledBins; }
   Int_t    GetNdimensions() const { return fNdimensions; }
   Bool_t   GetCalculateErrors() const { return fTsumw2 >= 0.; }
   void     CalculateErrors(Bool_t calc = kTRUE) {
      // Calculate errors (or not if "calc" == kFALSE)
      if (calc) Sumw2();
      else fTsumw2 = -1.;
   }

   Long64_t GetBin(const Int_t* idx, Bool_t allocate = kTRUE);
   Long64_t GetBin(const Double_t* x, Bool_t allocate = kTRUE);
   Long64_t GetBin(const char* name[], Bool_t allocate = kTRUE);

   void SetBinEdges(Int_t idim, const Double_t* bins);
   void SetBinContent(const Int_t* x, Double_t v);
   void SetBinContent(Long64_t bin, Double_t v);
   void SetBinError(const Int_t* x, Double_t e);
   void SetBinError(Long64_t bin, Double_t e);
   void AddBinContent(const Int_t* x, Double_t v = 1.);
   void AddBinContent(Long64_t bin, Double_t v = 1.);
   void SetEntries(Double_t entries) { fEntries = entries; }
   void SetTitle(const char *title);

   Double_t GetBinContent(const Int_t *idx) const;
   Double_t GetBinContent(Long64_t bin, Int_t* idx = 0) const;
   Double_t GetBinError(const Int_t *idx) const;
   Double_t GetBinError(Long64_t linidx) const;

   Double_t GetSparseFractionBins() const;
   Double_t GetSparseFractionMem() const;

   Double_t GetSumw() const  { return fTsumw; }
   Double_t GetSumw2() const { return fTsumw2; }
   Double_t GetSumwx(Int_t dim) const  { return fTsumwx[dim]; }
   Double_t GetSumwx2(Int_t dim) const { return fTsumwx2[dim]; }

   TH1D*      Projection(Int_t xDim, Option_t* option = "") const;
   TH2D*      Projection(Int_t yDim, Int_t xDim,
                         Option_t* option = "") const;
   TH3D*      Projection(Int_t xDim, Int_t yDim, Int_t zDim,
                         Option_t* option = "") const;
   THnSparse* Projection(Int_t ndim, const Int_t* dim,
                         Option_t* option = "") const;

   THnSparse* Rebin(Int_t group) const;
   THnSparse* Rebin(const Int_t* group) const;

   Long64_t   Merge(TCollection* list);

   void Scale(Double_t c);
   void Add(const THnSparse* h, Double_t c=1.);
   void Multiply(const THnSparse* h);
   void Multiply(TF1* f, Double_t c = 1.);
   void Divide(const THnSparse* h);
   void Divide(const THnSparse* h1, const THnSparse* h2, Double_t c1 = 1., Double_t c2 = 1., Option_t* option="");
   void RebinnedAdd(const THnSparse* h, Double_t c=1.);

   void Reset(Option_t* option = "");
   void Sumw2();

   Double_t ComputeIntegral();
   void GetRandom(Double_t *rand, Bool_t subBinRandom = kTRUE);

   void Print(Option_t* option = "") const;
   void PrintEntries(Long64_t from = 0, Long64_t howmany = -1, Option_t* options = 0) const;
   void PrintBin(Int_t* coord, Option_t* options) const {
      PrintBin(-1, coord, options);
   }
   void PrintBin(Long64_t idx, Option_t* options) const;

   void Browse(TBrowser *b);
   Bool_t IsFolder() const { return kTRUE; }

   //void Draw(Option_t* option = "");

   ClassDef(THnSparse, 2); // Interfaces of sparse n-dimensional histogram
};



//______________________________________________________________________________
//
// Templated implementation of the abstract base THnSparse.
// All functionality and the interfaces to be used are in THnSparse!
//
// THnSparse does not know how to store any bin content itself. Instead, this
// is delegated to the derived, templated class: the template parameter decides
// what the format for the bin content is. In fact it even defines the array
// itself; possible implementations probably derive from TArray.
//
// Typedefs exist for template parematers with ROOT's generic types:
//
//   Templated name        Typedef       Bin content type
//   THnSparseT<TArrayC>   THnSparseC    Char_r
//   THnSparseT<TArrayS>   THnSparseS    Short_t
//   THnSparseT<TArrayI>   THnSparseI    Int_t
//   THnSparseT<TArrayL>   THnSparseL    Long_t
//   THnSparseT<TArrayF>   THnSparseF    Float_t
//   THnSparseT<TArrayD>   THnSparseD    Double_t
//
// We recommend to use THnSparseC wherever possible, and to map its value space
// of 256 possible values to e.g. float values outside the class. This saves an
// enourmous amount of memory. Only if more than 256 values need to be
// distinguished should e.g. THnSparseS or even THnSparseF be chosen.
//
// Implementation detail: the derived, templated class is kept extremely small
// on purpose. That way the (templated thus inlined) uses of this class will
// only create a small amount of machine code, in contrast to e.g. STL.
//______________________________________________________________________________

template <class CONT>
class THnSparseT: public THnSparse {
 public:
   THnSparseT() {}
   THnSparseT(const char* name, const char* title, Int_t dim,
              const Int_t* nbins, const Double_t* xmin = 0,
              const Double_t* xmax = 0, Int_t chunksize = 1024 * 16):
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
