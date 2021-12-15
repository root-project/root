// @(#)root/hist:$Id$
// Author: Axel Naumann, Nov 2011

/*************************************************************************
 * Copyright (C) 1995-2012, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_THN
#define ROOT_THN

#include "THnBase.h"

#include "TNDArray.h"

#include "TArrayD.h"

#include "TAxis.h"

class TH1;
class TH1D;
class TH2D;
class TH3D;
class THnSparse;
class TF1;

class THn: public THnBase {

protected:
   void AllocCoordBuf() const;
   void InitStorage(Int_t* nbins, Int_t chunkSize);

   THn() = default;
   THn(const char* name, const char* title, Int_t dim, const Int_t* nbins,
       const Double_t* xmin, const Double_t* xmax);

   THn(const char *name, const char *title, Int_t dim, const Int_t *nbins,
       const std::vector<std::vector<double>> &xbins);

public:
   virtual ~THn();

   static THn* CreateHn(const char* name, const char* title, const TH1* h1) {
      return (THn*) CreateHnAny(name, title, h1, kFALSE /*THn*/, -1);
   }
   static THn* CreateHn(const char* name, const char* title, const THnBase* hn) {
      return (THn*) CreateHnAny(name, title, hn, kFALSE /*THn*/, -1);
   }

   ROOT::Internal::THnBaseBinIter* CreateIter(Bool_t respectAxisRange) const;
   Long64_t GetNbins() const { return GetArray().GetNbins(); }

   Long64_t GetBin(const Int_t* idx) const {
      return GetArray().GetBin(idx);
   }
   Long64_t GetBin(const Double_t* x) const {
      if (fCoordBuf.empty())
         AllocCoordBuf();
      for (Int_t d = 0; d < fNdimensions; ++d) {
         fCoordBuf[d] = GetAxis(d)->FindFixBin(x[d]);
      }
      return GetArray().GetBin(fCoordBuf.data());
   }
   Long64_t GetBin(const char* name[]) const {
      if (fCoordBuf.empty())
         AllocCoordBuf();
      for (Int_t d = 0; d < fNdimensions; ++d) {
         fCoordBuf[d] = GetAxis(d)->FindBin(name[d]);
      }
      return GetArray().GetBin(fCoordBuf.data());
   }

   Long64_t GetBin(const Int_t* idx, Bool_t /*allocate*/ = kTRUE) {
      return const_cast<const THn*>(this)->GetBin(idx);
   }
   Long64_t GetBin(const Double_t* x, Bool_t /*allocate*/ = kTRUE) {
      return const_cast<const THn*>(this)->GetBin(x);
   }
   Long64_t GetBin(const char* name[], Bool_t /*allocate*/ = kTRUE) {
      return const_cast<const THn*>(this)->GetBin(name);
   }

   /// Increment the bin content of "bin" by "w", return the bin index.
   void FillBin(Long64_t bin, Double_t w) {
      GetArray().AddAt(bin, w);
      if (GetCalculateErrors()) {
         fSumw2.AddAt(bin, w * w);
      }
      FillBinBase(w);
   }

   /// Forwards to THnBase::SetBinContent().
   /// Non-virtual, CINT-compatible replacement of a using declaration.
   void SetBinContent(const Int_t* idx, Double_t v) {
      THnBase::SetBinContent(idx, v);
   }
   void SetBinContent(Long64_t bin, Double_t v) {
      GetArray().SetAsDouble(bin, v);
   }
   void SetBinError2(Long64_t bin, Double_t e2) {
      if (!GetCalculateErrors()) Sumw2();
      fSumw2.At(bin) = e2;
   }
   /// Forwards to THnBase::SetBinContent().
   /// Non-virtual, CINT-compatible replacement of a using declaration.
   void AddBinContent(const Int_t* idx, Double_t v = 1.) {
      THnBase::AddBinContent(idx, v);
   }
   void AddBinContent(Long64_t bin, Double_t v = 1.) {
      GetArray().AddAt(bin, v);
   }
   void AddBinError2(Long64_t bin, Double_t e2) {
      fSumw2.At(bin) += e2;
   }
   /// Forwards to THnBase::GetBinContent() overload.
   /// Non-virtual, CINT-compatible replacement of a using declaration.
   Double_t GetBinContent(const Int_t *idx) const {
      return THnBase::GetBinContent(idx);
   }
   /// Get the content of bin, and set its index if idx is != 0.
   Double_t GetBinContent(Long64_t bin, Int_t* idx = 0) const {
      if (idx) {
         const TNDArray& arr = GetArray();
         Long64_t prevCellSize = arr.GetNbins();
         for (Int_t i = 0; i < GetNdimensions(); ++i) {
            Long64_t cellSize = arr.GetCellSize(i);
            idx[i] = (bin % prevCellSize) / cellSize;
            prevCellSize = cellSize;
         }
      }
      return GetArray().AtAsDouble(bin);
   }
   Double_t GetBinError2(Long64_t linidx) const {
      return GetCalculateErrors() ? fSumw2.At(linidx) : GetBinContent(linidx);
   }

   virtual const TNDArray& GetArray() const = 0;
   virtual TNDArray& GetArray() = 0;

   void Sumw2();

   /// Forwards to THnBase::Projection().
   /// Non-virtual, as a CINT-compatible replacement of a using declaration.
   TH1D*      Projection(Int_t xDim, Option_t* option = "") const {
      return THnBase::Projection(xDim, option);
   }

   /// Forwards to THnBase::Projection().
   /// Non-virtual, as a CINT-compatible replacement of a using declaration.
   TH2D*      Projection(Int_t yDim, Int_t xDim,
                         Option_t* option = "") const {
      return THnBase::Projection(yDim, xDim, option);
   }

   /// Forwards to THnBase::Projection().
   /// Non-virtual, as a CINT-compatible replacement of a using declaration.
   TH3D*      Projection(Int_t xDim, Int_t yDim, Int_t zDim,
                         Option_t* option = "") const {
      return THnBase::Projection(xDim, yDim, zDim, option);
   }

   THn*       Projection(Int_t ndim, const Int_t* dim,
                         Option_t* option = "") const {
      return (THn*) ProjectionND(ndim, dim, option);
   }

   THn*       Rebin(Int_t group) const {
      return (THn*) RebinBase(group);
   }
   THn*       Rebin(const Int_t* group) const {
      return (THn*) RebinBase(group);
   }

   void Reset(Option_t* option = "");

protected:
   TNDArrayT<Double_t> fSumw2; // bin error, lazy allocation happens in TNDArrayT
   mutable std::vector<Int_t> fCoordBuf; //! Temporary buffer

   ClassDef(THn, 1); //Base class for multi-dimensional histogram
};


//______________________________________________________________________________
/** \class THnT
 Templated implementation of the abstract base THn.
 All functionality and the interfaces to be used are in THn!

 THn does not know how to store any bin content itself. Instead, this
 is delegated to the derived, templated class: the template parameter decides
 what the format for the bin content is. The actual storage is delegated to
 TNDArrayT<T>.

 Typedefs exist for template parameters with ROOT's generic types:

 Templated name   |     Typedef   |    Bin content type
 -----------------|---------------|--------------------
   THnT<Char_t>   |       THnC    |     Char_t
   THnT<Short_t>  |       THnS    |     Short_t
   THnT<Int_t>    |       THnI    |     Int_t
   THnT<Long_t>   |       THnL    |     Long_t
   THnT<Float_t>  |       THnF    |     Float_t
   THnT<Double_t> |       THnD    |     Double_t

 We recommend to use THnC wherever possible, and to map its value space
 of 256 possible values to e.g. float values outside the class. This saves an
 enormous amount of memory. Only if more than 256 values need to be
 distinguished should e.g. THnS or even THnF be chosen.

 Implementation detail: the derived, templated class is kept extremely small
 on purpose. That way the (templated thus inlined) uses of this class will
 only create a small amount of machine code, in contrast to e.g. STL.
*/

template <typename T>
class THnT: public THn {
public:
   THnT() {}

   THnT(const char* name, const char* title,
       Int_t dim, const Int_t* nbins,
       const Double_t* xmin, const Double_t* xmax):
   THn(name, title, dim, nbins, xmin, xmax),
   fArray(dim, nbins, true)  {}

   THnT(const char *name, const char *title, Int_t dim, const Int_t *nbins,
        const std::vector<std::vector<double>> &xbins)
      : THn(name, title, dim, nbins, xbins), fArray(dim, nbins, true)
   {
   }

   const TNDArray& GetArray() const { return fArray; }
   TNDArray& GetArray() { return fArray; }

protected:
   TNDArrayT<T> fArray; ///< Bin content
   ClassDef(THnT, 1);   ///< Multi-dimensional histogram with templated storage
};

typedef THnT<Float_t>  THnF;
typedef THnT<Double_t> THnD;
typedef THnT<Char_t>   THnC;
typedef THnT<Short_t>  THnS;
typedef THnT<Int_t>    THnI;
typedef THnT<Long_t>   THnL;
typedef THnT<Long64_t> THnL64;

#endif // ROOT_THN
