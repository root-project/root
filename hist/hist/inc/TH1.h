// @(#)root/hist:$Id$
// Author: Rene Brun   26/12/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TH1
#define ROOT_TH1


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TH1                                                                  //
//                                                                      //
// 1-Dim histogram base class.                                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TAxis.h"

#include "TAttLine.h"

#include "TAttFill.h"

#include "TAttMarker.h"

#include "TArrayC.h"
#include "TArrayS.h"
#include "TArrayI.h"
#include "TArrayL64.h"
#include "TArrayF.h"
#include "TArrayD.h"
#include "TDirectory.h"
#include "Foption.h"

#include "TVectorFfwd.h"
#include "TVectorDfwd.h"

#include "TFitResultPtr.h"

#include <algorithm>
#include <cfloat>
#include <string>
#include <stdexcept>
#include <type_traits>
#include <array>
#include <numeric>

class TF1;
class TH1D;
class TBrowser;
class TDirectory;
class TList;
class TCollection;
class TVirtualFFT;
class TVirtualHistPainter;
class TRandom;

namespace ROOT::Internal {
/**
 * \brief Creates a sliced copy of the given histogram.
 *
 * \tparam T The type of the histogram.
 * \param histo The histogram to slice.
 * \param args A vector of integers specifying the low and upper edges of the slice for each dimension.
 *
 * \return A new histogram object that is a sliced version of the input histogram.
 */
template <typename T>
T Slice(const T &histo, std::vector<Int_t> &args)
{
   static_assert(std::is_pointer_v<decltype(histo.fArray)>,
                 "The type of the histogram to slice must expose the internal array data.");

   TDirectory::TContext _{nullptr};
   T slicedHisto = histo;

   using ValueType = std::remove_pointer_t<decltype(slicedHisto.fArray)>;
   slicedHisto.template SliceHistoInPlace<ValueType>(args, slicedHisto.fArray, slicedHisto.fN);
   return slicedHisto;
}

/**
 * \brief Sets the content of a slice in a histogram.
 *
 * \tparam T The type of the histogram.
 * \param histo The histogram to set the slice content for.
 * \param input A vector of values to assign to the bins in the specified slice. All input types are converted
 *              to Double_t as that's the type `SetBinContent` eventually uses.
 * \param sliceEdges A vector of pairs specifying the low and upper edges of the slice for each dimension.
 */
template <typename T>
void SetSliceContent(T &histo, const std::vector<Double_t> &input,
                     const std::vector<std::pair<Int_t, Int_t>> &sliceEdges)
{
   static_assert(std::is_pointer_v<decltype(histo.fArray)>,
                 "The type of the histogram to slice must expose the internal array data.");

   using ValueType = std::remove_pointer_t<decltype(histo.fArray)>;
   histo.template SetSliceContent<ValueType>(input, sliceEdges, histo.fArray);
}
} // namespace ROOT::Internal

class TH1 : public TNamed, public TAttLine, public TAttFill, public TAttMarker {

public:

   /// Enumeration specifying type of statistics for bin errors
   enum  EBinErrorOpt {
         kNormal = 0,    ///< Errors with Normal (Wald) approximation: errorUp=errorLow= sqrt(N)
         kPoisson = 1 ,  ///< Errors from Poisson interval at 68.3% (1 sigma)
         kPoisson2 = 2   ///< Errors from Poisson interval at 95% CL (~ 2 sigma)
   };

   /// Enumeration specifying which axes can be extended
   enum {
      kNoAxis  = 0,      ///< NOTE: Must always be 0 !!!
      kXaxis = BIT(0),
      kYaxis = BIT(1),
      kZaxis = BIT(2),
      kAllAxes = kXaxis | kYaxis | kZaxis
   };

   /// Enumeration specifying the way to treat statoverflow
   enum  EStatOverflows {
         kIgnore = 0,   ///< Override global flag ignoring the overflows
         kConsider = 1, ///< Override global flag considering the overflows
         kNeutral = 2,  ///< Adapt to the global flag
   };

   /// Enumeration specifying inconsistencies between two histograms,
   /// in increasing severity.
   enum  EInconsistencyBits {
         kFullyConsistent = 0,
         kDifferentLabels = BIT(0),
         kDifferentBinLimits = BIT(1),
         kDifferentAxisLimits = BIT(2),
         kDifferentNumberOfBins = BIT(3),
         kDifferentDimensions = BIT(4)
   };

   friend class TH1Merger;

protected:
    Int_t         fNcells;          ///<  Number of bins(1D), cells (2D) +U/Overflows
    TAxis         fXaxis;           ///<  X axis descriptor
    TAxis         fYaxis;           ///<  Y axis descriptor
    TAxis         fZaxis;           ///<  Z axis descriptor
    Short_t       fBarOffset;       ///<  (1000*offset) for bar charts or legos
    Short_t       fBarWidth;        ///<  (1000*width) for bar charts or legos
    Double_t      fEntries;         ///<  Number of entries
    Double_t      fTsumw;           ///<  Total Sum of weights
    Double_t      fTsumw2;          ///<  Total Sum of squares of weights
    Double_t      fTsumwx;          ///<  Total Sum of weight*X
    Double_t      fTsumwx2;         ///<  Total Sum of weight*X*X
    Double_t      fMaximum;         ///<  Maximum value for plotting
    Double_t      fMinimum;         ///<  Minimum value for plotting
    Double_t      fNormFactor;      ///<  Normalization factor
    TArrayD       fContour;         ///<  Array to display contour levels
    TArrayD       fSumw2;           ///<  Array of sum of squares of weights
    TString       fOption;          ///<  Histogram options
    TList        *fFunctions;       ///<->Pointer to list of functions (fits and user)
    Int_t         fBufferSize;      ///<  fBuffer size
    Double_t     *fBuffer;          ///<[fBufferSize] entry buffer
    TDirectory   *fDirectory;       ///<! Pointer to directory holding this histogram
    Int_t         fDimension;       ///<! Histogram dimension (1, 2 or 3 dim)
    Double_t     *fIntegral;        ///<! Integral of bins used by GetRandom
    TVirtualHistPainter *fPainter;  ///<! Pointer to histogram painter
    EBinErrorOpt  fBinStatErrOpt;   ///<  Option for bin statistical errors
    EStatOverflows fStatOverflows;  ///<  Per object flag to use under/overflows in statistics
    static Int_t  fgBufferSize;     ///<! Default buffer size for automatic histograms
    static Bool_t fgAddDirectory;   ///<! Flag to add histograms to the directory
    static Bool_t fgStatOverflows;  ///<! Flag to use under/overflows in statistics
    static Bool_t fgDefaultSumw2;   ///<! Flag to call TH1::Sumw2 automatically at histogram creation time

public:
   static Int_t FitOptionsMake(Option_t *option, Foption_t &Foption);

private:
   void    Build();

   TH1(const TH1&) = delete;
   TH1& operator=(const TH1&) = delete;

   /**
    * \brief Slices a histogram in place based on the specified bin ranges for each dimension.
    *
    * This function modifies the histogram by extracting a sub-region defined by the provided
    * bin ranges for each dimension. The resulting histogram will have updated bin counts,
    * edges, and contents. Bin contents outside the range fall into the flow bins.
    * The histogram's internal data array is freed and reallocated by this function.
    * This function is used by the python implementation of the Unified Histogram Interface (UHI)
    * for slicing.
    *
    * \tparam ValueType The type of the histogram's data.
    * \param args A vector of integers specifying the low and upper edges of the slice for each dimension.
    * \param dataArray A pointer to the histogram's data array.
    * \param fN The size of dataArray.
    */
   template <typename ValueType>
   void SliceHistoInPlace(std::vector<Int_t> &args, ValueType *&dataArray, Int_t &fN)
   {
      constexpr Int_t kMaxDim = 3;
      Int_t ndim = args.size() / 2;
      if (ndim != fDimension) {
         throw std::invalid_argument(
            Form("Number of dimensions in slice (%d) does not match histogram dimension (%d).", ndim, fDimension));
      }

      // Compute new bin counts and edges
      std::array<Int_t, kMaxDim> nBins{}, totalBins{};
      std::array<Double_t, kMaxDim> lowEdge{}, upEdge{};
      for (decltype(ndim) d = 0; d < ndim; ++d) {
         const auto &axis = (d == 0 ? fXaxis : d == 1 ? fYaxis : fZaxis);
         auto start = std::max(1, args[d * 2]);
         auto end = std::min(axis.GetNbins() + 1, args[d * 2 + 1]);
         nBins[d] = end - start;
         lowEdge[d] = axis.GetBinLowEdge(start);
         upEdge[d] = axis.GetBinLowEdge(end);
         totalBins[d] = axis.GetNbins() + 2;
         args[2 * d] = start;
         args[2 * d + 1] = end;
      }

      // Compute layout sizes for slice
      size_t rowSz = nBins[0] + 2;
      size_t planeSz = rowSz * (ndim > 1 ? nBins[1] + 2 : 1);
      size_t newSize = planeSz * (ndim > 2 ? nBins[2] + 2 : 1);

      // Allocate the new array
      auto *newArr = new ValueType[newSize]();

      auto rowIncr = 1 + (ndim > 1 ? (rowSz - 1) : 0);
      ValueType under = 0, over = 0;
      Bool_t firstUnder = false;
      Int_t lastOverIdx = 0;

      // Copy the valid slice bins
      size_t dstIdx = 1 + (ndim > 1 ? rowSz : 0) + (ndim > 2 ? planeSz : 0);
      for (auto z = (ndim > 2 ? args[4] : 0); z < (ndim > 2 ? args[5] : 1); ++z) {
         for (auto y = (ndim > 1 ? args[2] : 0); y < (ndim > 1 ? args[3] : 1); ++y) {
            size_t rowStart = z * totalBins[1] * totalBins[0] + y * totalBins[0] + args[0];
            std::copy_n(dataArray + rowStart, nBins[0], newArr + dstIdx);
            if (!firstUnder) {
               under = std::accumulate(dataArray, dataArray + rowStart, ValueType{});
               firstUnder = true;
            } else {
               under += std::accumulate(dataArray + rowStart - args[0], dataArray + rowStart, ValueType{});
            }
            lastOverIdx = rowStart - args[0] + totalBins[0];
            over += std::accumulate(dataArray + rowStart + nBins[0], dataArray + lastOverIdx, ValueType{});
            dstIdx += rowIncr;
         }
         if (ndim > 2) {
            dstIdx += 2 * rowIncr;
         }
      }

      // Copy the flow bins
      over += std::accumulate(dataArray + lastOverIdx, dataArray + fN, ValueType{});
      newArr[0] = under;
      newArr[newSize - 1] = over;

      // Assign the new array
      delete[] dataArray;
      dataArray = newArr;

      // Reconfigure Axes
      if (ndim == 1) {
         this->SetBins(nBins[0], lowEdge[0], upEdge[0]);
      } else if (ndim == 2) {
         this->SetBins(nBins[0], lowEdge[0], upEdge[0], nBins[1], lowEdge[1], upEdge[1]);
      } else if (ndim == 3) {
         this->SetBins(nBins[0], lowEdge[0], upEdge[0], nBins[1], lowEdge[1], upEdge[1], nBins[2], lowEdge[2],
                       upEdge[2]);
      }

      // Update the statistics
      ResetStats();
   }

   template <typename T>
   friend T ROOT::Internal::Slice(const T &histo, std::vector<Int_t> &args);

   /**
    * \brief Sets the content of a slice of bins in a histogram.
    *
    * This function allows setting the content of a slice of bins in a histogram
    * by specifying the edges of the slice and the corresponding values to assign.
    *
    * \tparam ValueType The type of the histogram's data.
    * \param values A vector of values to assign to the bins in the specified slice.
    * \param sliceEdges A vector of pairs specifying the low and upper edges of the slice for each dimension.
    * \param dataArray A pointer to the histogram's data array.
    */
   template <typename ValueType>
   void SetSliceContent(const std::vector<Double_t> &values, const std::vector<std::pair<Int_t, Int_t>> &sliceEdges,
                        ValueType *dataArray)
   {
      const Int_t ndim = sliceEdges.size();
      if (ndim != fDimension) {
         throw std::invalid_argument(Form(
            "Number of edges in the specified slice (%d) does not match histogram dimension (%d).", ndim, fDimension));
      }

      // Get the indices to set
      auto getSliceIndices = [](const std::vector<std::pair<Int_t, Int_t>> &edges) -> std::vector<std::vector<Int_t>> {
         const auto dim = edges.size();
         if (dim == 0) {
            return {};
         }

         std::vector<std::vector<Int_t>> slices(dim);
         for (size_t d = 0; d < dim; ++d) {
            for (auto val = edges[d].first; val < edges[d].second; ++val) {
               slices[d].push_back(val);
            }
         }

         size_t totalCombinations = 1;
         for (const auto &slice : slices) {
            totalCombinations *= slice.size();
         }

         std::vector<std::vector<Int_t>> result(totalCombinations, std::vector<Int_t>(3, 0));
         for (size_t d = 0; d < slices.size(); ++d) {
            size_t repeat = 1;
            for (size_t i = d + 1; i < slices.size(); ++i) {
               repeat *= slices[i].size();
            }

            size_t index = 0;
            for (size_t i = 0; i < totalCombinations; ++i) {
               result[i][d] = slices[d][(index / repeat) % slices[d].size()];
               ++index;
            }
         }

         return result;
      };

      auto sliceIndices = getSliceIndices(sliceEdges);

      if (values.size() != sliceIndices.size()) {
         throw std::invalid_argument("Number of provided values does not match number of bins to set.");
      }

      for (size_t i = 0; i < sliceIndices.size(); ++i) {
         auto globalBin = this->GetBin(sliceIndices[i][0], sliceIndices[i][1], sliceIndices[i][2]);

         // Set the bin content
         dataArray[globalBin] = values[i];
      }

      // Update the statistics
      ResetStats();
   }

   template <typename T>
   friend void ROOT::Internal::SetSliceContent(T &histo, const std::vector<Double_t> &input,
                                               const std::vector<std::pair<Int_t, Int_t>> &sliceEdges);

protected:
   TH1();
   TH1(const char *name,const char *title,Int_t nbinsx,Double_t xlow,Double_t xup);
   TH1(const char *name,const char *title,Int_t nbinsx,const Float_t *xbins);
   TH1(const char *name,const char *title,Int_t nbinsx,const Double_t *xbins);

   Int_t            AxisChoice(Option_t *axis) const;
   virtual Int_t    BufferFill(Double_t x, Double_t w);
   virtual Bool_t   FindNewAxisLimits(const TAxis* axis, const Double_t point, Double_t& newMin, Double_t &newMax);
   TString          ProvideSaveName(Option_t *option, Bool_t testfdir = kFALSE);
   virtual void     SavePrimitiveHelp(std::ostream &out, const char *hname, Option_t *option = "");
   static Bool_t    RecomputeAxisLimits(TAxis& destAxis, const TAxis& anAxis);
   static Bool_t    SameLimitsAndNBins(const TAxis& axis1, const TAxis& axis2);
   Bool_t   IsEmpty() const;
   UInt_t GetAxisLabelStatus() const;

   inline static Double_t AutoP2GetPower2(Double_t x, Bool_t next = kTRUE);
   inline static Int_t AutoP2GetBins(Int_t n);
   virtual Int_t AutoP2FindLimits(Double_t min, Double_t max);

   virtual Double_t DoIntegral(Int_t ix1, Int_t ix2, Int_t iy1, Int_t iy2, Int_t iz1, Int_t iz2, Double_t & err,
                               Option_t * opt, Bool_t doerr = kFALSE) const;

   virtual void     DoFillN(Int_t ntimes, const Double_t *x, const Double_t *w, Int_t stride=1);
   Bool_t    GetStatOverflowsBehaviour() const { return EStatOverflows::kNeutral == fStatOverflows ? fgStatOverflows : EStatOverflows::kConsider == fStatOverflows; }

   static bool CheckAxisLimits(const TAxis* a1, const TAxis* a2);
   static bool CheckBinLimits(const TAxis* a1, const TAxis* a2);
   static bool CheckBinLabels(const TAxis* a1, const TAxis* a2);
   static bool CheckEqualAxes(const TAxis* a1, const TAxis* a2);
   static bool CheckConsistentSubAxes(const TAxis *a1, Int_t firstBin1, Int_t lastBin1, const TAxis *a2, Int_t firstBin2=0, Int_t lastBin2=0);
   int LoggedInconsistency(const char* name, const TH1* h1, const TH1* h2, bool useMerge=false) const;

public:
   /// TH1 status bits
   enum EStatusBits {
      kNoStats     = BIT(9),   ///< Don't draw stats box
      kUserContour = BIT(10),  ///< User specified contour levels
      // kCanRebin    = BIT(11), ///< FIXME DEPRECATED - to be removed, replaced by SetCanExtend / CanExtendAllAxes
      kLogX        = BIT(15),  ///< X-axis in log scale
      kIsZoomed   = BIT(16),   ///< Bit set when zooming on Y axis
      kNoTitle     = BIT(17),  ///< Don't draw the histogram title
      kIsAverage   = BIT(18),  ///< Bin contents are average (used by Add)
      kIsNotW      = BIT(19),  ///< Histogram is forced to be not weighted even when the histogram is filled with weighted
                               /// different than 1.
      kAutoBinPTwo = BIT(20),  ///< Use Power(2)-based algorithm for autobinning
      kIsHighlight = BIT(21)   ///< bit set if histo is highlight
   };
   /// Size of statistics data (size of  array used in GetStats()/ PutStats )
   ///  - s[0]  = sumw       s[1]  = sumw2
   ///  - s[2]  = sumwx      s[3]  = sumwx2
   ///  - s[4]  = sumwy      s[5]  = sumwy2   s[6]  = sumwxy
   ///  - s[7]  = sumwz      s[8]  = sumwz2   s[9]  = sumwxz   s[10]  = sumwyz
   ///  - s[11] = sumwt      s[12] = sumwt2                 (11 and 12 used only by TProfile3D)
   enum {
      kNstat       = 13  ///< Size of statistics data (up to TProfile3D)
   };


   ~TH1() override;

   virtual Bool_t   Add(TF1 *h1, Double_t c1=1, Option_t *option="");
   virtual Bool_t   Add(const TH1 *h1, Double_t c1=1);
   virtual Bool_t   Add(const TH1 *h, const TH1 *h2, Double_t c1=1, Double_t c2=1);
   /// Increment bin content by 1.
   /// Passing an out-of-range bin leads to undefined behavior
   virtual void     AddBinContent(Int_t bin) = 0;
   /// Increment bin content by a weight w.
   /// Passing an out-of-range bin leads to undefined behavior
   virtual void     AddBinContent(Int_t bin, Double_t w) = 0;
   static  void     AddDirectory(Bool_t add=kTRUE);
   static  Bool_t   AddDirectoryStatus();
           void     Browse(TBrowser *b) override;
   virtual Bool_t   CanExtendAllAxes() const;
   virtual Double_t Chi2Test(const TH1* h2, Option_t *option = "UU", Double_t *res = nullptr) const;
   virtual Double_t Chi2TestX(const TH1* h2, Double_t &chi2, Int_t &ndf, Int_t &igood,Option_t *option = "UU",  Double_t *res = nullptr) const;
   virtual Double_t Chisquare(TF1 * f1, Option_t *option = "") const;
   static  Int_t    CheckConsistency(const TH1* h1, const TH1* h2);
   virtual void     ClearUnderflowAndOverflow();
   virtual Double_t ComputeIntegral(Bool_t onlyPositive = false);
           TObject* Clone(const char *newname = "") const override;
           void     Copy(TObject &hnew) const override;
   virtual void     DirectoryAutoAdd(TDirectory *);
           Int_t    DistancetoPrimitive(Int_t px, Int_t py) override;
   virtual Bool_t   Divide(TF1 *f1, Double_t c1=1);
   virtual Bool_t   Divide(const TH1 *h1);
   virtual Bool_t   Divide(const TH1 *h1, const TH1 *h2, Double_t c1=1, Double_t c2=1, Option_t *option="");
           void     Draw(Option_t *option = "") override;
   virtual TH1     *DrawCopy(Option_t *option="", const char * name_postfix = "_copy") const;
   virtual TH1     *DrawNormalized(Option_t *option="", Double_t norm=1) const;
   virtual void     DrawPanel(); // *MENU*
   virtual Int_t    BufferEmpty(Int_t action=0);
   virtual void     Eval(TF1 *f1, Option_t *option="");
           void     ExecuteEvent(Int_t event, Int_t px, Int_t py) override;
   virtual void     ExtendAxis(Double_t x, TAxis *axis);
   virtual TH1     *FFT(TH1* h_output, Option_t *option);
   virtual Int_t    Fill(Double_t x);
   virtual Int_t    Fill(Double_t x, Double_t w);
   virtual Int_t    Fill(const char *name, Double_t w);
   virtual void     FillN(Int_t ntimes, const Double_t *x, const Double_t *w, Int_t stride=1);
   virtual void     FillN(Int_t, const Double_t *, const Double_t *, const Double_t *, Int_t) {}
   virtual void     FillRandom(TF1 *f1, Int_t ntimes=5000, TRandom * rng = nullptr);
           void     FillRandom(const char *fname, Int_t ntimes=5000, TRandom * rng = nullptr);
   virtual void     FillRandom(TH1 *h, Int_t ntimes=5000, TRandom * rng = nullptr);
   virtual Int_t    FindBin(Double_t x, Double_t y=0, Double_t z=0);
   virtual Int_t    FindFixBin(Double_t x, Double_t y=0, Double_t z=0) const;
   virtual Int_t    FindFirstBinAbove(Double_t threshold=0, Int_t axis=1, Int_t firstBin=1, Int_t lastBin=-1) const;
   virtual Int_t    FindLastBinAbove (Double_t threshold=0, Int_t axis=1, Int_t firstBin=1, Int_t lastBin=-1) const;
           TObject *FindObject(const char *name) const override;
           TObject *FindObject(const TObject *obj) const override;
   virtual TFitResultPtr    Fit(const char *formula ,Option_t *option="" ,Option_t *goption="", Double_t xmin=0, Double_t xmax=0); // *MENU*
   virtual TFitResultPtr    Fit(TF1 *f1 ,Option_t *option="" ,Option_t *goption="", Double_t xmin=0, Double_t xmax=0);
   virtual void     FitPanel(); // *MENU*
   TH1             *GetAsymmetry(TH1* h2, Double_t c2=1, Double_t dc2=0);
   Int_t            GetBufferLength() const {return fBuffer ? (Int_t)fBuffer[0] : 0;}
   Int_t            GetBufferSize  () const {return fBufferSize;}
   const   Double_t *GetBuffer() const {return fBuffer;}
   static  Int_t    GetDefaultBufferSize();
   virtual Double_t *GetIntegral();
   TH1             *GetCumulative(Bool_t forward = kTRUE, const char* suffix = "_cumulative") const;

   TList           *GetListOfFunctions() const { return fFunctions; }

   virtual Int_t    GetNdivisions(Option_t *axis="X") const;
   virtual Color_t  GetAxisColor(Option_t *axis="X") const;
   virtual Color_t  GetLabelColor(Option_t *axis="X") const;
   virtual Style_t  GetLabelFont(Option_t *axis="X") const;
   virtual Float_t  GetLabelOffset(Option_t *axis="X") const;
   virtual Float_t  GetLabelSize(Option_t *axis="X") const;
   virtual Style_t  GetTitleFont(Option_t *axis="X") const;
   virtual Float_t  GetTitleOffset(Option_t *axis="X") const;
   virtual Float_t  GetTitleSize(Option_t *axis="X") const;
   virtual Float_t  GetTickLength(Option_t *axis="X") const;
   virtual Float_t  GetBarOffset() const {return Float_t(0.001*Float_t(fBarOffset));}
   virtual Float_t  GetBarWidth() const  {return Float_t(0.001*Float_t(fBarWidth));}
   virtual Int_t    GetContour(Double_t *levels = nullptr);
   virtual Double_t GetContourLevel(Int_t level) const;
   virtual Double_t GetContourLevelPad(Int_t level) const;

   virtual Int_t    GetBin(Int_t binx, Int_t biny=0, Int_t binz=0) const;
   virtual void     GetBinXYZ(Int_t binglobal, Int_t &binx, Int_t &biny, Int_t &binz) const;
   virtual Double_t GetBinCenter(Int_t bin) const;
   virtual Double_t GetBinContent(Int_t bin) const;
   virtual Double_t GetBinContent(Int_t bin, Int_t) const { return GetBinContent(bin); }
   virtual Double_t GetBinContent(Int_t bin, Int_t, Int_t) const { return GetBinContent(bin); }
   virtual Double_t GetBinError(Int_t bin) const;
   virtual Double_t GetBinError(Int_t binx, Int_t biny) const { return GetBinError(GetBin(binx, biny)); } // for 2D histograms only
   virtual Double_t GetBinError(Int_t binx, Int_t biny, Int_t binz) const { return GetBinError(GetBin(binx, biny, binz)); } // for 3D histograms only
   virtual Double_t GetBinErrorLow(Int_t bin) const;
   virtual Double_t GetBinErrorUp(Int_t bin) const;
   virtual EBinErrorOpt  GetBinErrorOption() const { return fBinStatErrOpt; }
   virtual Double_t GetBinLowEdge(Int_t bin) const;
   virtual Double_t GetBinWidth(Int_t bin) const;
   virtual Double_t GetBinWithContent(Double_t c, Int_t &binx, Int_t firstx = 0, Int_t lastx = 0,Double_t maxdiff = 0) const;
   virtual void     GetCenter(Double_t *center) const;
   static  Bool_t   GetDefaultSumw2();
   TDirectory      *GetDirectory() const {return fDirectory;}
   virtual Double_t GetEntries() const;
   virtual Double_t GetEffectiveEntries() const;
   virtual TF1     *GetFunction(const char *name) const;
   virtual Int_t    GetDimension() const { return fDimension; }
   virtual Double_t GetKurtosis(Int_t axis=1) const;
   virtual void     GetLowEdge(Double_t *edge) const;
   virtual Double_t GetMaximum(Double_t maxval=FLT_MAX) const;
   virtual Int_t    GetMaximumBin() const;
   virtual Int_t    GetMaximumBin(Int_t &locmax, Int_t &locmay, Int_t &locmaz) const;
   virtual Double_t GetMaximumStored() const {return fMaximum;}
   virtual Double_t GetMinimum(Double_t minval=-FLT_MAX) const;
   virtual Int_t    GetMinimumBin() const;
   virtual Int_t    GetMinimumBin(Int_t &locmix, Int_t &locmiy, Int_t &locmiz) const;
   virtual Double_t GetMinimumStored() const {return fMinimum;}
   virtual void     GetMinimumAndMaximum(Double_t& min, Double_t& max) const;
   virtual Double_t GetMean(Int_t axis=1) const;
   virtual Double_t GetMeanError(Int_t axis=1) const;
   virtual Int_t    GetNbinsX() const {return fXaxis.GetNbins();}
   virtual Int_t    GetNbinsY() const {return fYaxis.GetNbins();}
   virtual Int_t    GetNbinsZ() const {return fZaxis.GetNbins();}
   virtual Int_t    GetNcells() const {return fNcells; }
   virtual Double_t GetNormFactor() const {return fNormFactor;}
           char    *GetObjectInfo(Int_t px, Int_t py) const override;
          Option_t *GetOption() const  override { return fOption.Data(); }

   TVirtualHistPainter *GetPainter(Option_t *option="");

   virtual Int_t    GetQuantiles(Int_t n, Double_t *xp, const Double_t *p = nullptr);
   virtual Double_t GetRandom(TRandom * rng = nullptr) const;
   virtual void     GetStats(Double_t *stats) const;
   virtual Double_t GetStdDev(Int_t axis=1) const;
   virtual Double_t GetStdDevError(Int_t axis=1) const;
   Double_t         GetSumOfAllWeights(const bool includeOverflow) const;
   /// Return the sum of weights across all bins excluding under/overflows.
   /// \see TH1::GetSumOfAllWeights()
   virtual Double_t GetSumOfWeights() const { return GetSumOfAllWeights(false); }
   virtual TArrayD *GetSumw2() {return &fSumw2;}
   virtual const TArrayD *GetSumw2() const {return &fSumw2;}
   virtual Int_t    GetSumw2N() const {return fSumw2.fN;}
           /// This function returns the Standard Deviation (Sigma) of the distribution not the Root Mean Square (RMS).
           /// The name "RMS" is been often used as a synonym for the Standard Deviation and it was introduced many years ago (Hbook/PAW times).
           /// We keep the name GetRMS for continuity as an alias to GetStdDev. GetStdDev() should be used instead.
           Double_t GetRMS(Int_t axis=1) const { return GetStdDev(axis); }
           Double_t GetRMSError(Int_t axis=1) const { return GetStdDevError(axis); }

   virtual Double_t GetSkewness(Int_t axis=1) const;
           EStatOverflows GetStatOverflows() const { return fStatOverflows; } ///< Get the behaviour adopted by the object about the statoverflows. See EStatOverflows for more information.
           TAxis*   GetXaxis()  { return &fXaxis; }
           TAxis*   GetYaxis()  { return &fYaxis; }
           TAxis*   GetZaxis()  { return &fZaxis; }
     const TAxis*   GetXaxis() const { return &fXaxis; }
     const TAxis*   GetYaxis() const { return &fYaxis; }
     const TAxis*   GetZaxis() const { return &fZaxis; }
   virtual Double_t Integral(Option_t *option="") const;
   virtual Double_t Integral(Int_t binx1, Int_t binx2, Option_t *option="") const;
   virtual Double_t IntegralAndError(Int_t binx1, Int_t binx2, Double_t & err, Option_t *option="") const;
   virtual Double_t Interpolate(Double_t x) const;
   virtual Double_t Interpolate(Double_t x, Double_t y) const;
   virtual Double_t Interpolate(Double_t x, Double_t y, Double_t z) const;
           Bool_t   IsBinOverflow(Int_t bin, Int_t axis = 0) const;
           Bool_t   IsBinUnderflow(Int_t bin, Int_t axis = 0) const;
   virtual Bool_t   IsHighlight() const { return TestBit(kIsHighlight); }
   virtual Double_t AndersonDarlingTest(const TH1 *h2, Option_t *option="") const;
   virtual Double_t AndersonDarlingTest(const TH1 *h2, Double_t &advalue) const;
   virtual Double_t KolmogorovTest(const TH1 *h2, Option_t *option="") const;
   virtual void     LabelsDeflate(Option_t *axis="X");
   virtual void     LabelsInflate(Option_t *axis="X");
   virtual void     LabelsOption(Option_t *option="h", Option_t *axis="X");
   virtual Long64_t Merge(TCollection *list) { return Merge(list,""); }
           Long64_t Merge(TCollection *list, Option_t * option);
   virtual Bool_t   Multiply(TF1 *f1, Double_t c1=1);
   virtual Bool_t   Multiply(const TH1 *h1);
   virtual Bool_t   Multiply(const TH1 *h1, const TH1 *h2, Double_t c1=1, Double_t c2=1, Option_t *option="");
   virtual void     Normalize(Option_t *option=""); // *MENU*
           void     Paint(Option_t *option = "") override;
           void     Print(Option_t *option = "") const override;
   virtual void     PutStats(Double_t *stats);
   virtual TH1     *Rebin(Int_t ngroup = 2, const char *newname = "", const Double_t *xbins = nullptr);  // *MENU*
   virtual TH1     *RebinX(Int_t ngroup = 2, const char *newname = "") { return Rebin(ngroup,newname, (Double_t*) nullptr); }
   virtual void     Rebuild(Option_t *option = "");
           void     RecursiveRemove(TObject *obj) override;
   virtual void     Reset(Option_t *option = "");
   virtual void     ResetStats();
           void     SaveAs(const char *filename = "hist", Option_t *option = "") const override;  // *MENU*
           void     SavePrimitive(std::ostream &out, Option_t *option = "") override;
   virtual void     Scale(Double_t c1=1, Option_t *option="");  // *MENU*
   virtual void     SetAxisColor(Color_t color=1, Option_t *axis="X");
   virtual void     SetAxisRange(Double_t xmin, Double_t xmax, Option_t *axis="X");
   virtual void     SetBarOffset(Float_t offset=0.25) {fBarOffset = Short_t(1000*offset);}
   virtual void     SetBarWidth(Float_t width=0.5) {fBarWidth = Short_t(1000*width);}
   virtual void     SetBinContent(Int_t bin, Double_t content);
   virtual void     SetBinContent(Int_t bin, Int_t, Double_t content) { SetBinContent(bin, content); }
   virtual void     SetBinContent(Int_t bin, Int_t, Int_t, Double_t content) { SetBinContent(bin, content); }
   virtual void     SetBinError(Int_t bin, Double_t error);
   virtual void     SetBinError(Int_t binx, Int_t biny, Double_t error);
   virtual void     SetBinError(Int_t binx, Int_t biny, Int_t binz, Double_t error);
   virtual void     SetBins(Int_t nx, Double_t xmin, Double_t xmax);
   virtual void     SetBins(Int_t nx, const Double_t *xBins);
   virtual void     SetBins(Int_t nx, Double_t xmin, Double_t xmax, Int_t ny, Double_t ymin, Double_t ymax);
   virtual void     SetBins(Int_t nx, const Double_t *xBins, Int_t ny, const Double_t *yBins);
   virtual void     SetBins(Int_t nx, Double_t xmin, Double_t xmax, Int_t ny, Double_t ymin, Double_t ymax,
                            Int_t nz, Double_t zmin, Double_t zmax);
   virtual void     SetBins(Int_t nx, const Double_t *xBins, Int_t ny, const Double_t * yBins, Int_t nz,
                            const Double_t *zBins);
   virtual void     SetBinsLength(Int_t = -1) { } //redefined in derived classes
   virtual void     SetBinErrorOption(EBinErrorOpt type) { fBinStatErrOpt = type; }
   virtual void     SetBuffer(Int_t bufsize, Option_t *option="");
   virtual UInt_t   SetCanExtend(UInt_t extendBitMask);
   virtual void     SetContent(const Double_t *content);
   virtual void     SetContour(Int_t nlevels, const Double_t *levels = nullptr);
   virtual void     SetContourLevel(Int_t level, Double_t value);
   virtual void     SetColors(Color_t linecolor = -1, Color_t markercolor = -1, Color_t fillcolor = -1);
   static  void     SetDefaultBufferSize(Int_t bufsize=1000);
   static  void     SetDefaultSumw2(Bool_t sumw2=kTRUE);
   virtual void     SetDirectory(TDirectory *dir);
   virtual void     SetEntries(Double_t n) { fEntries = n; }
   virtual void     SetError(const Double_t *error);
   virtual void     SetHighlight(Bool_t set = kTRUE); // *TOGGLE* *GETTER=IsHighlight
   virtual void     SetLabelColor(Color_t color=1, Option_t *axis="X");
   virtual void     SetLabelFont(Style_t font=62, Option_t *axis="X");
   virtual void     SetLabelOffset(Float_t offset=0.005, Option_t *axis="X");
   virtual void     SetLabelSize(Float_t size=0.02, Option_t *axis="X");

   /*
    * Set the minimum / maximum value for the Y axis (1-D histograms) or Z axis (2-D histograms)
    *   By default the maximum / minimum value used in drawing is the maximum / minimum value of the histogram
    * plus a margin of 10%. If these functions are called, the values are used without any extra margin.
    */
   virtual void     SetMaximum(Double_t maximum = -1111) { fMaximum = maximum; } // *MENU*
   virtual void     SetMinimum(Double_t minimum = -1111) { fMinimum = minimum; } // *MENU*

           void     SetName(const char *name) override; // *MENU*
           void     SetNameTitle(const char *name, const char *title) override;
   virtual void     SetNdivisions(Int_t n=510, Option_t *axis="X");
   virtual void     SetNormFactor(Double_t factor=1) {fNormFactor = factor;}
   virtual void     SetStats(Bool_t stats=kTRUE); // *MENU*
   virtual void     SetOption(Option_t *option=" ") {fOption = option;}
   virtual void     SetTickLength(Float_t length=0.02, Option_t *axis="X");
   virtual void     SetTitleFont(Style_t font=62, Option_t *axis="X");
   virtual void     SetTitleOffset(Float_t offset=1, Option_t *axis="X");
   virtual void     SetTitleSize(Float_t size=0.02, Option_t *axis="X");
           void     SetStatOverflows(EStatOverflows statOverflows) { fStatOverflows = statOverflows; } ///< See GetStatOverflows for more information.
           void     SetTitle(const char *title) override;  // *MENU*
   virtual void     SetXTitle(const char *title) {fXaxis.SetTitle(title);}
   virtual void     SetYTitle(const char *title) {fYaxis.SetTitle(title);}
   virtual void     SetZTitle(const char *title) {fZaxis.SetTitle(title);}
   virtual TH1     *ShowBackground(Int_t niter=20, Option_t *option="same"); // *MENU*
   virtual Int_t    ShowPeaks(Double_t sigma=2, Option_t *option="", Double_t threshold=0.05); // *MENU*
   virtual void     Smooth(Int_t ntimes=1, Option_t *option=""); // *MENU*
   static  void     SmoothArray(Int_t NN, Double_t *XX, Int_t ntimes=1);
   static  void     StatOverflows(Bool_t flag=kTRUE);
   virtual void     Sumw2(Bool_t flag = kTRUE);
           void     UseCurrentStyle() override;
   static  TH1     *TransformHisto(TVirtualFFT *fft, TH1* h_output,  Option_t *option);

   static void      SavePrimitiveFunctions(std::ostream &out, const char *varname, TList *lst);

   // TODO: Remove obsolete methods in v6-04
   virtual Double_t GetCellContent(Int_t binx, Int_t biny) const
                        { Obsolete("GetCellContent", "v6-00", "v6-04"); return GetBinContent(GetBin(binx, biny)); }
   virtual Double_t GetCellError(Int_t binx, Int_t biny) const
                        { Obsolete("GetCellError", "v6-00", "v6-04"); return GetBinError(binx, biny); }
   virtual void     RebinAxis(Double_t x, TAxis *axis)
                        { Obsolete("RebinAxis", "v6-00", "v6-04"); ExtendAxis(x, axis); }
   virtual void     SetCellContent(Int_t binx, Int_t biny, Double_t content)
                        { Obsolete("SetCellContent", "v6-00", "v6-04"); SetBinContent(GetBin(binx, biny), content); }
   virtual void     SetCellError(Int_t binx, Int_t biny, Double_t content)
                        { Obsolete("SetCellError", "v6-00", "v6-04"); SetBinError(binx, biny, content); }

   ClassDefOverride(TH1,8)  //1-Dim histogram base class

protected:

   /// Raw retrieval of bin content on internal data structure
   /// see convention for numbering bins in TH1::GetBin
   virtual Double_t RetrieveBinContent(Int_t bin) const = 0;

   /// Raw update of bin content on internal data structure
   /// see convention for numbering bins in TH1::GetBin
   virtual void     UpdateBinContent(Int_t bin, Double_t content) = 0;

   virtual Double_t GetBinErrorSqUnchecked(Int_t bin) const { return fSumw2.fN ? fSumw2.fArray[bin] : RetrieveBinContent(bin); }
};

namespace cling {
  std::string printValue(TH1 *val);
}

//________________________________________________________________________

class TH1C : public TH1, public TArrayC {

public:
   TH1C();
   TH1C(const char *name,const char *title,Int_t nbinsx,Double_t xlow,Double_t xup);
   TH1C(const char *name,const char *title,Int_t nbinsx,const Float_t  *xbins);
   TH1C(const char *name,const char *title,Int_t nbinsx,const Double_t *xbins);
   TH1C(const TH1C &h1c);
   TH1C& operator=(const TH1C &h1);
   ~TH1C() override;

   void     AddBinContent(Int_t bin) override;
   void     AddBinContent(Int_t bin, Double_t w) override;
   void     Copy(TObject &hnew) const override;
   void     Reset(Option_t *option="") override;
   void     SetBinsLength(Int_t n=-1) override;

   ClassDefOverride(TH1C,3)  //1-Dim histograms (one char per channel)

   friend  TH1C     operator*(Double_t c1, const TH1C &h1);
   friend  TH1C     operator*(const TH1C &h1, Double_t c1);
   friend  TH1C     operator+(const TH1C &h1, const TH1C &h2);
   friend  TH1C     operator-(const TH1C &h1, const TH1C &h2);
   friend  TH1C     operator*(const TH1C &h1, const TH1C &h2);
   friend  TH1C     operator/(const TH1C &h1, const TH1C &h2);

protected:
   Double_t RetrieveBinContent(Int_t bin) const override { return Double_t (fArray[bin]); }
   void     UpdateBinContent(Int_t bin, Double_t content) override { fArray[bin] = Char_t (content); }
};

TH1C operator*(Double_t c1, const TH1C &h1);
inline
TH1C operator*(const TH1C &h1, Double_t c1) {return operator*(c1,h1);}
TH1C operator+(const TH1C &h1, const TH1C &h2);
TH1C operator-(const TH1C &h1, const TH1C &h2);
TH1C operator*(const TH1C &h1, const TH1C &h2);
TH1C operator/(const TH1C &h1, const TH1C &h2);

//________________________________________________________________________

class TH1S : public TH1, public TArrayS {

public:
   TH1S();
   TH1S(const char *name,const char *title,Int_t nbinsx,Double_t xlow,Double_t xup);
   TH1S(const char *name,const char *title,Int_t nbinsx,const Float_t  *xbins);
   TH1S(const char *name,const char *title,Int_t nbinsx,const Double_t *xbins);
   TH1S(const TH1S &h1s);
   TH1S& operator=(const TH1S &h1);
   ~TH1S() override;

   void     AddBinContent(Int_t bin) override;
   void     AddBinContent(Int_t bin, Double_t w) override;
   void     Copy(TObject &hnew) const override;
   void     Reset(Option_t *option="") override;
   void     SetBinsLength(Int_t n=-1) override;

   ClassDefOverride(TH1S,3)  //1-Dim histograms (one short per channel)

   friend  TH1S     operator*(Double_t c1, const TH1S &h1);
   friend  TH1S     operator*(const TH1S &h1, Double_t c1);
   friend  TH1S     operator+(const TH1S &h1, const TH1S &h2);
   friend  TH1S     operator-(const TH1S &h1, const TH1S &h2);
   friend  TH1S     operator*(const TH1S &h1, const TH1S &h2);
   friend  TH1S     operator/(const TH1S &h1, const TH1S &h2);

protected:
   Double_t RetrieveBinContent(Int_t bin) const override { return Double_t (fArray[bin]); }
   void     UpdateBinContent(Int_t bin, Double_t content) override { fArray[bin] = Short_t (content); }
};

TH1S operator*(Double_t c1, const TH1S &h1);
inline
TH1S operator*(const TH1S &h1, Double_t c1) {return operator*(c1,h1);}
TH1S operator+(const TH1S &h1, const TH1S &h2);
TH1S operator-(const TH1S &h1, const TH1S &h2);
TH1S operator*(const TH1S &h1, const TH1S &h2);
TH1S operator/(const TH1S &h1, const TH1S &h2);

//________________________________________________________________________

class TH1I: public TH1, public TArrayI {

public:
   TH1I();
   TH1I(const char *name,const char *title,Int_t nbinsx,Double_t xlow,Double_t xup);
   TH1I(const char *name,const char *title,Int_t nbinsx,const Float_t  *xbins);
   TH1I(const char *name,const char *title,Int_t nbinsx,const Double_t *xbins);
   TH1I(const TH1I &h1i);
   TH1I& operator=(const TH1I &h1);
   ~TH1I() override;

   void     AddBinContent(Int_t bin) override;
   void     AddBinContent(Int_t bin, Double_t w) override;
   void     Copy(TObject &hnew) const override;
   void     Reset(Option_t *option="") override;
   void     SetBinsLength(Int_t n=-1) override;

   ClassDefOverride(TH1I,3)  //1-Dim histograms (one 32 bit integer per channel)

   friend  TH1I     operator*(Double_t c1, const TH1I &h1);
   friend  TH1I     operator*(const TH1I &h1, Double_t c1);
   friend  TH1I     operator+(const TH1I &h1, const TH1I &h2);
   friend  TH1I     operator-(const TH1I &h1, const TH1I &h2);
   friend  TH1I     operator*(const TH1I &h1, const TH1I &h2);
   friend  TH1I     operator/(const TH1I &h1, const TH1I &h2);

protected:
   Double_t RetrieveBinContent(Int_t bin) const override { return Double_t (fArray[bin]); }
   void     UpdateBinContent(Int_t bin, Double_t content) override { fArray[bin] = Int_t (content); }
};

TH1I operator*(Double_t c1, const TH1I &h1);
inline
TH1I operator*(const TH1I &h1, Double_t c1) {return operator*(c1,h1);}
TH1I operator+(const TH1I &h1, const TH1I &h2);
TH1I operator-(const TH1I &h1, const TH1I &h2);
TH1I operator*(const TH1I &h1, const TH1I &h2);
TH1I operator/(const TH1I &h1, const TH1I &h2);

//________________________________________________________________________

class TH1L: public TH1, public TArrayL64 {

public:
   TH1L();
   TH1L(const char *name,const char *title,Int_t nbinsx,Double_t xlow,Double_t xup);
   TH1L(const char *name,const char *title,Int_t nbinsx,const Float_t  *xbins);
   TH1L(const char *name,const char *title,Int_t nbinsx,const Double_t *xbins);
   TH1L(const TH1L &h1l);
   TH1L& operator=(const TH1L &h1);
   ~TH1L() override;

   void     AddBinContent(Int_t bin) override;
   void     AddBinContent(Int_t bin, Double_t w) override;
   void     Copy(TObject &hnew) const override;
   void     Reset(Option_t *option="") override;
   void     SetBinsLength(Int_t n=-1) override;

   ClassDefOverride(TH1L,0)  //1-Dim histograms (one 64 bit integer per channel)

   friend  TH1L     operator*(Double_t c1, const TH1L &h1);
   friend  TH1L     operator*(const TH1L &h1, Double_t c1);
   friend  TH1L     operator+(const TH1L &h1, const TH1L &h2);
   friend  TH1L     operator-(const TH1L &h1, const TH1L &h2);
   friend  TH1L     operator*(const TH1L &h1, const TH1L &h2);
   friend  TH1L     operator/(const TH1L &h1, const TH1L &h2);

protected:
   Double_t RetrieveBinContent(Int_t bin) const override { return Double_t (fArray[bin]); }
   void     UpdateBinContent(Int_t bin, Double_t content) override { fArray[bin] = Int_t (content); }
};

TH1L operator*(Double_t c1, const TH1L &h1);
inline
TH1L operator*(const TH1L &h1, Double_t c1) {return operator*(c1,h1);}
TH1L operator+(const TH1L &h1, const TH1L &h2);
TH1L operator-(const TH1L &h1, const TH1L &h2);
TH1L operator*(const TH1L &h1, const TH1L &h2);
TH1L operator/(const TH1L &h1, const TH1L &h2);

//________________________________________________________________________

class TH1F : public TH1, public TArrayF {

public:
   TH1F();
   TH1F(const char *name,const char *title,Int_t nbinsx,Double_t xlow,Double_t xup);
   TH1F(const char *name,const char *title,Int_t nbinsx,const Float_t  *xbins);
   TH1F(const char *name,const char *title,Int_t nbinsx,const Double_t *xbins);
   explicit TH1F(const TVectorF &v);
   TH1F(const TH1F &h1f);
   TH1F& operator=(const TH1F &h1);
   ~TH1F() override;

   /// Increment bin content by 1.
   /// Passing an out-of-range bin leads to undefined behavior
   void     AddBinContent(Int_t bin) override {++fArray[bin];}
   /// Increment bin content by a weight w.
   /// \warning The value of w is cast to `Float_t` before being added.
   /// Passing an out-of-range bin leads to undefined behavior
   void     AddBinContent(Int_t bin, Double_t w) override
                          { fArray[bin] += Float_t (w); }
   void     Copy(TObject &hnew) const override;
   void     Reset(Option_t *option = "") override;
   void     SetBinsLength(Int_t n=-1) override;

   ClassDefOverride(TH1F,3)  //1-Dim histograms (one float per channel)

   friend  TH1F     operator*(Double_t c1, const TH1F &h1);
   friend  TH1F     operator*(const TH1F &h1, Double_t c1);
   friend  TH1F     operator+(const TH1F &h1, const TH1F &h2);
   friend  TH1F     operator-(const TH1F &h1, const TH1F &h2);
   friend  TH1F     operator*(const TH1F &h1, const TH1F &h2);
   friend  TH1F     operator/(const TH1F &h1, const TH1F &h2);

protected:
   Double_t RetrieveBinContent(Int_t bin) const override { return Double_t (fArray[bin]); }
   void     UpdateBinContent(Int_t bin, Double_t content) override { fArray[bin] = Float_t (content); }
};

TH1F operator*(Double_t c1, const TH1F &h1);
inline
TH1F operator*(const TH1F &h1, Double_t c1) {return operator*(c1,h1);}
TH1F operator+(const TH1F &h1, const TH1F &h2);
TH1F operator-(const TH1F &h1, const TH1F &h2);
TH1F operator*(const TH1F &h1, const TH1F &h2);
TH1F operator/(const TH1F &h1, const TH1F &h2);

//________________________________________________________________________

class TH1D : public TH1, public TArrayD {

public:
   TH1D();
   TH1D(const char *name,const char *title,Int_t nbinsx,Double_t xlow,Double_t xup);
   TH1D(const char *name,const char *title,Int_t nbinsx,const Float_t  *xbins);
   TH1D(const char *name,const char *title,Int_t nbinsx,const Double_t *xbins);
   explicit TH1D(const TVectorD &v);
   TH1D(const TH1D &h1d);
   TH1D& operator=(const TH1D &h1);
   ~TH1D() override;

   /// Increment bin content by 1.
   /// Passing an out-of-range bin leads to undefined behavior
   void     AddBinContent(Int_t bin) override {++fArray[bin];}
   /// Increment bin content by a weight w
   /// Passing an out-of-range bin leads to undefined behavior
   void     AddBinContent(Int_t bin, Double_t w) override
                          {fArray[bin] += Double_t (w);}
   void     Copy(TObject &hnew) const override;
   void     Reset(Option_t *option = "") override;
   void     SetBinsLength(Int_t n=-1) override;

   ClassDefOverride(TH1D,3)  //1-Dim histograms (one double per channel)

   friend  TH1D     operator*(Double_t c1, const TH1D &h1);
   friend  TH1D     operator*(const TH1D &h1, Double_t c1);
   friend  TH1D     operator+(const TH1D &h1, const TH1D &h2);
   friend  TH1D     operator-(const TH1D &h1, const TH1D &h2);
   friend  TH1D     operator*(const TH1D &h1, const TH1D &h2);
   friend  TH1D     operator/(const TH1D &h1, const TH1D &h2);

protected:
   Double_t RetrieveBinContent(Int_t bin) const override { return fArray[bin]; }
   void     UpdateBinContent(Int_t bin, Double_t content) override { fArray[bin] = content; }
};

TH1D operator*(Double_t c1, const TH1D &h1);
inline
TH1D operator*(const TH1D &h1, Double_t c1) {return operator*(c1,h1);}
TH1D operator+(const TH1D &h1, const TH1D &h2);
TH1D operator-(const TH1D &h1, const TH1D &h2);
TH1D operator*(const TH1D &h1, const TH1D &h2);
TH1D operator/(const TH1D &h1, const TH1D &h2);

   extern TH1 *R__H(Int_t hid);
   extern TH1 *R__H(const char *hname);

#endif
