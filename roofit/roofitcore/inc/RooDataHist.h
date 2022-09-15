/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooDataHist.h,v 1.37 2007/05/11 09:11:30 verkerke Exp $
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/
#ifndef ROO_DATA_HIST
#define ROO_DATA_HIST

#include "RooAbsData.h"
#include "RooDirItem.h"
#include "RooArgSet.h"

#include "ROOT/RStringView.hxx"
#include "Rtypes.h"

#include <map>
#include <vector>
#include <string>
#include <functional>
#include <memory>
#include <unordered_map>

class TAxis ;
class TObject ;
class RooAbsArg;
class RooCategory ;
class RooPlot;
class RooAbsLValue ;
namespace RooFit {
namespace TestStatistics {
class RooAbsL;
}
}

class RooDataHist : public RooAbsData, public RooDirItem {
public:

  // Constructors, factory methods etc.
  RooDataHist() ;
  RooDataHist(RooStringView name, RooStringView title, const RooArgSet& vars, const char* binningName=nullptr) ;
  RooDataHist(RooStringView name, RooStringView title, const RooArgSet& vars, const RooAbsData& data, double initWgt=1.0) ;
  RooDataHist(RooStringView name, RooStringView title, const RooArgList& vars, const TH1* hist, double initWgt=1.0) ;
  RooDataHist(RooStringView name, RooStringView title, const RooArgList& vars, RooCategory& indexCat, std::map<std::string,TH1*> histMap, double initWgt=1.0) ;
  RooDataHist(RooStringView name, RooStringView title, const RooArgList& vars, RooCategory& indexCat, std::map<std::string,RooDataHist*> dhistMap, double wgt=1.0) ;
  //RooDataHist(const char *name, const char *title, const RooArgList& vars, double initWgt=1.0) ;
  RooDataHist(RooStringView name, RooStringView title, const RooArgList& vars, const RooCmdArg& arg1, const RooCmdArg& arg2=RooCmdArg(), const RooCmdArg& arg3=RooCmdArg(),
        const RooCmdArg& arg4=RooCmdArg(),const RooCmdArg& arg5=RooCmdArg(),const RooCmdArg& arg6=RooCmdArg(),const RooCmdArg& arg7=RooCmdArg(),const RooCmdArg& arg8=RooCmdArg()) ;
  RooDataHist& operator=(const RooDataHist&) = delete;

  RooDataHist(const RooDataHist& other, const char* newname = nullptr) ;
  TObject* Clone(const char* newname="") const override {
    return new RooDataHist(*this, newname && newname[0] != '\0' ? newname : GetName());
  }
  ~RooDataHist() override ;

  /// Return empty clone of this RooDataHist.
  RooAbsData* emptyClone(const char* newName=nullptr, const char* newTitle=nullptr, const RooArgSet*vars=nullptr, const char* /*wgtVarName*/=nullptr) const override {
    return new RooDataHist(newName?newName:GetName(),newTitle?newTitle:GetTitle(),vars?*vars:*get()) ;
  }

  /// Add `wgt` to the bin content enclosed by the coordinates passed in `row`.
  virtual void add(const RooArgSet& row, double wgt=1.0) { add(row,wgt,-1.); }
  void add(const RooArgSet& row, double weight, double sumw2) override ;
  void set(std::size_t binNumber, double weight, double wgtErr);
  void set(const RooArgSet& row, double weight, double wgtErr=-1.) ;
  void set(const RooArgSet& row, double weight, double wgtErrLo, double wgtErrHi) ;

  void add(const RooAbsData& dset, const RooFormulaVar* cutVar=nullptr, double weight=1.0 ) ;
  void add(const RooAbsData& dset, const char* cut, double weight=1.0 ) ;

  /// Get bin centre of current bin.
  const RooArgSet* get() const override { return &_vars; }
  const RooArgSet* get(Int_t binNumber) const override;
  virtual const RooArgSet* get(const RooArgSet& coord) const;
  Int_t numEntries() const override;
  double sumEntries() const override;
  double sumEntries(const char* cutSpec, const char* cutRange=nullptr) const override;

  /// Always returns true as all histograms use event weights.
  bool isWeighted() const override { return true; }
  bool isNonPoissonWeighted() const override ;

  RooSpan<const double> getWeightBatch(std::size_t first, std::size_t len, bool sumW2=false) const override;

  /// Retrieve all bin volumes. Bins are indexed according to getIndex().
  RooSpan<const double> binVolumes(std::size_t first, std::size_t len) const {
    return {_binv + first, len};
  }

  double sum(bool correctForBinSize, bool inverseCorr=false) const ;
  double sum(const RooArgSet& sumSet, const RooArgSet& sliceSet, bool correctForBinSize, bool inverseCorr=false) ;
  double sum(const RooArgSet& sumSet,
               const RooArgSet& sliceSet,
               bool correctForBinSize,
               bool inverseCorr,
               const std::map<const RooAbsArg*, std::pair<double, double> >& ranges,
               std::function<double(int)> getBinScale = [](int){ return 1.0; } );

  void weights(double* output, RooSpan<double const> xVals, int intOrder, bool correctForBinSize, bool cdfBoundaries);
  /// Return weight of i-th bin. \see getIndex()
  double weight(std::size_t i) const { return _wgt[i]; }
  double weightFast(const RooArgSet& bin, int intOrder, bool correctForBinSize, bool cdfBoundaries);
  double weight(const RooArgSet& bin, Int_t intOrder=1, bool correctForBinSize=false, bool cdfBoundaries=false, bool oneSafe=false);
  /// Return squared weight sum of i-th bin. \see getIndex()
  double weightSquared(std::size_t i) const { return get_sumw2(i); }
  /// Return bin volume of i-th bin. \see getIndex()
  double binVolume(std::size_t i) const { return _binv[i]; }
  double binVolume(const RooArgSet& bin) const;

  TIterator* sliceIterator(RooAbsArg& sliceArg, const RooArgSet& otherArgs) ;

  void weightError(double& lo, double& hi, ErrorType etype=Poisson) const override;
  /// \copydoc RooAbsData::weightError(RooAbsData::ErrorType) const
  double weightError(ErrorType etype=Poisson) const override {
    // Return symmetric error on current bin calculated either from Poisson statistics or from SumOfWeights
    double lo,hi ;
    weightError(lo,hi,etype) ;
    return (lo+hi)/2 ;
  }

  using RooAbsData::plotOn ;
  RooPlot *plotOn(RooPlot *frame, PlotOpt o) const override;

  void reset() override;

  void printMultiline(std::ostream& os, Int_t content, bool verbose=false, TString indent="") const override;
  void printArgs(std::ostream& os) const override;
  void printValue(std::ostream& os) const override;
  void printDataHistogram(std::ostream& os, RooRealVar* obs) const;

  void SetName(const char *name) override;
  void SetNameTitle(const char *name, const char* title) override;

  Int_t getIndex(const RooAbsCollection& coord, bool fast = false) const;
  /// \copydoc getIndex(const RooAbsCollection&,bool) const
  ///
  /// \note This overload only exists because there is an implicit conversion from RooAbsArg
  /// to RooArgSet, and this needs to remain supported. This enables code like
  /// ```
  /// RooRealVar x(...);
  /// dataHist.getIndex(x);
  /// ```
  /// It is, however, recommended to use
  /// ```
  /// dataHist.getIndex(RooArgSet(x));
  /// ```
  /// in this case.
  Int_t getIndex(const RooArgSet& coord, bool fast = false) const {
    return getIndex(static_cast<const RooAbsCollection&>(coord), fast);
  }

  void removeSelfFromDir() { removeFromDir(this) ; }


  ////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /// @name Deprecated functions
  /// These functions rely on the fact that an event has been loaded before they are called. It is advised
  /// to switch to their counterparts that take bin numbers as arguments. In this way, code like,
  /// ```
  ///   const RooArgSet* coordinates = dataHist.get(i); // Need this to achieve side effect on next call of weight() - bad.
  ///   const double weight = dataHist.weight();
  ///   processEvent(coordinates, weight);
  /// ```
  /// becomes
  /// ```
  ///   processEvent(dataHist.get(i), dataHist.weight(i));
  /// ```
  /// The index of a set of coordinates can be computed using getIndex().
  /// @{

  /// Return weight of last bin that was requested with get().
  /// \deprecated Use the safer weight(std::size_t) const.
  double weight() const override
  R__SUGGEST_ALTERNATIVE("Use the safer weight(std::size_t) const.")
  { return get_curWeight(); }
  /// Return squared weight of last bin that was requested with get().
  /// \deprecated Use the safer weightSquared(std::size_t) const.
  double weightSquared() const override
  R__SUGGEST_ALTERNATIVE("Use the safer weightSquared(std::size_t) const.")
  { return get_curSumW2(); }
  /// Return volume of current bin. \deprecated Use binVolume(std::size_t) const.
  double binVolume() const
  R__SUGGEST_ALTERNATIVE("Use binVolume(std::size_t) const.")
  { return _binv[_curIndex]; }
  /// Write `weight` into current bin. \deprecated Use set(std::size_t,double,double)
  void set(double wgt, double wgtErr=-1)
  R__SUGGEST_ALTERNATIVE("Use set(std::size_t,double,double).");

  void dump2();

  ///@}
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////

  /// Structure to cache information on the histogram variable that is
  /// frequently used for histogram weights retrieval.
  struct VarInfo {
    size_t nRealVars = 0;
    size_t realVarIdx1 = 0;
    size_t realVarIdx2 = 0;
    bool initialized = false;
  };

  std::vector<std::unique_ptr<const RooAbsBinning>> const& getBinnings() const { return _lvbins; }

  double const* weightArray()   const { return _wgt; }
  double const* wgtErrLoArray() const { return _errLo; }
  double const* wgtErrHiArray() const { return _errHi; }
  double const* sumW2Array()    const { return _sumw2; }

protected:

  friend class RooAbsCachedPdf ;
  friend class RooAbsCachedReal ;
  friend class RooDataHistSliceIter ;
  // for access into copied dataset:
  friend class RooFit::TestStatistics::RooAbsL;

  std::size_t calcTreeIndex(const RooAbsCollection& coords, bool fast) const;
  /// Legacy overload to calculate the tree index from the current value of `_vars`.
  /// \deprecated Use calcTreeIndex(const RooArgSet&,bool) const.
  Int_t calcTreeIndex() const { return static_cast<Int_t>(calcTreeIndex(_vars, true)); }

  void setAllWeights(double value) ;

  void initialize(const char* binningName=nullptr,bool fillTree=true) ;
  RooDataHist(RooStringView name, RooStringView title, RooDataHist* h, const RooArgSet& varSubset,
        const RooFormulaVar* cutVar, const char* cutRange, Int_t nStart, Int_t nStop) ;
  RooAbsData* reduceEng(const RooArgSet& varSubset, const RooFormulaVar* cutVar, const char* cutRange=nullptr,
                  std::size_t nStart=0, std::size_t nStop=std::numeric_limits<std::size_t>::max()) override;
  double interpolateDim(int iDim, double xval, size_t centralIdx, int intOrder, bool correctForBinSize, bool cdfBoundaries) ;
  const std::vector<double>& calculatePartialBinVolume(const RooArgSet& dimSet) const ;
  void checkBinBounds() const;

  void adjustBinning(const RooArgList& vars, const TH1& href, Int_t* offset=nullptr) ;
  void importTH1(const RooArgList& vars, const TH1& histo, double initWgt, bool doDensityCorrection) ;
  void importTH1Set(const RooArgList& vars, RooCategory& indexCat, std::map<std::string,TH1*> hmap, double initWgt, bool doDensityCorrection) ;
  void importDHistSet(const RooArgList& vars, RooCategory& indexCat, std::map<std::string,RooDataHist*> dmap, double initWgt) ;

  double get_wgt(std::size_t idx)   const { return _wgt[idx]; }
  double get_errLo(std::size_t idx) const { return _errLo ? _errLo[idx] : -1.; }
  double get_errHi(std::size_t idx) const { return _errHi ? _errHi[idx] : -1.; }
  // If sumw2 is not being tracked, assume that all previous fill operations had a weight of 1, i.e., return the bare weight of the bin.
  double get_sumw2(std::size_t idx) const { return _sumw2 ? _sumw2[idx] : _wgt[idx]; }

  double get_curWeight()   const { return get_wgt(_curIndex); }
  double get_curWgtErrLo() const { return get_errLo(_curIndex); }
  double get_curWgtErrHi() const { return get_errHi(_curIndex); }
  double get_curSumW2()    const { return get_sumw2(_curIndex); }

  Int_t get_curIndex() const { return _curIndex; }

  Int_t _arrSize{0}; // Size of member arrays.
  std::vector<Int_t> _idxMult ; // Multiplier jump table for index calculation

  double*         _wgt  {nullptr}; ///<[_arrSize] Weight array
  mutable double* _errLo{nullptr}; ///<[_arrSize] Low-side error on weight array
  mutable double* _errHi{nullptr}; ///<[_arrSize] High-side error on weight array
  mutable double* _sumw2{nullptr}; ///<[_arrSize] Sum of weights^2
  double*         _binv {nullptr}; ///<[_arrSize] Bin volume array

  mutable ULong64_t _curIndex{std::numeric_limits<ULong64_t>::max()}; ///< Current index

  mutable std::unordered_map<int,std::vector<double>> _pbinvCache ; ///<! Cache for arrays of partial bin volumes
  std::vector<RooAbsLValue*> _lvvars ; ///<! List of observables casted as RooAbsLValue
  std::vector<std::unique_ptr<const RooAbsBinning>> _lvbins ; ///<! List of used binnings associated with lvalues
  mutable std::vector<std::vector<double> > _binbounds;     ///<! list of bin bounds per dimension

  enum CacheSumState_t{kInvalid = 0, kNoBinCorrection = 1, kCorrectForBinSize = 2, kInverseBinCorr = 3};
  mutable Int_t _cache_sum_valid{kInvalid}; ///<! Is cache sum valid? Needs to be Int_t instead of CacheSumState_t for subclasses.
  mutable double _cache_sum{0.};          ///<! Cache for sum of entries ;

private:

  void interpolateQuadratic(double* output, RooSpan<const double> xVals, bool correctForBinSize, bool cdfBoundaries);
  void interpolateLinear(double* output, RooSpan<const double> xVals, bool correctForBinSize, bool cdfBoundaries);
  double weightInterpolated(const RooArgSet& bin, int intOrder, bool correctForBinSize, bool cdfBoundaries);

  void _adjustBinning(RooRealVar &theirVar, const TAxis &axis, RooRealVar *ourVar, Int_t *offset);
  void registerWeightArraysToDataStore() const;
  VarInfo const& getVarInfo();

  VarInfo _varInfo; ///<!
  std::vector<double> _interpolationBuffer; ///<! Buffer to contain values used for weight interpolation

  ClassDefOverride(RooDataHist, 8) // Binned data set
};

#endif

