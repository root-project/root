/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooAbsData.h,v 1.33 2007/07/16 21:04:28 wouter Exp $
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
#ifndef ROO_ABS_DATA
#define ROO_ABS_DATA

#include "RooPrintable.h"
#include "RooAbsCategory.h"
#include "RooArgSet.h"
#include "RooArgList.h"
#include "RooSpan.h"
#include "RooNameReg.h"
#include "RooFit/Detail/DataMap.h"

#include "ROOT/RStringView.hxx"
#include "TNamed.h"

#include <map>
#include <string>

class RooAbsArg;
class RooAbsReal ;
class RooRealVar;
class RooAbsRealLValue;
class RooAbsCategoryLValue;
class Roo1DTable ;
class RooPlot;
class RooArgList;
class RooSimultaneous;
class TH1;
class TH2F;
class RooAbsBinning ;
class Roo1DTable ;
class RooAbsDataStore ;
template<typename T> class TMatrixTSym;
using TMatrixDSym = TMatrixTSym<Double_t>;
class RooFormulaVar;
namespace RooBatchCompute{
struct RunContext;
}
namespace RooFit {
namespace TestStatistics {
class RooAbsL;
struct ConstantTermsOptimizer;
}
}


// Writes a templated constructor for compatibility with ROOT builds using the
// C++14 standard or earlier, taking `ROOT::Internal::TStringView` instead of
// `std::string_view`. This means one can still use a `TString` for the name or
// title parameter. The condition in the following `#if` should be kept in
// sync with the one in TString.h.
#if (__cplusplus >= 201700L) && !defined(_MSC_VER) && (!defined(__clang_major__) || __clang_major__ > 5)
#define WRITE_TSTRING_COMPATIBLE_CONSTRUCTOR(Class_t) // does nothing
#else
#define WRITE_TSTRING_COMPATIBLE_CONSTRUCTOR(Class_t)                                             \
  template<typename ...Args_t>                                                                    \
  Class_t(ROOT::Internal::TStringView name, ROOT::Internal::TStringView title, Args_t &&... args) \
    : Class_t(std::string_view(name), std::string_view(title), std::forward<Args_t>(args)...) {}  \
  template<typename ...Args_t>                                                                    \
  Class_t(ROOT::Internal::TStringView name, std::string_view title, Args_t &&... args)            \
    : Class_t(std::string_view(name), title, std::forward<Args_t>(args)...) {}                    \
  template<typename ...Args_t>                                                                    \
  Class_t(std::string_view name, ROOT::Internal::TStringView title, Args_t &&... args)            \
    : Class_t(name, std::string_view(title), std::forward<Args_t>(args)...) {}
#endif


class RooAbsData : public TNamed, public RooPrintable {
public:

  // Constructors, factory methods etc.
  RooAbsData() ; 
  RooAbsData(std::string_view name, std::string_view title, const RooArgSet& vars, RooAbsDataStore* store=0) ;
  RooAbsData(const RooAbsData& other, const char* newname = 0) ;

  WRITE_TSTRING_COMPATIBLE_CONSTRUCTOR(RooAbsData)

  RooAbsData& operator=(const RooAbsData& other);
  virtual ~RooAbsData() ;
  virtual RooAbsData* emptyClone(const char* newName=0, const char* newTitle=0, const RooArgSet* vars=0, const char* wgtVarName=0) const = 0 ;

  // Reduction methods
  RooAbsData* reduce(const RooCmdArg& arg1,const RooCmdArg& arg2=RooCmdArg(),const RooCmdArg& arg3=RooCmdArg(),const RooCmdArg& arg4=RooCmdArg(),
                     const RooCmdArg& arg5=RooCmdArg(),const RooCmdArg& arg6=RooCmdArg(),const RooCmdArg& arg7=RooCmdArg(),const RooCmdArg& arg8=RooCmdArg()) ;
  RooAbsData* reduce(const char* cut) ;
  RooAbsData* reduce(const RooFormulaVar& cutVar) ;
  RooAbsData* reduce(const RooArgSet& varSubset, const char* cut=0) ;
  RooAbsData* reduce(const RooArgSet& varSubset, const RooFormulaVar& cutVar) ;

  RooAbsDataStore* store() { return _dstore ; }
  const RooAbsDataStore* store() const { return _dstore ; }
  const TTree* tree() const ;
  TTree *GetClonedTree() const;

  void convertToVectorStore() ;
  virtual void convertToTreeStore();

  void attachBuffers(const RooArgSet& extObs) ;
  void resetBuffers() ;
 
  
  virtual void Draw(Option_t* option = "") ;

  void checkInit() const ; 

  // Change name of observable
  virtual Bool_t changeObservableName(const char* from, const char* to) ;

  // Add one ore more rows of data
  virtual void add(const RooArgSet& row, Double_t weight=1, Double_t weightError=0) = 0 ; // DERIVED
  virtual void fill() ; 

  // Load a given row of data
  virtual inline const RooArgSet* get() const { 
    // Return current row of dataset
    return &_vars ; 
  } 
  virtual Double_t weight() const = 0 ; // DERIVED
  virtual Double_t weightSquared() const = 0 ; // DERIVED
  virtual Bool_t valid() const { return kTRUE ; }

  enum ErrorType { Poisson, SumW2, None, Auto, Expected } ;
  /// Return the symmetric error on the current weight.
  /// See also weightError(double&,double&,ErrorType) const for asymmetric errors.
  /// \param[in] etype Type of error to compute. May throw if not supported.
  virtual double weightError(ErrorType /*etype*/=Poisson) const {
    // Dummy implementation returning zero, because not all deriving classes
    // need to implement a non-zero weight error.
    return 0.0;
  }
  /// Return the asymmetric errors on the current weight.
  /// See also weightError(ErrorType) const for symmetric error.
  /// \param[out] lo Low error.
  /// \param[out] hi High error.
  /// \param[in] etype Type of error to compute. May throw if not supported.
  virtual void weightError(double& lo, double& hi, ErrorType /*etype*/=Poisson) const {
    // Dummy implementation returning zero, because not all deriving classes
    // need to implement a non-zero weight error.
    lo=0;
    hi=0;
  }

  virtual const RooArgSet* get(Int_t index) const ;

  using RealSpans = std::map<RooFit::Detail::DataKey, RooSpan<const double>>;
  using CategorySpans = std::map<RooFit::Detail::DataKey, RooSpan<const RooAbsCategory::value_type>>;

  RealSpans getBatches(std::size_t first = 0, std::size_t len = std::numeric_limits<std::size_t>::max()) const;
  CategorySpans getCategoryBatches(std::size_t first = 0, std::size_t len = std::numeric_limits<std::size_t>::max()) const;

  ////////////////////////////////////////////////////////////////////////////////
  /// Return event weights of all events in range [first, first+len).
  /// If no contiguous structure of weights is stored, an empty batch can be returned.
  /// This indicates that the weight is constant. Use weight() to retrieve it.
  virtual RooSpan<const double> getWeightBatch(std::size_t first, std::size_t len, bool sumW2=false) const = 0;

  /// Return number of entries in dataset, *i.e.*, count unweighted entries.
  virtual Int_t numEntries() const ;
  /// Return effective number of entries in dataset, *i.e.*, sum all weights.
  virtual Double_t sumEntries() const = 0 ;
  /// Return effective number of entries in dataset inside range or after cuts, *i.e.*, sum certain weights.
  /// \param[in] cutSpec Apply given cut when counting (*e.g.* `0 < x && x < 5`). Passing `"1"` selects all events.
  /// \param[in] cutRange If the observables have a range with this name, only count events inside this range.
  virtual Double_t sumEntries(const char* cutSpec, const char* cutRange=0) const = 0 ; // DERIVED
  double sumEntriesW2() const;
  virtual Bool_t isWeighted() const { 
    // Do events in dataset have weights?
    return kFALSE ; 
  }
  virtual Bool_t isNonPoissonWeighted() const { 
    // Do events in dataset have non-integer weights?
    return kFALSE ; 
  }
  virtual void reset() ;


  Bool_t getRange(const RooAbsRealLValue& var, Double_t& lowest, Double_t& highest, Double_t marginFrac=0, Bool_t symMode=kFALSE) const ;

  // Plot the distribution of a real valued arg
  virtual Roo1DTable* table(const RooArgSet& catSet, const char* cuts="", const char* opts="") const ;
  virtual Roo1DTable* table(const RooAbsCategory& cat, const char* cuts="", const char* opts="") const ;
  /// \see RooPlot* plotOn(RooPlot* frame, const RooLinkedList& cmdList) const
  virtual RooPlot* plotOn(RooPlot* frame, 
			  const RooCmdArg& arg1=RooCmdArg::none(), const RooCmdArg& arg2=RooCmdArg::none(),
			  const RooCmdArg& arg3=RooCmdArg::none(), const RooCmdArg& arg4=RooCmdArg::none(),
			  const RooCmdArg& arg5=RooCmdArg::none(), const RooCmdArg& arg6=RooCmdArg::none(),
			  const RooCmdArg& arg7=RooCmdArg::none(), const RooCmdArg& arg8=RooCmdArg::none()) const ;

  virtual RooPlot* plotOn(RooPlot* frame, const RooLinkedList& cmdList) const ;

  // WVE --- This needs to be public to avoid CINT problems
  struct PlotOpt {
   PlotOpt() : cuts(""), drawOptions("P"), bins(0), etype(RooAbsData::Poisson), cutRange(0), histName(0), histInvisible(kFALSE),
              addToHistName(0),addToWgtSelf(1.),addToWgtOther(1.),xErrorSize(1),refreshFrameNorm(kFALSE),correctForBinWidth(kTRUE),
              scaleFactor(1.) {} ;
   const char* cuts ;
   Option_t* drawOptions ;
   RooAbsBinning* bins ;
   RooAbsData::ErrorType etype ;
   const char* cutRange ;
   const char* histName ;
   Bool_t histInvisible ;
   const char* addToHistName ;
   Double_t addToWgtSelf ;
   Double_t addToWgtOther ;
   Double_t xErrorSize ;
   Bool_t refreshFrameNorm ;
   Bool_t correctForBinWidth ;
   Double_t scaleFactor ;
  } ;
	
  // Split a dataset by a category
  virtual TList* split(const RooAbsCategory& splitCat, Bool_t createEmptyDataSets=kFALSE) const ;

  // Split a dataset by categories of a RooSimultaneous
  virtual TList* split(const RooSimultaneous& simpdf, Bool_t createEmptyDataSets=kFALSE) const ;

  // Fast splitting for SimMaster setData
  Bool_t canSplitFast() const ; 
  RooAbsData* getSimData(const char* idxstate) ;
			
  /// Calls createHistogram(const char *name, const RooAbsRealLValue& xvar, const RooLinkedList& argList) const
  TH1 *createHistogram(const char *name, const RooAbsRealLValue& xvar,
                       const RooCmdArg& arg1=RooCmdArg::none(), const RooCmdArg& arg2=RooCmdArg::none(), 
                       const RooCmdArg& arg3=RooCmdArg::none(), const RooCmdArg& arg4=RooCmdArg::none(), 
                       const RooCmdArg& arg5=RooCmdArg::none(), const RooCmdArg& arg6=RooCmdArg::none(), 
                       const RooCmdArg& arg7=RooCmdArg::none(), const RooCmdArg& arg8=RooCmdArg::none()) const ;
  /// Create and fill a ROOT histogram TH1,TH2 or TH3 with the values of this dataset.
  TH1 *createHistogram(const char *name, const RooAbsRealLValue& xvar, const RooLinkedList& argList) const ;
  TH1 *createHistogram(const char* varNameList, Int_t xbins=0, Int_t ybins=0, Int_t zbins=0) const ;

  // Fill an existing histogram
  virtual TH1 *fillHistogram(TH1 *hist, const RooArgList &plotVars, const char *cuts= "", const char* cutRange=0) const;

  // Printing interface (human readable)
  inline virtual void Print(Option_t *options= 0) const {
    // Print contents on stdout
    printStream(defaultPrintStream(),defaultPrintContents(options),defaultPrintStyle(options));
  }

  virtual void printName(std::ostream& os) const ;
  virtual void printTitle(std::ostream& os) const ;
  virtual void printClassName(std::ostream& os) const ;
  void printMultiline(std::ostream& os, Int_t contents, Bool_t verbose=kFALSE, TString indent="") const ;

  virtual Int_t defaultPrintContents(Option_t* opt) const ;

  void setDirtyProp(Bool_t flag) ;
  
  Double_t moment(const RooRealVar& var, Double_t order, const char* cutSpec=0, const char* cutRange=0) const ;
  Double_t moment(const RooRealVar& var, Double_t order, Double_t offset, const char* cutSpec=0, const char* cutRange=0) const ;
  Double_t standMoment(const RooRealVar& var, Double_t order, const char* cutSpec=0, const char* cutRange=0) const ;

  Double_t mean(const RooRealVar& var, const char* cutSpec=0, const char* cutRange=0) const { return moment(var,1,0,cutSpec,cutRange) ; }
  Double_t sigma(const RooRealVar& var, const char* cutSpec=0, const char* cutRange=0) const { return sqrt(moment(var,2,cutSpec,cutRange)) ; }
  Double_t skewness(const RooRealVar& var, const char* cutSpec=0, const char* cutRange=0) const { return standMoment(var,3,cutSpec,cutRange) ; }
  Double_t kurtosis(const RooRealVar& var, const char* cutSpec=0, const char* cutRange=0) const { return standMoment(var,4,cutSpec,cutRange) ; }

  Double_t covariance(RooRealVar &x,RooRealVar &y, const char* cutSpec=0, const char* cutRange=0) const { return corrcov(x,y,cutSpec,cutRange,kFALSE) ; }
  Double_t correlation(RooRealVar &x,RooRealVar &y, const char* cutSpec=0, const char* cutRange=0) const { return corrcov(x,y,cutSpec,cutRange,kTRUE) ; }

  TMatrixDSym* covarianceMatrix(const char* cutSpec=0, const char* cutRange=0) const { return covarianceMatrix(*get(),cutSpec,cutRange) ; }
  TMatrixDSym* correlationMatrix(const char* cutSpec=0, const char* cutRange=0) const { return correlationMatrix(*get(),cutSpec,cutRange) ; }
  TMatrixDSym* covarianceMatrix(const RooArgList& vars, const char* cutSpec=0, const char* cutRange=0) const { return corrcovMatrix(vars,cutSpec,cutRange,kFALSE) ; }
  TMatrixDSym* correlationMatrix(const RooArgList& vars, const char* cutSpec=0, const char* cutRange=0) const { return corrcovMatrix(vars,cutSpec,cutRange,kTRUE) ; }
  
  RooRealVar* meanVar(const RooRealVar &var, const char* cutSpec=0, const char* cutRange=0) const ;
  RooRealVar* rmsVar(const RooRealVar &var, const char* cutSpec=0, const char* cutRange=0) const ;

  virtual RooPlot* statOn(RooPlot* frame, 
                          const RooCmdArg& arg1=RooCmdArg::none(), const RooCmdArg& arg2=RooCmdArg::none(), 
                          const RooCmdArg& arg3=RooCmdArg::none(), const RooCmdArg& arg4=RooCmdArg::none(), 
                          const RooCmdArg& arg5=RooCmdArg::none(), const RooCmdArg& arg6=RooCmdArg::none(), 
                          const RooCmdArg& arg7=RooCmdArg::none(), const RooCmdArg& arg8=RooCmdArg::none()) ;

  virtual RooPlot* statOn(RooPlot* frame, const char *what, 
			  const char *label= "", Int_t sigDigits= 2,
			  Option_t *options= "NELU", Double_t xmin=0.15, 
			  Double_t xmax= 0.65,Double_t ymax=0.85, 
                          const char* cutSpec=0, const char* cutRange=0, 
                          const RooCmdArg* formatCmd=0);

  virtual void RecursiveRemove(TObject *obj);

  Bool_t hasFilledCache() const ; 

  void addOwnedComponent(const char* idxlabel, RooAbsData& data) ;
  static void claimVars(RooAbsData*) ;
  static Bool_t releaseVars(RooAbsData*) ;

  enum StorageType { Tree, Vector, Composite };

  static void setDefaultStorageType(StorageType s) ;

  static StorageType getDefaultStorageType();

  /// Returns snapshot of global observables stored in this data.
  /// \return Pointer to a RooArgSet with the snapshot of global observables
  ///         stored in the data. Can be `nullptr` if no global observales are
  ///         stored.
  RooArgSet const* getGlobalObservables() const { return _globalObservables.get(); }
  void setGlobalObservables(RooArgSet const& globalObservables);

  /// De-duplicated pointer to this object's name.
  /// This can be used for fast name comparisons.
  /// like `if (namePtr() == other.namePtr())`.
  /// \note TNamed::GetName() will return a pointer that's
  /// different for each object, but namePtr() always points
  /// to a unique instance.
  inline const TNamed* namePtr() const {
    return _namePtr ;
  }

  void SetName(const char* name) ;
  void SetNameTitle(const char *name, const char *title) ;


protected:

  static StorageType defaultStorageType ;

  StorageType storageType;

  Double_t corrcov(const RooRealVar& x, const RooRealVar& y, const char* cutSpec, const char* cutRange, Bool_t corr) const  ;
  TMatrixDSym* corrcovMatrix(const RooArgList& vars, const char* cutSpec, const char* cutRange, Bool_t corr) const  ;

  virtual void optimizeReadingWithCaching(RooAbsArg& arg, const RooArgSet& cacheList, const RooArgSet& keepObsList) ;
  Bool_t allClientsCached(RooAbsArg*, const RooArgSet&) ;


 // PlotOn implementation
  virtual RooPlot *plotOn(RooPlot *frame, PlotOpt o) const ;
  virtual RooPlot *plotAsymOn(RooPlot* frame, const RooAbsCategoryLValue& asymCat, PlotOpt o) const ;
  virtual RooPlot *plotEffOn(RooPlot* frame, const RooAbsCategoryLValue& effCat, PlotOpt o) const ;
 
 
  // Constant term optimizer interface
  friend class RooAbsArg ;
  friend class RooAbsReal ;
  friend class RooAbsOptTestStatistic ;
  friend class RooAbsCachedPdf ;
  friend struct RooFit::TestStatistics::ConstantTermsOptimizer;
  // for access into copied dataset:
  friend class RooFit::TestStatistics::RooAbsL;

  virtual void cacheArgs(const RooAbsArg* owner, RooArgSet& varSet, const RooArgSet* nset=0, Bool_t skipZeroWeights=kFALSE) ;
  virtual void resetCache() ;
  virtual void setArgStatus(const RooArgSet& set, Bool_t active) ;
  virtual void attachCache(const RooAbsArg* newOwner, const RooArgSet& cachedVars) ;

  virtual RooAbsData* cacheClone(const RooAbsArg* newCacheOwner, const RooArgSet* newCacheVars, const char* newName=0) = 0 ; // DERIVED
  virtual RooAbsData* reduceEng(const RooArgSet& varSubset, const RooFormulaVar* cutVar, const char* cutRange=0, 
	                        std::size_t nStart = 0, std::size_t = std::numeric_limits<std::size_t>::max(), Bool_t copyCache=kTRUE) = 0 ; // DERIVED

  RooRealVar* dataRealVar(const char* methodname, const RooRealVar& extVar) const ;

  // Column structure definition
  RooArgSet _vars;         // Dimensions of this data set
  RooArgSet _cachedVars ;  //! External variables cached with this data set

  RooAbsDataStore* _dstore ; // Data storage implementation

  std::map<std::string,RooAbsData*> _ownedComponents ; // Owned external components

  std::unique_ptr<RooArgSet> _globalObservables; // Snapshot of global observables

  mutable const TNamed * _namePtr ; //! De-duplicated name pointer. This will be equal for all objects with the same name.

private:
  void copyGlobalObservables(const RooAbsData& other);

   ClassDef(RooAbsData, 6) // Abstract data collection
};

#endif
