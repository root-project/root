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
#include "RooFit/UniqueId.h"

#include <ROOT/RConfig.hxx> // R__DEPRECATED
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


class RooAbsData : public TNamed, public RooPrintable {
public:

  // Constructors, factory methods etc.
  RooAbsData() ;
  RooAbsData(RooStringView name, RooStringView title, const RooArgSet& vars, RooAbsDataStore* store=0) ;
  RooAbsData(const RooAbsData& other, const char* newname = 0) ;

  RooAbsData& operator=(const RooAbsData& other);
  ~RooAbsData() override ;
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


  void Draw(Option_t* option = "") override ;

  void checkInit() const ;

  // Change name of observable
  virtual bool changeObservableName(const char* from, const char* to) ;

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
  virtual bool valid() const { return true ; }

  enum ErrorType { Poisson, SumW2, None, Auto, Expected } ;
  /// Return the symmetric error on the current weight.
  /// See also weightError(double&,double&,ErrorType) const for asymmetric errors.
  // \param[in] etype Type of error to compute. May throw if not supported.
  virtual double weightError(ErrorType /*etype*/=Poisson) const {
    // Dummy implementation returning zero, because not all deriving classes
    // need to implement a non-zero weight error.
    return 0.0;
  }
  /// Return the asymmetric errors on the current weight.
  /// See also weightError(ErrorType) const for symmetric error.
  /// \param[out] lo Low error.
  /// \param[out] hi High error.
  // \param[in] etype Type of error to compute. May throw if not supported.
  virtual void weightError(double& lo, double& hi, ErrorType /*etype*/=Poisson) const {
    // Dummy implementation returning zero, because not all deriving classes
    // need to implement a non-zero weight error.
    lo=0;
    hi=0;
  }

  virtual const RooArgSet* get(Int_t index) const ;

  virtual void getBatches(RooBatchCompute::RunContext& evalData,
      std::size_t first = 0, std::size_t len = std::numeric_limits<std::size_t>::max()) const;
  virtual std::map<const std::string, RooSpan<const RooAbsCategory::value_type>> getCategoryBatches(std::size_t first = 0, std::size_t len = std::numeric_limits<std::size_t>::max()) const;

  ////////////////////////////////////////////////////////////////////////////////
  /// Return event weights of all events in range [first, first+len).
  /// If no contiguous structure of weights is stored, an empty batch can be returned.
  /// This indicates that the weight is constant. Use weight() to retrieve it.
  virtual RooSpan<const double> getWeightBatch(std::size_t first, std::size_t len, bool sumW2=false) const = 0;

  /// Return number of entries in dataset, i.e., count unweighted entries.
  virtual Int_t numEntries() const ;
  /// Return effective number of entries in dataset, i.e., sum all weights.
  virtual Double_t sumEntries() const = 0 ;
  /// Return effective number of entries in dataset inside range or after cuts, i.e., sum certain weights.
  /// \param[in] cutSpec Apply given cut when counting (e.g. `0 < x && x < 5`). Passing `"1"` selects all events.
  /// \param[in] cutRange If the observables have a range with this name, only count events inside this range.
  virtual Double_t sumEntries(const char* cutSpec, const char* cutRange=0) const = 0 ; // DERIVED
  double sumEntriesW2() const;
  virtual bool isWeighted() const {
    // Do events in dataset have weights?
    return false ;
  }
  virtual bool isNonPoissonWeighted() const {
    // Do events in dataset have non-integer weights?
    return false ;
  }
  virtual void reset() ;


  bool getRange(const RooAbsRealLValue& var, Double_t& lowest, Double_t& highest, Double_t marginFrac=0, bool symMode=false) const ;

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
   PlotOpt() : cuts(""), drawOptions("P"), bins(0), etype(RooAbsData::Poisson), cutRange(0), histName(0), histInvisible(false),
              addToHistName(0),addToWgtSelf(1.),addToWgtOther(1.),xErrorSize(1),refreshFrameNorm(false),correctForBinWidth(true),
              scaleFactor(1.) {} ;
   const char* cuts ;
   Option_t* drawOptions ;
   RooAbsBinning* bins ;
   RooAbsData::ErrorType etype ;
   const char* cutRange ;
   const char* histName ;
   bool histInvisible ;
   const char* addToHistName ;
   Double_t addToWgtSelf ;
   Double_t addToWgtOther ;
   Double_t xErrorSize ;
   bool refreshFrameNorm ;
   bool correctForBinWidth ;
   Double_t scaleFactor ;
  } ;

  // Split a dataset by a category
  virtual TList* split(const RooAbsCategory& splitCat, bool createEmptyDataSets=false) const ;

  // Split a dataset by categories of a RooSimultaneous
  virtual TList* split(const RooSimultaneous& simpdf, bool createEmptyDataSets=false) const ;

  // Fast splitting for SimMaster setData
  bool canSplitFast() const ;
  RooAbsData* getSimData(const char* idxstate) ;

  /// Calls createHistogram(const char *name, const RooAbsRealLValue& xvar, const RooLinkedList& argList) const
  TH1 *createHistogram(const char *name, const RooAbsRealLValue& xvar,
                       const RooCmdArg& arg1=RooCmdArg::none(), const RooCmdArg& arg2=RooCmdArg::none(),
                       const RooCmdArg& arg3=RooCmdArg::none(), const RooCmdArg& arg4=RooCmdArg::none(),
                       const RooCmdArg& arg5=RooCmdArg::none(), const RooCmdArg& arg6=RooCmdArg::none(),
                       const RooCmdArg& arg7=RooCmdArg::none(), const RooCmdArg& arg8=RooCmdArg::none()) const ;
  // Developer note: the `binArgX` parameter has no default `none` value,
  // because then the signature would be ambiguous with the deprecated bin
  // integer overload below. When the deprecated overload is removed, a default
  // value must be set (tutorial failures will remind us to do that).
  TH1 *createHistogram(const char* varNameList,
                       const RooCmdArg& binArgX, const RooCmdArg& binArgY=RooCmdArg::none(),
                       const RooCmdArg& binArgZ=RooCmdArg::none()) const;
  /// Create and fill a ROOT histogram TH1,TH2 or TH3 with the values of this dataset.
  TH1 *createHistogram(const char *name, const RooAbsRealLValue& xvar, const RooLinkedList& argList) const ;
  TH1 *createHistogram(const char* varNameList, Int_t xbins=0, Int_t ybins=0, Int_t zbins=0) const
      R__DEPRECATED(6, 30, "Use the overload of RooAbsData::createHistogram that takes RooFit command arguments.");
  TH2F* createHistogram(const RooAbsRealLValue& var1, const RooAbsRealLValue& var2, const char* cuts="",
         const char *name= "hist") const;
  TH2F* createHistogram(const RooAbsRealLValue& var1, const RooAbsRealLValue& var2, int nx, int ny,
                        const char* cuts="", const char *name="hist") const;

  // Fill an existing histogram
  virtual TH1 *fillHistogram(TH1 *hist, const RooArgList &plotVars, const char *cuts= "", const char* cutRange=0) const;

  // Printing interface (human readable)
  inline void Print(Option_t *options= 0) const override {
    // Print contents on stdout
    printStream(defaultPrintStream(),defaultPrintContents(options),defaultPrintStyle(options));
  }

  void printName(std::ostream& os) const override ;
  void printTitle(std::ostream& os) const override ;
  void printClassName(std::ostream& os) const override ;
  void printMultiline(std::ostream& os, Int_t contents, bool verbose=false, TString indent="") const override ;

  Int_t defaultPrintContents(Option_t* opt) const override ;

  void setDirtyProp(bool flag) ;

  Double_t moment(const RooRealVar& var, Double_t order, const char* cutSpec=0, const char* cutRange=0) const ;
  Double_t moment(const RooRealVar& var, Double_t order, Double_t offset, const char* cutSpec=0, const char* cutRange=0) const ;
  Double_t standMoment(const RooRealVar& var, Double_t order, const char* cutSpec=0, const char* cutRange=0) const ;

  Double_t mean(const RooRealVar& var, const char* cutSpec=0, const char* cutRange=0) const { return moment(var,1,0,cutSpec,cutRange) ; }
  Double_t sigma(const RooRealVar& var, const char* cutSpec=0, const char* cutRange=0) const { return sqrt(moment(var,2,cutSpec,cutRange)) ; }
  Double_t skewness(const RooRealVar& var, const char* cutSpec=0, const char* cutRange=0) const { return standMoment(var,3,cutSpec,cutRange) ; }
  Double_t kurtosis(const RooRealVar& var, const char* cutSpec=0, const char* cutRange=0) const { return standMoment(var,4,cutSpec,cutRange) ; }

  Double_t covariance(RooRealVar &x,RooRealVar &y, const char* cutSpec=0, const char* cutRange=0) const { return corrcov(x,y,cutSpec,cutRange,false) ; }
  Double_t correlation(RooRealVar &x,RooRealVar &y, const char* cutSpec=0, const char* cutRange=0) const { return corrcov(x,y,cutSpec,cutRange,true) ; }

  TMatrixDSym* covarianceMatrix(const char* cutSpec=0, const char* cutRange=0) const { return covarianceMatrix(*get(),cutSpec,cutRange) ; }
  TMatrixDSym* correlationMatrix(const char* cutSpec=0, const char* cutRange=0) const { return correlationMatrix(*get(),cutSpec,cutRange) ; }
  TMatrixDSym* covarianceMatrix(const RooArgList& vars, const char* cutSpec=0, const char* cutRange=0) const { return corrcovMatrix(vars,cutSpec,cutRange,false) ; }
  TMatrixDSym* correlationMatrix(const RooArgList& vars, const char* cutSpec=0, const char* cutRange=0) const { return corrcovMatrix(vars,cutSpec,cutRange,true) ; }

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

  void RecursiveRemove(TObject *obj) override;

  bool hasFilledCache() const ;

  void addOwnedComponent(const char* idxlabel, RooAbsData& data) ;
  static void claimVars(RooAbsData*) ;
  static bool releaseVars(RooAbsData*) ;

  enum StorageType { Tree, Vector, Composite };

  static void setDefaultStorageType(StorageType s) ;

  static StorageType getDefaultStorageType();

  /// Returns snapshot of global observables stored in this data.
  /// \return Pointer to a RooArgSet with the snapshot of global observables
  ///         stored in the data. Can be `nullptr` if no global observables are
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

  void SetName(const char* name) override ;
  void SetNameTitle(const char *name, const char *title) override ;

  /// Returns a unique ID that is different for every instantiated RooAbsData object.
  /// This ID can be used whether two RooAbsData are the same object, which is safer
  /// than memory address comparisons that might result in false positives when
  /// memory is reused.
  RooFit::UniqueId<RooAbsData> const& uniqueId() const { return _uniqueId; }

protected:

  static StorageType defaultStorageType ;

  StorageType storageType;

  Double_t corrcov(const RooRealVar& x, const RooRealVar& y, const char* cutSpec, const char* cutRange, bool corr) const  ;
  TMatrixDSym* corrcovMatrix(const RooArgList& vars, const char* cutSpec, const char* cutRange, bool corr) const  ;

  virtual void optimizeReadingWithCaching(RooAbsArg& arg, const RooArgSet& cacheList, const RooArgSet& keepObsList) ;
  bool allClientsCached(RooAbsArg*, const RooArgSet&) ;


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

  virtual void cacheArgs(const RooAbsArg* owner, RooArgSet& varSet, const RooArgSet* nset=0, bool skipZeroWeights=false) ;
  virtual void resetCache() ;
  virtual void setArgStatus(const RooArgSet& set, bool active) ;
  virtual void attachCache(const RooAbsArg* newOwner, const RooArgSet& cachedVars) ;

  virtual RooAbsData* reduceEng(const RooArgSet& varSubset, const RooFormulaVar* cutVar, const char* cutRange=0,
                           std::size_t nStart = 0, std::size_t = std::numeric_limits<std::size_t>::max()) = 0 ;

  RooRealVar* dataRealVar(const char* methodname, const RooRealVar& extVar) const ;

  // Column structure definition
  RooArgSet _vars;         ///< Dimensions of this data set
  RooArgSet _cachedVars ;  ///<! External variables cached with this data set

  RooAbsDataStore* _dstore ; ///< Data storage implementation

  std::map<std::string,RooAbsData*> _ownedComponents ; ///< Owned external components

  std::unique_ptr<RooArgSet> _globalObservables; ///< Snapshot of global observables

  mutable const TNamed * _namePtr ; ///<! De-duplicated name pointer. This will be equal for all objects with the same name.

private:
  void copyGlobalObservables(const RooAbsData& other);

  const RooFit::UniqueId<RooAbsData> _uniqueId; ///<!

   ClassDefOverride(RooAbsData, 6) // Abstract data collection
};

#endif
