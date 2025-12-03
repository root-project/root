/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooAbsReal.h,v 1.75 2007/07/13 21:50:24 wouter Exp $
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
#ifndef ROO_ABS_REAL
#define ROO_ABS_REAL

#include "RooAbsArg.h"
#include "RooArgList.h"
#include "RooArgProxy.h"
#include "RooArgSet.h"
#include "RooCmdArg.h"
#include "RooCurve.h"
#include "RooFit/CodegenContext.h"
#include "RooFit/EvalContext.h"
#include "RooGlobalFunc.h"

#include <ROOT/RSpan.hxx>

#include <TList.h>
#include <TObjString.h>

class RooDataSet ;
class RooPlot;
class RooRealVar;
class RooAbsFunc;
class RooAbsCategoryLValue ;
class RooLinkedList ;
class RooNumIntConfig ;
class RooDataHist ;
class RooFunctor ;
class RooFitResult ;
class RooAbsMoment ;
class RooDerivative ;
class RooVectorDataStore ;
struct TreeReadBuffer; /// A space to attach TBranches
namespace RooBatchCompute {
struct RunContext;
}

class TH1;
class TH1F;
class TH2F;
class TH3F;

#include <iostream>
#include <list>
#include <map>
#include <string>
#include <sstream>

class RooAbsReal : public RooAbsArg {
public:
  using value_type = double;

  /// A RooAbsReal::Ref can be constructed from a `RooAbsReal&` or a `double`
  /// that will be implicitly converted to a RooConstVar&. The RooAbsReal::Ref
  /// can be used as a replacement for `RooAbsReal&`. With this type
  /// definition, you can write RooFit interfaces that accept both RooAbsReal,
  /// or simply a number that will be implicitly converted to a RooConstVar&.
  class Ref {
  public:
     inline Ref(RooAbsReal &ref) : _ref{ref} {}
     Ref(double val);
     inline operator RooAbsReal &() const { return _ref; }

  private:
     RooAbsReal &_ref;
  };

  // Constructors, assignment etc
  RooAbsReal() ;
  RooAbsReal(const char *name, const char *title, const char *unit= "") ;
  RooAbsReal(const char *name, const char *title, double minVal, double maxVal,
        const char *unit= "") ;
  RooAbsReal(const RooAbsReal& other, const char* name=nullptr);
  ~RooAbsReal() override;




  //////////////////////////////////////////////////////////////////////////////////
  /// Evaluate object. Returns either cached value or triggers a recalculation.
  /// The recalculation happens by calling getValV(), which in the end calls the
  /// virtual evaluate() functions of the respective PDFs.
  /// \param[in] normalisationSet getValV() reacts differently depending on the value of the normalisation set.
  /// If the set is `nullptr`, an unnormalised value is returned.
  /// \note The normalisation is arbitrary, because it is up to the implementation
  /// of the PDF to e.g. leave out normalisation constants for speed reasons. The range
  /// of the variables is also ignored.
  ///
  /// To normalise the result properly, a RooArgSet has to be passed, which contains
  /// the variables to normalise over.
  /// These are integrated over their current ranges to compute the normalisation constant,
  /// and the unnormalised result is divided by this value.
  inline double getVal(const RooArgSet* normalisationSet = nullptr) const {
    // Sometimes, the calling code uses an empty RooArgSet to request evaluation
    // without normalization set instead of following the `nullptr` convention.
    // To remove this ambiguity which might not always be correctly handled in
    // downstream code, we set `normalisationSet` to nullptr if it is pointing
    // to an empty set.
    if(normalisationSet && normalisationSet->empty()) {
      normalisationSet = nullptr;
    }
#ifdef ROOFIT_CHECK_CACHED_VALUES
    return _DEBUG_getVal(normalisationSet);
#else

#ifndef _WIN32
    return (_fast && !_inhibitDirty) ? _value : getValV(normalisationSet) ;
#else
    return (_fast && !inhibitDirty()) ? _value : getValV(normalisationSet) ;
#endif

#endif
  }

  /// Like getVal(const RooArgSet*), but always requires an argument for normalisation.
  inline  double getVal(const RooArgSet& normalisationSet) const {
    // Sometimes, the calling code uses an empty RooArgSet to request evaluation
    // without normalization set instead of following the `nullptr` convention.
    // To remove this ambiguity which might not always be correctly handled in
    // downstream code, we set `normalisationSet` to nullptr if it is an empty set.
    return _fast ? _value : getValV(normalisationSet.empty() ? nullptr : &normalisationSet) ;
  }

  double getVal(RooArgSet &&) const;

  virtual double getValV(const RooArgSet* normalisationSet = nullptr) const ;

  double getPropagatedError(const RooFitResult &fr, const RooArgSet &nset = {}) const;

  bool operator==(double value) const ;
  bool operator==(const RooAbsArg& other) const override;
  bool isIdentical(const RooAbsArg& other, bool assumeSameType=false) const override;


  inline const Text_t *getUnit() const {
    // Return string with unit description
    return _unit.Data();
  }
  inline void setUnit(const char *unit) {
    // Set unit description to given string
    _unit= unit;
  }
  TString getTitle(bool appendUnit= false) const;

  // Lightweight interface adaptors (caller takes ownership)
  RooFit::OwningPtr<RooAbsFunc> bindVars(const RooArgSet &vars, const RooArgSet* nset=nullptr, bool clipInvalid=false) const;

  // Create a fundamental-type object that can hold our value.
  RooFit::OwningPtr<RooAbsArg> createFundamental(const char* newname=nullptr) const override;

  // Analytical integration support
  virtual Int_t getAnalyticalIntegralWN(RooArgSet& allVars, RooArgSet& analVars, const RooArgSet* normSet, const char* rangeName=nullptr) const ;
  virtual double analyticalIntegralWN(Int_t code, const RooArgSet* normSet, const char* rangeName=nullptr) const ;
  virtual Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* rangeName=nullptr) const ;
  virtual double analyticalIntegral(Int_t code, const char* rangeName=nullptr) const ;
  virtual bool forceAnalyticalInt(const RooAbsArg& /*dep*/) const {
    // Interface to force RooRealIntegral to offer given observable for internal integration
    // even if this is deemed unsafe. This default implementation returns always false
    return false ;
  }
  virtual void forceNumInt(bool flag=true) {
    // If flag is true, all advertised analytical integrals will be ignored
    // and all integrals are calculated numerically
    _forceNumInt = flag ;
  }
  bool getForceNumInt() const { return _forceNumInt ; }

  // Chi^2 fits to histograms
  virtual RooFit::OwningPtr<RooFitResult> chi2FitTo(RooDataHist& data, const RooCmdArg& arg1={},  const RooCmdArg& arg2={},
                              const RooCmdArg& arg3={},  const RooCmdArg& arg4={}, const RooCmdArg& arg5={},
                              const RooCmdArg& arg6={},  const RooCmdArg& arg7={}, const RooCmdArg& arg8={}) ;
  virtual RooFit::OwningPtr<RooFitResult> chi2FitTo(RooDataHist& data, const RooLinkedList& cmdList) ;

  virtual RooFit::OwningPtr<RooAbsReal> createChi2(RooDataHist& data, const RooLinkedList& cmdList) ;
  virtual RooFit::OwningPtr<RooAbsReal> createChi2(RooDataHist& data, const RooCmdArg& arg1={},  const RooCmdArg& arg2={},
             const RooCmdArg& arg3={},  const RooCmdArg& arg4={}, const RooCmdArg& arg5={},
             const RooCmdArg& arg6={},  const RooCmdArg& arg7={}, const RooCmdArg& arg8={}) ;

  // Chi^2 fits to X-Y datasets
  virtual RooFit::OwningPtr<RooFitResult> chi2FitTo(RooDataSet& xydata, const RooCmdArg& arg1={},  const RooCmdArg& arg2={},
                              const RooCmdArg& arg3={},  const RooCmdArg& arg4={}, const RooCmdArg& arg5={},
                              const RooCmdArg& arg6={},  const RooCmdArg& arg7={}, const RooCmdArg& arg8={}) ;
  virtual RooFit::OwningPtr<RooFitResult> chi2FitTo(RooDataSet& xydata, const RooLinkedList& cmdList) ;

  virtual RooFit::OwningPtr<RooAbsReal> createChi2(RooDataSet& data, const RooLinkedList& cmdList) ;
  virtual RooFit::OwningPtr<RooAbsReal> createChi2(RooDataSet& data, const RooCmdArg& arg1={},  const RooCmdArg& arg2={},
               const RooCmdArg& arg3={},  const RooCmdArg& arg4={}, const RooCmdArg& arg5={},
               const RooCmdArg& arg6={},  const RooCmdArg& arg7={}, const RooCmdArg& arg8={}) ;

  virtual RooFit::OwningPtr<RooAbsReal> createProfile(const RooArgSet& paramsOfInterest) ;


  RooFit::OwningPtr<RooAbsReal> createIntegral(const RooArgSet& iset, const RooCmdArg& arg1, const RooCmdArg& arg2={},
                             const RooCmdArg& arg3={}, const RooCmdArg& arg4={},
              const RooCmdArg& arg5={}, const RooCmdArg& arg6={},
              const RooCmdArg& arg7={}, const RooCmdArg& arg8={}) const ;

  /// Create integral over observables in iset in range named rangeName.
  RooFit::OwningPtr<RooAbsReal> createIntegral(const RooArgSet& iset, const char* rangeName) const {
    return createIntegral(iset,nullptr,nullptr,rangeName) ;
  }
  /// Create integral over observables in iset in range named rangeName with integrand normalized over observables in nset
  RooFit::OwningPtr<RooAbsReal> createIntegral(const RooArgSet& iset, const RooArgSet& nset, const char* rangeName=nullptr) const {
    return createIntegral(iset,&nset,nullptr,rangeName) ;
  }
  /// Create integral over observables in iset in range named rangeName with integrand normalized over observables in nset while
  /// using specified configuration for any numeric integration.
  RooFit::OwningPtr<RooAbsReal> createIntegral(const RooArgSet& iset, const RooArgSet& nset, const RooNumIntConfig& cfg, const char* rangeName=nullptr) const {
    return createIntegral(iset,&nset,&cfg,rangeName) ;
  }
  /// Create integral over observables in iset in range named rangeName using specified configuration for any numeric integration.
  RooFit::OwningPtr<RooAbsReal> createIntegral(const RooArgSet& iset, const RooNumIntConfig& cfg, const char* rangeName=nullptr) const {
    return createIntegral(iset,nullptr,&cfg,rangeName) ;
  }
  virtual RooFit::OwningPtr<RooAbsReal> createIntegral(const RooArgSet& iset, const RooArgSet* nset=nullptr, const RooNumIntConfig* cfg=nullptr, const char* rangeName=nullptr) const ;


  void setParameterizeIntegral(const RooArgSet& paramVars) ;

  // Create running integrals
  RooFit::OwningPtr<RooAbsReal> createRunningIntegral(const RooArgSet& iset, const RooArgSet& nset={}) ;
  RooFit::OwningPtr<RooAbsReal> createRunningIntegral(const RooArgSet& iset, const RooCmdArg& arg1, const RooCmdArg& arg2={},
         const RooCmdArg& arg3={}, const RooCmdArg& arg4={},
         const RooCmdArg& arg5={}, const RooCmdArg& arg6={},
         const RooCmdArg& arg7={}, const RooCmdArg& arg8={}) ;
  RooFit::OwningPtr<RooAbsReal> createIntRI(const RooArgSet& iset, const RooArgSet& nset={}) ;
  RooFit::OwningPtr<RooAbsReal> createScanRI(const RooArgSet& iset, const RooArgSet& nset, Int_t numScanBins, Int_t intOrder) ;


  // Optimized accept/reject generator support
  virtual Int_t getMaxVal(const RooArgSet& vars) const ;
  virtual double maxVal(Int_t code) const ;
  virtual Int_t minTrialSamples(const RooArgSet& /*arGenObs*/) const { return 0 ; }


  // Plotting options
  void setPlotLabel(const char *label);
  const char *getPlotLabel() const;

  virtual double defaultErrorLevel() const {
    // Return default level for MINUIT error analysis
    return 1.0 ;
  }

  const RooNumIntConfig* getIntegratorConfig() const ;
  RooNumIntConfig* getIntegratorConfig() ;
  static RooNumIntConfig* defaultIntegratorConfig()  ;
  RooNumIntConfig* specialIntegratorConfig() const ;
  RooNumIntConfig* specialIntegratorConfig(bool createOnTheFly) ;
  void setIntegratorConfig() ;
  void setIntegratorConfig(const RooNumIntConfig& config) ;

  virtual void fixAddCoefNormalization(const RooArgSet& addNormSet=RooArgSet(),bool force=true) ;
  virtual void fixAddCoefRange(const char* rangeName=nullptr,bool force=true) ;

  virtual void preferredObservableScanOrder(const RooArgSet& obs, RooArgSet& orderedObs) const ;

  // User entry point for plotting
  virtual RooPlot* plotOn(RooPlot* frame,
           const RooCmdArg& arg1={}, const RooCmdArg& arg2={},
           const RooCmdArg& arg3={}, const RooCmdArg& arg4={},
           const RooCmdArg& arg5={}, const RooCmdArg& arg6={},
           const RooCmdArg& arg7={}, const RooCmdArg& arg8={},
           const RooCmdArg& arg9={}, const RooCmdArg& arg10={}
              ) const ;


  enum ScaleType { Raw, Relative, NumEvent, RelativeExpected } ;

  // Fill an existing histogram
  TH1 *fillHistogram(TH1 *hist, const RooArgList &plotVars,
           double scaleFactor= 1, const RooArgSet *projectedVars= nullptr, bool scaling=true,
           const RooArgSet* condObs=nullptr, bool setError=true) const;

  // Create 1,2, and 3D histograms from and fill it
  TH1 *createHistogram(RooStringView varNameList, Int_t xbins=0, Int_t ybins=0, Int_t zbins=0) const ;
  TH1* createHistogram(const char *name, const RooAbsRealLValue& xvar, RooLinkedList& argList) const ;
  TH1 *createHistogram(const char *name, const RooAbsRealLValue& xvar,
                       const RooCmdArg& arg1={}, const RooCmdArg& arg2={},
                       const RooCmdArg& arg3={}, const RooCmdArg& arg4={},
                       const RooCmdArg& arg5={}, const RooCmdArg& arg6={},
                       const RooCmdArg& arg7={}, const RooCmdArg& arg8={}) const ;

  // Fill a RooDataHist
  RooDataHist* fillDataHist(RooDataHist *hist, const RooArgSet* nset, double scaleFactor,
             bool correctForBinVolume=false, bool showProgress=false) const ;

  // I/O streaming interface (machine readable)
  bool readFromStream(std::istream& is, bool compact, bool verbose=false) override ;
  void writeToStream(std::ostream& os, bool compact) const override ;

  // Printing interface (human readable)
  void printValue(std::ostream& os) const override ;
  void printMultiline(std::ostream& os, Int_t contents, bool verbose=false, TString indent="") const override ;

  inline void setCachedValue(double value, bool notifyClients = true) final;

  // Evaluation error logging
  class EvalError {
  public:
    EvalError() { }
    EvalError(const EvalError& other) : _msg(other._msg), _srvval(other._srvval) { }
    void setMessage(const char* tmp) { std::string s(tmp); s.swap(_msg); }
    void setServerValues(const char* tmp) { std::string s(tmp); s.swap(_srvval); }
    std::string _msg;
    std::string _srvval;
  } ;

  enum ErrorLoggingMode { PrintErrors, CollectErrors, CountErrors, Ignore } ;

  /// Context to temporarily change the error logging mode as long as the context is alive.
  class EvalErrorContext {
  public:
     EvalErrorContext(ErrorLoggingMode m) : _old{evalErrorLoggingMode()} { setEvalErrorLoggingMode(m); }

     EvalErrorContext(EvalErrorContext const&) = delete;
     EvalErrorContext(EvalErrorContext &&) = delete;
     EvalErrorContext& operator=(EvalErrorContext const&) = delete;
     EvalErrorContext& operator=(EvalErrorContext &&) = delete;

     ~EvalErrorContext() { setEvalErrorLoggingMode(_old); }
  private:
     ErrorLoggingMode _old;
  };

  static ErrorLoggingMode evalErrorLoggingMode() ;
  static void setEvalErrorLoggingMode(ErrorLoggingMode m) ;
  void logEvalError(const char* message, const char* serverValueString=nullptr) const ;
  static void logEvalError(const RooAbsReal* originator, const char* origName, const char* message, const char* serverValueString=nullptr) ;
  static void printEvalErrors(std::ostream&os=std::cout, Int_t maxPerNode=10000000) ;
  static Int_t numEvalErrors() ;
  static Int_t numEvalErrorItems();
  static std::map<const RooAbsArg *, std::pair<std::string, std::list<RooAbsReal::EvalError>>>::iterator evalErrorIter();

  static void clearEvalErrorLog() ;

  /// Tests if the distribution is binned. Unless overridden by derived classes, this always returns false.
  virtual bool isBinnedDistribution(const RooArgSet& /*obs*/) const { return false ; }
  virtual std::list<double>* binBoundaries(RooAbsRealLValue& obs, double xlo, double xhi) const;
  virtual std::list<double>* plotSamplingHint(RooAbsRealLValue& obs, double xlo, double xhi) const;

  RooFunctor* functor(const RooArgList& obs, const RooArgList& pars=RooArgList(), const RooArgSet& nset=RooArgSet()) const ;
  TF1* asTF(const RooArgList& obs, const RooArgList& pars=RooArgList(), const RooArgSet& nset=RooArgSet()) const ;

  RooDerivative* derivative(RooRealVar& obs, Int_t order=1, double eps=0.001) ;
  RooDerivative* derivative(RooRealVar& obs, const RooArgSet& normSet, Int_t order, double eps=0.001) ;

  RooAbsMoment* moment(RooRealVar& obs, Int_t order, bool central, bool takeRoot) ;
  RooAbsMoment* moment(RooRealVar& obs, const RooArgSet& normObs, Int_t order, bool central, bool takeRoot, bool intNormObs) ;

  RooAbsMoment* mean(RooRealVar& obs) { return moment(obs,1,false,false) ; }
  RooAbsMoment* mean(RooRealVar& obs, const RooArgSet& nset) { return moment(obs,nset,1,false,false,true) ; }
  RooAbsMoment* sigma(RooRealVar& obs) { return moment(obs,2,true,true) ; }
  RooAbsMoment* sigma(RooRealVar& obs, const RooArgSet& nset) { return moment(obs,nset,2,true,true,true) ; }

  double findRoot(RooRealVar& x, double xmin, double xmax, double yval) ;


  virtual bool setData(RooAbsData& /*data*/, bool /*cloneData*/=true) { return true ; }

  virtual void enableOffsetting(bool);
  virtual bool isOffsetting() const { return false ; }
  virtual double offset() const { return 0 ; }

  static void setHideOffset(bool flag);
  static bool hideOffset() ;

  bool isSelectedComp() const ;
  void selectComp(bool flag) {
     // If flag is true, only selected component will be included in evaluates of RooAddPdf components
     _selectComp = flag ;
  }

  const RooAbsReal* createPlotProjection(const RooArgSet& depVars, const RooArgSet& projVars, RooArgSet*& cloneSet) const ;
  const RooAbsReal *createPlotProjection(const RooArgSet &dependentVars, const RooArgSet *projectedVars,
                     RooArgSet *&cloneSet, const char* rangeName=nullptr, const RooArgSet* condObs=nullptr) const;
  virtual void doEval(RooFit::EvalContext &) const;

  virtual bool hasGradient() const { return false; }
  virtual void gradient(double *) const {
    if(!hasGradient()) throw std::runtime_error("RooAbsReal::gradient(double *) not implemented by this class!");
  }

  // PlotOn with command list
  virtual RooPlot* plotOn(RooPlot* frame, RooLinkedList& cmdList) const ;

protected:
  friend class BatchInterfaceAccessor;
  friend class RooVectorDataStore;
  friend class RooRealBinding;
  friend class RooRealSumPdf;
  friend class RooRealSumFunc;
  friend class RooAddHelpers;
  friend class RooAddPdf;
  friend class RooAddModel;
  friend class AddCacheElem;
  friend class RooFit::EvalContext;

  // Hook for objects with normalization-dependent parameters interpretation
  virtual void selectNormalization(const RooArgSet* depSet=nullptr, bool force=false) ;
  virtual void selectNormalizationRange(const char* rangeName=nullptr, bool force=false) ;

  // Helper functions for plotting
  bool plotSanityChecks(RooPlot* frame) const ;
  void makeProjectionSet(const RooAbsArg* plotVar, const RooArgSet* allVars,
          RooArgSet& projectedVars, bool silent) const ;

  TString integralNameSuffix(const RooArgSet& iset, const RooArgSet* nset=nullptr, const char* rangeName=nullptr, bool omitEmpty=false) const ;

  void plotOnCompSelect(RooArgSet* selNodes) const ;
  RooPlot* plotOnWithErrorBand(RooPlot* frame,const RooFitResult& fr, double Z, const RooArgSet* params, const RooLinkedList& argList, bool method1) const ;

  template<typename... Proxies>
  bool matchArgs(const RooArgSet& allDeps, RooArgSet& analDeps, const RooArgProxy& a, const Proxies&... proxies) const
  {
    TList nameList;
    // Fold expression to add all proxy names to the list
    nameList.Add(new TObjString(a.absArg()->GetName()));
    (nameList.Add(new TObjString(proxies.absArg()->GetName())), ...);

    bool result = matchArgsByName(allDeps, analDeps, nameList);
    nameList.Delete(); // Clean up the list contents
    return result;
  }

  bool matchArgs(const RooArgSet& allDeps, RooArgSet& numDeps,
         const RooArgSet& set) const ;

  RooFit::OwningPtr<RooAbsReal> createIntObj(const RooArgSet& iset, const RooArgSet* nset, const RooNumIntConfig* cfg, const char* rangeName) const ;
  void findInnerMostIntegration(const RooArgSet& allObs, RooArgSet& innerObs, const char* rangeName) const ;

  // Internal consistency checking (needed by RooDataSet)
  /// Check if current value is valid.
  bool isValid() const override { return isValidReal(_value); }
  /// Interface function to check if given value is a valid value for this object. Returns true unless overridden.
  virtual bool isValidReal(double /*value*/, bool printError = false) const { (void)printError; return true; }

  // Function evaluation and error tracing
  double traceEval(const RooArgSet* set) const ;

  /// Evaluate this PDF / function / constant. Needs to be overridden by all derived classes.
  virtual double evaluate() const = 0;

  // Hooks for RooDataSet interface
  void syncCache(const RooArgSet* set=nullptr) override { getVal(set) ; }
  void copyCache(const RooAbsArg* source, bool valueOnly=false, bool setValDirty=true) override ;
  void attachToTree(TTree& t, Int_t bufSize=32000) override ;
  void attachToVStore(RooVectorDataStore& vstore) override ;
  void setTreeBranchStatus(TTree& t, bool active) override ;
  void fillTreeBranch(TTree& t) override ;

  struct PlotOpt {
     Option_t *drawOptions = "L";
     double scaleFactor = 1.0;
     ScaleType stype = Relative;
     const RooAbsData *projData = nullptr;
     bool binProjData = false;
     const RooArgSet *projSet = nullptr;
     double precision = 1e-3;
     bool shiftToZero = false;
     const RooArgSet *projDataSet = nullptr;
     const char *normRangeName = nullptr;
     double rangeLo = 0.0;
     double rangeHi = 0.0;
     bool postRangeFracScale = false;
     RooCurve::WingMode wmode = RooCurve::Extended;
     const char *projectionRangeName = nullptr;
     bool curveInvisible = false;
     const char *curveName = nullptr;
     const char *addToCurveName = nullptr;
     double addToWgtSelf = 1.0;
     double addToWgtOther = 1.0;
     Int_t numCPU = 1;
     RooFit::MPSplit interleave = RooFit::Interleave;
     const char *curveNameSuffix = "";
     Int_t numee = 10;
     double eeval = 0.0;
     bool doeeval = false;
     bool progress = false;
     const RooFitResult *errorFR = nullptr;
  };

  // Plot implementation functions
  virtual RooPlot *plotOn(RooPlot* frame, PlotOpt o) const;

  virtual RooPlot *plotAsymOn(RooPlot *frame, const RooAbsCategoryLValue& asymCat, PlotOpt o) const;

  bool matchArgsByName(const RooArgSet &allArgs, RooArgSet &matchedArgs, const TList &nameList) const;

  bool redirectServersHook(const RooAbsCollection & newServerList, bool mustReplaceAll,
                                   bool nameChange, bool isRecursiveStep) override;

  static void globalSelectComp(bool flag) ;

  // This struct can be used to flip the global switch to select components.
  // Doing this with RAII prevents forgetting to reset the state.
  struct GlobalSelectComponentRAII {
      GlobalSelectComponentRAII(bool state) :
      _oldState{_globalSelectComp} {
        if (state != RooAbsReal::_globalSelectComp)
          RooAbsReal::_globalSelectComp = state;
      }

      ~GlobalSelectComponentRAII() {
        if (RooAbsReal::_globalSelectComp != _oldState)
          RooAbsReal::_globalSelectComp = _oldState;
      }

      bool _oldState;
  };


private:

  /// Debug version of getVal(), which is slow and does error checking.
  double _DEBUG_getVal(const RooArgSet* normalisationSet) const;

  //--------------------------------------------------------------------

 protected:

   double _plotMin = 0.0;                                  ///< Minimum of plot range
   double _plotMax = 0.0;                                  ///< Maximum of plot range
   Int_t _plotBins = 100;                                  ///< Number of plot bins
   mutable double _value = 0.0;                            ///< Cache for current value of object
   TString _unit;                                          ///< Unit for objects value
   TString _label;                                         ///< Plot label for objects value
   bool _forceNumInt = false;                              ///< Force numerical integration if flag set
   std::unique_ptr<RooNumIntConfig> _specIntegratorConfig; // Numeric integrator configuration specific for this object
   TreeReadBuffer *_treeReadBuffer = nullptr;              //! A buffer for reading values from trees
   bool _selectComp = true;                                //! Component selection flag for RooAbsPdf::plotCompOn
   mutable RooFit::UniqueId<RooArgSet>::Value_t _lastNormSetId = RooFit::UniqueId<RooArgSet>::nullval; ///<!

   static bool _globalSelectComp; // Global activation switch for component selection
   static bool _hideOffset;       ///< Offset hiding flag

   ClassDefOverride(RooAbsReal,3); // Abstract real-valued variable
};


////////////////////////////////////////////////////////////////////////////////
/// Overwrite the value stored in this object's cache.
/// This can be used to fake a computation that resulted in `value`.
/// \param[in] value Value to write.
/// \param[in] notifyClients If true, notify users of this object that its value changed.
/// This is the default.
void RooAbsReal::setCachedValue(double value, bool notifyClients) {
  _value = value;

  if (notifyClients) {
    setValueDirty();
    _valueDirty = false;
  }
}


#endif
