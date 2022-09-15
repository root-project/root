/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooAbsPdf.h,v 1.90 2007/07/21 21:32:52 wouter Exp $
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
#ifndef ROO_ABS_PDF
#define ROO_ABS_PDF

#include "RooAbsReal.h"
#include "RooObjCacheManager.h"
#include "RooCmdArg.h"
#include "RooFit/UniqueId.h"

class RooDataSet;
class RooDataHist ;
class RooArgSet ;
class RooAbsGenContext ;
class RooFitResult ;
class RooExtendPdf ;
class RooCategory ;
class TPaveText;
class TH1F;
class TH2F;
class TList ;
class RooLinkedList ;
class RooMinimizer ;
class RooNumGenConfig ;
class RooRealIntegral ;

namespace RooBatchCompute {
struct RunContext;
}

class RooAbsPdf : public RooAbsReal {
public:

  // Constructors, assignment etc
  RooAbsPdf() ;
  RooAbsPdf(const char *name, const char *title=nullptr) ;
  RooAbsPdf(const char *name, const char *title, double minVal, double maxVal) ;
  // RooAbsPdf(const RooAbsPdf& other, const char* name=nullptr);
  ~RooAbsPdf() override;

  // Toy MC generation

  ////////////////////////////////////////////////////////////////////////////////
  /// See RooAbsPdf::generate(const RooArgSet&,const RooCmdArg&,const RooCmdArg&,const RooCmdArg&,const RooCmdArg&,const RooCmdArg&,const RooCmdArg&)
  /// \param[in] whatVars Set of observables to generate for each event according to this model.
  /// \param[in] nEvents How many events to generate
  /// \param arg1,arg2,arg3,arg4,arg5 Optional command arguments.
  RooDataSet *generate(const RooArgSet &whatVars, Int_t nEvents, const RooCmdArg& arg1,
                       const RooCmdArg& arg2=RooCmdArg::none(), const RooCmdArg& arg3=RooCmdArg::none(),
                       const RooCmdArg& arg4=RooCmdArg::none(), const RooCmdArg& arg5=RooCmdArg::none()) {
    return generate(whatVars,RooFit::NumEvents(nEvents),arg1,arg2,arg3,arg4,arg5) ;
  }
  RooDataSet *generate(const RooArgSet &whatVars,
                       const RooCmdArg& arg1=RooCmdArg::none(),const RooCmdArg& arg2=RooCmdArg::none(),
                       const RooCmdArg& arg3=RooCmdArg::none(),const RooCmdArg& arg4=RooCmdArg::none(),
                       const RooCmdArg& arg5=RooCmdArg::none(),const RooCmdArg& arg6=RooCmdArg::none()) ;
  RooDataSet *generate(const RooArgSet &whatVars, double nEvents = 0, bool verbose=false, bool autoBinned=true,
             const char* binnedTag="", bool expectedData=false, bool extended = false) const;
  RooDataSet *generate(const RooArgSet &whatVars, const RooDataSet &prototype, Int_t nEvents= 0,
             bool verbose=false, bool randProtoOrder=false, bool resampleProto=false) const;


  class GenSpec {
  public:
    virtual ~GenSpec() ;
    GenSpec() { _genContext = nullptr ; _protoData = nullptr ; _init = false ; _extended=false, _nGen=0 ; _randProto = false ; _resampleProto=false ; }
  private:
    GenSpec(RooAbsGenContext* context, const RooArgSet& whatVars, RooDataSet* protoData, Int_t nGen, bool extended,
       bool randProto, bool resampleProto, TString dsetName, bool init=false) ;
    GenSpec(const GenSpec& other) ;

    friend class RooAbsPdf ;
    RooAbsGenContext* _genContext ;
    RooArgSet _whatVars ;
    RooDataSet* _protoData ;
    Int_t _nGen ;
    bool _extended ;
    bool _randProto ;
    bool _resampleProto ;
    TString _dsetName ;
    bool _init ;

    ClassDef(GenSpec,0) // Generation specification
  } ;

  ///Prepare GenSpec configuration object for efficient generation of multiple datasets from identical specification.
  GenSpec* prepareMultiGen(const RooArgSet &whatVars,
            const RooCmdArg& arg1=RooCmdArg::none(),const RooCmdArg& arg2=RooCmdArg::none(),
            const RooCmdArg& arg3=RooCmdArg::none(),const RooCmdArg& arg4=RooCmdArg::none(),
            const RooCmdArg& arg5=RooCmdArg::none(),const RooCmdArg& arg6=RooCmdArg::none()) ;
  ///Generate according to GenSpec obtained from prepareMultiGen().
  RooDataSet* generate(GenSpec&) const ;


  ////////////////////////////////////////////////////////////////////////////////
  /// As RooAbsPdf::generateBinned(const RooArgSet&, const RooCmdArg&,const RooCmdArg&, const RooCmdArg&,const RooCmdArg&, const RooCmdArg&,const RooCmdArg&)
  /// \param[in] whatVars set
  /// \param[in] nEvents How many events to generate
  /// \param arg1,arg2,arg3,arg4,arg5 ordered arguments
  virtual RooDataHist *generateBinned(const RooArgSet &whatVars, double nEvents, const RooCmdArg& arg1,
               const RooCmdArg& arg2=RooCmdArg::none(), const RooCmdArg& arg3=RooCmdArg::none(),
               const RooCmdArg& arg4=RooCmdArg::none(), const RooCmdArg& arg5=RooCmdArg::none()) const {
    return generateBinned(whatVars,RooFit::NumEvents(nEvents),arg1,arg2,arg3,arg4,arg5);
  }
  virtual RooDataHist *generateBinned(const RooArgSet &whatVars,
               const RooCmdArg& arg1=RooCmdArg::none(),const RooCmdArg& arg2=RooCmdArg::none(),
               const RooCmdArg& arg3=RooCmdArg::none(),const RooCmdArg& arg4=RooCmdArg::none(),
               const RooCmdArg& arg5=RooCmdArg::none(),const RooCmdArg& arg6=RooCmdArg::none()) const;
  virtual RooDataHist *generateBinned(const RooArgSet &whatVars, double nEvents, bool expectedData=false, bool extended=false) const;

  virtual RooDataSet* generateSimGlobal(const RooArgSet& whatVars, Int_t nEvents) ;

  ///Helper calling plotOn(RooPlot*, RooLinkedList&) const
  RooPlot* plotOn(RooPlot* frame,
           const RooCmdArg& arg1=RooCmdArg::none(), const RooCmdArg& arg2=RooCmdArg::none(),
           const RooCmdArg& arg3=RooCmdArg::none(), const RooCmdArg& arg4=RooCmdArg::none(),
           const RooCmdArg& arg5=RooCmdArg::none(), const RooCmdArg& arg6=RooCmdArg::none(),
           const RooCmdArg& arg7=RooCmdArg::none(), const RooCmdArg& arg8=RooCmdArg::none(),
           const RooCmdArg& arg9=RooCmdArg::none(), const RooCmdArg& arg10=RooCmdArg::none()
              ) const override {
    return RooAbsReal::plotOn(frame,arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9,arg10) ;
  }
  RooPlot* plotOn(RooPlot* frame, RooLinkedList& cmdList) const override ;

  /// Add a box with parameter values (and errors) to the specified frame
  virtual RooPlot* paramOn(RooPlot* frame,
                           const RooCmdArg& arg1=RooCmdArg::none(), const RooCmdArg& arg2=RooCmdArg::none(),
                           const RooCmdArg& arg3=RooCmdArg::none(), const RooCmdArg& arg4=RooCmdArg::none(),
                           const RooCmdArg& arg5=RooCmdArg::none(), const RooCmdArg& arg6=RooCmdArg::none(),
                           const RooCmdArg& arg7=RooCmdArg::none(), const RooCmdArg& arg8=RooCmdArg::none()) ;

  virtual RooPlot* paramOn(RooPlot* frame, const RooAbsData* data, const char *label= "", Int_t sigDigits = 2,
            Option_t *options = "NELU", double xmin=0.65,
            double xmax = 0.9, double ymax = 0.9) ;

  // Built-in generator support
  virtual Int_t getGenerator(const RooArgSet& directVars, RooArgSet &generateVars, bool staticInitOK=true) const;
  virtual void initGenerator(Int_t code) ;
  virtual void generateEvent(Int_t code);
  virtual bool isDirectGenSafe(const RooAbsArg& arg) const ;

  // Configuration of MC generators used for this pdf
  const RooNumGenConfig* getGeneratorConfig() const ;
  static RooNumGenConfig* defaultGeneratorConfig()  ;
  RooNumGenConfig* specialGeneratorConfig() const ;
  RooNumGenConfig* specialGeneratorConfig(bool createOnTheFly) ;
  void setGeneratorConfig() ;
  void setGeneratorConfig(const RooNumGenConfig& config) ;

  // -log(L) fits to binned and unbinned data
  virtual RooFitResult* fitTo(RooAbsData& data, const RooCmdArg& arg1=RooCmdArg::none(),  const RooCmdArg& arg2=RooCmdArg::none(),
                              const RooCmdArg& arg3=RooCmdArg::none(),  const RooCmdArg& arg4=RooCmdArg::none(), const RooCmdArg& arg5=RooCmdArg::none(),
                              const RooCmdArg& arg6=RooCmdArg::none(),  const RooCmdArg& arg7=RooCmdArg::none(), const RooCmdArg& arg8=RooCmdArg::none()) ;
  virtual RooFitResult* fitTo(RooAbsData& data, const RooLinkedList& cmdList) ;

  /// Configuration struct for RooAbsPdf::minimizeNLL with all the default
  //values that also should be taked as the default values for
  //RooAbsPdf::fitTo.
  struct MinimizerConfig {
      double recoverFromNaN = 10.;
      int optConst = 2;
      int verbose = 0;
      int doSave = 0;
      int doTimer = 0;
      int printLevel = 1;
      int strat = 1;
      int initHesse = 0;
      int hesse = 1;
      int minos = 0;
      int numee = 10;
      int doEEWall = 1;
      int doWarn = 1;
      int doSumW2 = -1;
      int doAsymptotic = -1;
      const RooArgSet* minosSet = nullptr;
      std::string minType;
      std::string minAlg = "minuit";
  };
  std::unique_ptr<RooFitResult> minimizeNLL(RooAbsReal & nll, RooAbsData const& data, MinimizerConfig const& cfg);

  virtual RooAbsReal* createNLL(RooAbsData& data, const RooLinkedList& cmdList) ;
  virtual RooAbsReal* createNLL(RooAbsData& data, const RooCmdArg& arg1=RooCmdArg::none(),  const RooCmdArg& arg2=RooCmdArg::none(),
            const RooCmdArg& arg3=RooCmdArg::none(),  const RooCmdArg& arg4=RooCmdArg::none(), const RooCmdArg& arg5=RooCmdArg::none(),
            const RooCmdArg& arg6=RooCmdArg::none(),  const RooCmdArg& arg7=RooCmdArg::none(), const RooCmdArg& arg8=RooCmdArg::none()) ;

  // Chi^2 fits to histograms
  using RooAbsReal::chi2FitTo ;
  using RooAbsReal::createChi2 ;
  RooFitResult* chi2FitTo(RooDataHist& data, const RooLinkedList& cmdList) override ;
  RooAbsReal* createChi2(RooDataHist& data, const RooCmdArg& arg1=RooCmdArg::none(),  const RooCmdArg& arg2=RooCmdArg::none(),
             const RooCmdArg& arg3=RooCmdArg::none(),  const RooCmdArg& arg4=RooCmdArg::none(), const RooCmdArg& arg5=RooCmdArg::none(),
             const RooCmdArg& arg6=RooCmdArg::none(),  const RooCmdArg& arg7=RooCmdArg::none(), const RooCmdArg& arg8=RooCmdArg::none()) override ;

  // Chi^2 fits to X-Y datasets
  RooAbsReal* createChi2(RooDataSet& data, const RooLinkedList& cmdList) override ;


  // Constraint management
  virtual RooArgSet* getConstraints(const RooArgSet& /*observables*/, RooArgSet& /*constrainedParams*/, bool /*stripDisconnected*/) const {
    // Interface to retrieve constraint terms on this pdf. Default implementation returns null
    return nullptr ;
  }
  virtual RooArgSet* getAllConstraints(const RooArgSet& observables, RooArgSet& constrainedParams, bool stripDisconnected=true) const ;

  // Project p.d.f into lower dimensional p.d.f
  virtual RooAbsPdf* createProjection(const RooArgSet& iset) ;

  // Create cumulative density function from p.d.f
  RooAbsReal* createCdf(const RooArgSet& iset, const RooArgSet& nset=RooArgSet()) ;
  RooAbsReal* createCdf(const RooArgSet& iset, const RooCmdArg& arg1, const RooCmdArg& arg2=RooCmdArg::none(),
         const RooCmdArg& arg3=RooCmdArg::none(), const RooCmdArg& arg4=RooCmdArg::none(),
         const RooCmdArg& arg5=RooCmdArg::none(), const RooCmdArg& arg6=RooCmdArg::none(),
         const RooCmdArg& arg7=RooCmdArg::none(), const RooCmdArg& arg8=RooCmdArg::none()) ;
  RooAbsReal* createScanCdf(const RooArgSet& iset, const RooArgSet& nset, Int_t numScanBins, Int_t intOrder) ;

  // Function evaluation support
  double getValV(const RooArgSet* set=nullptr) const override ;
  virtual double getLogVal(const RooArgSet* set=nullptr) const ;

  RooSpan<const double> getValues(RooBatchCompute::RunContext& evalData, const RooArgSet* normSet) const override;
  using RooAbsReal::getValues;
  RooSpan<const double> getLogValBatch(std::size_t begin, std::size_t batchSize,
      const RooArgSet* normSet = nullptr) const;
  RooSpan<const double> getLogProbabilities(RooBatchCompute::RunContext& evalData, const RooArgSet* normSet = nullptr) const;
  void getLogProbabilities(RooSpan<const double> pdfValues, double * output) const;

  /// \copydoc getNorm(const RooArgSet*) const
  double getNorm(const RooArgSet& nset) const {
    return getNorm(&nset) ;
  }
  virtual double getNorm(const RooArgSet* set=nullptr) const ;

  virtual void resetErrorCounters(Int_t resetValue=10) ;
  void setTraceCounter(Int_t value, bool allNodes=false) ;

  double analyticalIntegralWN(Int_t code, const RooArgSet* normSet, const char* rangeName=nullptr) const override ;

  /// Shows if a PDF is self-normalized, which means that no attempt is made to add a normalization term.
  /// Always returns false, unless a PDF overrides this function.
  virtual bool selfNormalized() const {
    return false ;
  }

  // Support for extended maximum likelihood, switched off by default
  enum ExtendMode { CanNotBeExtended, CanBeExtended, MustBeExtended } ;
  /// Returns ability of PDF to provide extended likelihood terms. Possible
  /// answers are in the enumerator RooAbsPdf::ExtendMode.
  /// This default implementation always returns CanNotBeExtended.
  virtual ExtendMode extendMode() const { return CanNotBeExtended; }
  /// If true, PDF can provide extended likelihood term.
  inline bool canBeExtended() const {
    return (extendMode() != CanNotBeExtended) ;
  }
  /// If true PDF must provide extended likelihood term.
  inline bool mustBeExtended() const {
    return (extendMode() == MustBeExtended) ;
  }
  /// Return expected number of events to be used in calculation of extended
  /// likelihood.
  virtual double expectedEvents(const RooArgSet* nset) const ;
  /// Return expected number of events to be used in calculation of extended
  /// likelihood. This function should not be overridden, as it just redirects
  /// to the actual virtual function but takes a RooArgSet reference instead of
  /// pointer (\see expectedEvents(const RooArgSet*) const).
  double expectedEvents(const RooArgSet& nset) const {
    return expectedEvents(&nset) ;
  }

  // Printing interface (human readable)
  void printValue(std::ostream& os) const override ;
  void printMultiline(std::ostream& os, Int_t contents, bool verbose=false, TString indent="") const override ;

  static void verboseEval(Int_t stat) ;
  static int verboseEval() ;

  double extendedTerm(double sumEntries, double expected, double sumEntriesW2=0.0) const;
  double extendedTerm(double sumEntries, RooArgSet const* nset, double sumEntriesW2=0.0) const;
  double extendedTerm(RooAbsData const& data, bool weightSquared) const;

  void setNormRange(const char* rangeName) ;
  const char* normRange() const {
    return _normRange.Length()>0 ? _normRange.Data() : nullptr ;
  }
  void setNormRangeOverride(const char* rangeName) ;

  const RooAbsReal* getNormIntegral(const RooArgSet& nset) const { return getNormObj(nullptr,&nset,nullptr) ; }

  virtual const RooAbsReal* getNormObj(const RooArgSet* set, const RooArgSet* iset, const TNamed* rangeName=nullptr) const ;

  virtual RooAbsGenContext* binnedGenContext(const RooArgSet &vars, bool verbose= false) const ;

  virtual RooAbsGenContext* genContext(const RooArgSet &vars, const RooDataSet *prototype=nullptr,
                                  const RooArgSet* auxProto=nullptr, bool verbose= false) const ;

  virtual RooAbsGenContext* autoGenContext(const RooArgSet &vars, const RooDataSet* prototype=nullptr, const RooArgSet* auxProto=nullptr,
                  bool verbose=false, bool autoBinned=true, const char* binnedTag="") const ;

private:

  RooDataSet *generate(RooAbsGenContext& context, const RooArgSet& whatVars, const RooDataSet* prototype,
             double nEvents, bool verbose, bool randProtoOrder, bool resampleProto, bool skipInit=false,
             bool extended=false) const ;

  // Implementation version
  virtual RooPlot* paramOn(RooPlot* frame, const RooArgSet& params, bool showConstants=false,
                           const char *label= "", Int_t sigDigits = 2, Option_t *options = "NELU", double xmin=0.65,
            double xmax= 0.99,double ymax=0.95, const RooCmdArg* formatCmd=nullptr) ;

  void logBatchComputationErrors(RooSpan<const double>& outputs, std::size_t begin) const;
  bool traceEvalPdf(double value) const;

  /// Setter for the _normSet member, which should never be set directly.
  inline void setActiveNormSet(RooArgSet const* normSet) const {
    _normSet = normSet;
    // Also store the unique ID of the _normSet. This makes it possible to
    // detect if the pointer was invalidated.
    _normSetId = RooFit::getUniqueId(normSet);
  }

protected:

  /// Checks if `normSet` is the currently active normalization set of this
  /// PDF, meaning is exactly the same object as the one the `_normSet` member
  /// points to (or both are `nullptr`).
  inline bool isActiveNormSet(RooArgSet const* normSet) const {
    return RooFit::getUniqueId(normSet).value() == _normSetId;
  }

  double normalizeWithNaNPacking(double rawVal, double normVal) const;

  RooPlot *plotOn(RooPlot *frame, PlotOpt o) const override;

  friend class RooMCStudy ;

  Int_t* randomizeProtoOrder(Int_t nProto,Int_t nGen,bool resample=false) const ;

  // This also forces the definition of a copy ctor in derived classes
  RooAbsPdf(const RooAbsPdf& other, const char* name = nullptr);

  static Int_t _verboseEval ;

  virtual bool syncNormalization(const RooArgSet* dset, bool adjustProxies=true) const ;

  mutable double _rawValue ;
  mutable RooAbsReal* _norm = nullptr; //! Normalization integral (owned by _normMgr)
  mutable RooArgSet const* _normSet = nullptr; //! Normalization set with for above integral

  class CacheElem : public RooAbsCacheElement {
  public:
    CacheElem(RooAbsReal& norm) : _norm(&norm) {} ;
    ~CacheElem() override ;
    RooArgList containedArgs(Action) override { return RooArgList(*_norm) ; }
    RooAbsReal* _norm ;
  } ;
  mutable RooObjCacheManager _normMgr ; //! The cache manager

  bool redirectServersHook(const RooAbsCollection & newServerList, bool mustReplaceAll,
                                   bool nameChange, bool isRecursiveStep) override;


  mutable Int_t _errorCount ;        ///< Number of errors remaining to print
  mutable Int_t _traceCount ;        ///< Number of traces remaining to print
  mutable Int_t _negCount ;          ///< Number of negative probablities remaining to print

  bool _selectComp ;               ///< Component selection flag for RooAbsPdf::plotCompOn

  RooNumGenConfig* _specGeneratorConfig ; ///<! MC generator configuration specific for this object

  TString _normRange ; ///< Normalization range
  static TString _normRangeOverride ;

private:
  mutable RooFit::UniqueId<RooArgSet>::Value_t _normSetId = RooFit::UniqueId<RooArgSet>::nullval; ///<! Unique ID of the currently-active normalization set

  int calcAsymptoticCorrectedCovariance(RooMinimizer& minimizer, RooAbsData const& data);
  int calcSumW2CorrectedCovariance(RooMinimizer& minimizer, RooAbsReal & nll) const;

  ClassDefOverride(RooAbsPdf,5) // Abstract PDF with normalization support
};




#endif
