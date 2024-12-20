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

#include <RooAbsReal.h>
#include <RooDataHist.h>
#include <RooDataSet.h>
#include <RooFit/UniqueId.h>
#include <RooGlobalFunc.h>
#include <RooObjCacheManager.h>

class RooArgSet ;
class RooAbsGenContext ;
class RooFitResult ;
class RooExtendPdf ;
class RooCategory ;
class TPaveText;
class TH1F;
class TH2F;
class TList ;
class RooMinimizer ;
class RooNumGenConfig ;
class RooRealIntegral ;


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
  RooFit::OwningPtr<RooDataSet> generate(const RooArgSet &whatVars, Int_t nEvents, const RooCmdArg& arg1,
                       const RooCmdArg& arg2={}, const RooCmdArg& arg3={},
                       const RooCmdArg& arg4={}, const RooCmdArg& arg5={}) {
    return generate(whatVars,RooFit::NumEvents(nEvents),arg1,arg2,arg3,arg4,arg5) ;
  }
  RooFit::OwningPtr<RooDataSet> generate(const RooArgSet &whatVars,
                       const RooCmdArg& arg1={},const RooCmdArg& arg2={},
                       const RooCmdArg& arg3={},const RooCmdArg& arg4={},
                       const RooCmdArg& arg5={},const RooCmdArg& arg6={}) ;
  RooFit::OwningPtr<RooDataSet> generate(const RooArgSet &whatVars, double nEvents = 0, bool verbose=false, bool autoBinned=true,
             const char* binnedTag="", bool expectedData=false, bool extended = false) const;
  RooFit::OwningPtr<RooDataSet> generate(const RooArgSet &whatVars, const RooDataSet &prototype, Int_t nEvents= 0,
             bool verbose=false, bool randProtoOrder=false, bool resampleProto=false) const;


  class GenSpec {
  public:
    virtual ~GenSpec() ;
    GenSpec() = default;

  private:
    GenSpec(RooAbsGenContext* context, const RooArgSet& whatVars, RooDataSet* protoData, Int_t nGen, bool extended,
       bool randProto, bool resampleProto, TString dsetName, bool init=false) ;
    GenSpec(const GenSpec& other) ;

    friend class RooAbsPdf ;
    std::unique_ptr<RooAbsGenContext> _genContext;
    RooArgSet _whatVars ;
    RooDataSet* _protoData = nullptr;
    Int_t _nGen = 0;
    bool _extended = false;
    bool _randProto = false;
    bool _resampleProto = false;
    TString _dsetName ;
    bool _init = false;

    ClassDef(GenSpec,0) // Generation specification
  } ;

  ///Prepare GenSpec configuration object for efficient generation of multiple datasets from identical specification.
  GenSpec* prepareMultiGen(const RooArgSet &whatVars,
            const RooCmdArg& arg1={},const RooCmdArg& arg2={},
            const RooCmdArg& arg3={},const RooCmdArg& arg4={},
            const RooCmdArg& arg5={},const RooCmdArg& arg6={}) ;
  ///Generate according to GenSpec obtained from prepareMultiGen().
  RooFit::OwningPtr<RooDataSet> generate(GenSpec&) const ;


  ////////////////////////////////////////////////////////////////////////////////
  /// As RooAbsPdf::generateBinned(const RooArgSet&, const RooCmdArg&,const RooCmdArg&, const RooCmdArg&,const RooCmdArg&, const RooCmdArg&,const RooCmdArg&) const.
  /// \param[in] whatVars set
  /// \param[in] nEvents How many events to generate
  /// \param arg1,arg2,arg3,arg4,arg5 ordered arguments
  virtual RooFit::OwningPtr<RooDataHist> generateBinned(const RooArgSet &whatVars, double nEvents, const RooCmdArg& arg1,
               const RooCmdArg& arg2={}, const RooCmdArg& arg3={},
               const RooCmdArg& arg4={}, const RooCmdArg& arg5={}) const {
    return generateBinned(whatVars,RooFit::NumEvents(nEvents),arg1,arg2,arg3,arg4,arg5);
  }
  virtual RooFit::OwningPtr<RooDataHist> generateBinned(const RooArgSet &whatVars,
               const RooCmdArg& arg1={},const RooCmdArg& arg2={},
               const RooCmdArg& arg3={},const RooCmdArg& arg4={},
               const RooCmdArg& arg5={},const RooCmdArg& arg6={}) const;
  virtual RooFit::OwningPtr<RooDataHist> generateBinned(const RooArgSet &whatVars, double nEvents, bool expectedData=false, bool extended=false) const;

  virtual RooFit::OwningPtr<RooDataSet> generateSimGlobal(const RooArgSet& whatVars, Int_t nEvents) ;

  ///Helper calling plotOn(RooPlot*, RooLinkedList&) const
  RooPlot* plotOn(RooPlot* frame,
           const RooCmdArg& arg1={}, const RooCmdArg& arg2={},
           const RooCmdArg& arg3={}, const RooCmdArg& arg4={},
           const RooCmdArg& arg5={}, const RooCmdArg& arg6={},
           const RooCmdArg& arg7={}, const RooCmdArg& arg8={},
           const RooCmdArg& arg9={}, const RooCmdArg& arg10={}
              ) const override {
    return RooAbsReal::plotOn(frame,arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9,arg10) ;
  }
  RooPlot* plotOn(RooPlot* frame, RooLinkedList& cmdList) const override ;

  /// Add a box with parameter values (and errors) to the specified frame
  virtual RooPlot* paramOn(RooPlot* frame,
                           const RooCmdArg& arg1={}, const RooCmdArg& arg2={},
                           const RooCmdArg& arg3={}, const RooCmdArg& arg4={},
                           const RooCmdArg& arg5={}, const RooCmdArg& arg6={},
                           const RooCmdArg& arg7={}, const RooCmdArg& arg8={}) ;

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

  template <typename... CmdArgs_t>
  RooFit::OwningPtr<RooFitResult> fitTo(RooAbsData& data, CmdArgs_t const&... cmdArgs)
  {
    return RooFit::makeOwningPtr(fitToImpl(data, *RooFit::Detail::createCmdList(&cmdArgs...)));
  }

  template <typename... CmdArgs_t>
  RooFit::OwningPtr<RooAbsReal> createNLL(RooAbsData& data, CmdArgs_t const&... cmdArgs)
  {
    return RooFit::makeOwningPtr(createNLLImpl(data, *RooFit::Detail::createCmdList(&cmdArgs...)));
  }

  // Constraint management
  virtual RooArgSet* getConstraints(const RooArgSet& /*observables*/, RooArgSet const& /*constrainedParams*/, RooArgSet& /*pdfParams*/) const
  {
    // Interface to retrieve constraint terms on this pdf. Default implementation returns null
    return nullptr ;
  }
  RooArgSet* getAllConstraints(const RooArgSet& observables, RooArgSet& constrainedParams,
                               bool stripDisconnected=true) const ;

  // Project p.d.f into lower dimensional p.d.f
  virtual RooAbsPdf* createProjection(const RooArgSet& iset) ;

  // Create cumulative density function from p.d.f
  RooFit::OwningPtr<RooAbsReal> createCdf(const RooArgSet& iset, const RooArgSet& nset=RooArgSet()) ;
  RooFit::OwningPtr<RooAbsReal> createCdf(const RooArgSet& iset, const RooCmdArg& arg1, const RooCmdArg& arg2={},
         const RooCmdArg& arg3={}, const RooCmdArg& arg4={},
         const RooCmdArg& arg5={}, const RooCmdArg& arg6={},
         const RooCmdArg& arg7={}, const RooCmdArg& arg8={}) ;
  RooFit::OwningPtr<RooAbsReal> createScanCdf(const RooArgSet& iset, const RooArgSet& nset, Int_t numScanBins, Int_t intOrder) ;

  // Function evaluation support
  double getValV(const RooArgSet* set=nullptr) const override ;
  virtual double getLogVal(const RooArgSet* set=nullptr) const ;

  void getLogProbabilities(std::span<const double> pdfValues, double * output) const;

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
  /// pointer. \see expectedEvents(const RooArgSet*) const
  double expectedEvents(const RooArgSet& nset) const {
    return expectedEvents(&nset) ;
  }

  virtual std::unique_ptr<RooAbsReal> createExpectedEventsFunc(const RooArgSet* nset) const;

  // Printing interface (human readable)
  void printValue(std::ostream& os) const override ;
  void printMultiline(std::ostream& os, Int_t contents, bool verbose=false, TString indent="") const override ;

  static void verboseEval(Int_t stat) ;
  static int verboseEval() ;

  double extendedTerm(double sumEntries, double expected, double sumEntriesW2=0.0, bool doOffset=false) const;
  double extendedTerm(double sumEntries, RooArgSet const* nset, double sumEntriesW2=0.0, bool doOffset=false) const;
  double extendedTerm(RooAbsData const& data, bool weightSquared, bool doOffset=false) const;

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

  std::unique_ptr<RooAbsArg> compileForNormSet(RooArgSet const &normSet, RooFit::Detail::CompileContext & ctx) const override;

private:

  std::unique_ptr<RooDataSet> generate(RooAbsGenContext& context, const RooArgSet& whatVars, const RooDataSet* prototype,
             double nEvents, bool verbose, bool randProtoOrder, bool resampleProto, bool skipInit=false,
             bool extended=false) const ;

  // Implementation version
  virtual RooPlot* paramOn(RooPlot* frame, const RooArgSet& params, bool showConstants=false,
                           const char *label= "", double xmin=0.65,
                           double xmax= 0.99,double ymax=0.95, const RooCmdArg* formatCmd=nullptr) ;

  void logBatchComputationErrors(std::span<const double>& outputs, std::size_t begin) const;
  bool traceEvalPdf(double value) const;

  /// Setter for the _normSet member, which should never be set directly.
  inline void setActiveNormSet(RooArgSet const* normSet) const {
    _normSet = normSet;
    // Also store the unique ID of the _normSet. This makes it possible to
    // detect if the pointer was invalidated.
    _normSetId = RooFit::getUniqueId(normSet);
  }

protected:

  virtual std::unique_ptr<RooAbsReal> createNLLImpl(RooAbsData& data, const RooLinkedList& cmdList);
  virtual std::unique_ptr<RooFitResult> fitToImpl(RooAbsData& data, const RooLinkedList& cmdList);

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

  mutable double _rawValue = 0;
  mutable RooAbsReal* _norm = nullptr; //! Normalization integral (owned by _normMgr)
  mutable RooArgSet const* _normSet = nullptr; //! Normalization set with for above integral

  class CacheElem : public RooAbsCacheElement {
  public:
    CacheElem(RooAbsReal& norm) : _norm(&norm) {} ;
    ~CacheElem() override ;
    RooArgList containedArgs(Action) override { return RooArgList(*_norm) ; }
    std::unique_ptr<RooAbsReal> _norm;
  } ;
  mutable RooObjCacheManager _normMgr ; //! The cache manager

  bool redirectServersHook(const RooAbsCollection & newServerList, bool mustReplaceAll,
                                   bool nameChange, bool isRecursiveStep) override;

  mutable Int_t _errorCount = 0; ///< Number of errors remaining to print
  mutable Int_t _traceCount = 0; ///< Number of traces remaining to print
  mutable Int_t _negCount = 0;   ///< Number of negative probabilities remaining to print

  bool _selectComp = false; ///< Component selection flag for RooAbsPdf::plotCompOn

  std::unique_ptr<RooNumGenConfig> _specGeneratorConfig ; ///<! MC generator configuration specific for this object

  TString _normRange ; ///< Normalization range
  static TString _normRangeOverride ;

private:
  mutable RooFit::UniqueId<RooArgSet>::Value_t _normSetId = RooFit::UniqueId<RooArgSet>::nullval; ///<! Unique ID of the currently-active normalization set

  friend class RooAbsReal;
  friend class RooChi2Var;

  ClassDefOverride(RooAbsPdf,5) // Abstract PDF with normalization support
};




#endif
