/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooAbsReal.rdl,v 1.58 2003/05/12 20:25:51 wverkerke Exp $
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2002, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/
#ifndef ROO_ABS_REAL
#define ROO_ABS_REAL

#include "RooFitCore/RooAbsArg.hh"
#include "RooFitCore/RooCmdArg.hh"
#include "RooFitCore/RooCurve.hh"

class RooArgSet ;
class RooArgList ;
class RooDataSet ;
class RooPlot;
class RooRealVar;
class RooAbsFunc;
class RooAbsCategoryLValue ;
class RooCategory ;
class RooLinkedList ;
class RooIntegratorConfig ;

class TH1;
class TH1F;
class TH2F;
class TH3F;

class RooAbsReal : public RooAbsArg {
public:
  // Constructors, assignment etc
  inline RooAbsReal() ;
  RooAbsReal(const char *name, const char *title, const char *unit= "") ;
  RooAbsReal(const char *name, const char *title, Double_t minVal, Double_t maxVal, 
	     const char *unit= "") ;
  RooAbsReal(const RooAbsReal& other, const char* name=0);
  virtual ~RooAbsReal();

  // Return value and unit accessors
  virtual Double_t getVal(const RooArgSet* set=0) const ;
  inline  Double_t getVal(const RooArgSet& set) const { return getVal(&set) ; }
  Bool_t operator==(Double_t value) const ;
  virtual Bool_t operator==(const RooAbsArg& other) ;
  inline const Text_t *getUnit() const { return _unit.Data(); }
  inline void setUnit(const char *unit) { _unit= unit; }
  TString getTitle(Bool_t appendUnit= kFALSE) const;

  // Lightweight interface adaptors (caller takes ownership)
  RooAbsFunc *bindVars(const RooArgSet &vars, const RooArgSet* nset=0, Bool_t clipInvalid=kFALSE) const;

  // Create a fundamental-type object that can hold our value.
  RooAbsArg *createFundamental(const char* newname=0) const;

  // Analytical integration support
  virtual Int_t getAnalyticalIntegralWN(RooArgSet& allVars, RooArgSet& analVars, const RooArgSet* normSet) const ;
  virtual Double_t analyticalIntegralWN(Int_t code, const RooArgSet* normSet) const ;
  virtual Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars) const ;
  virtual Double_t analyticalIntegral(Int_t code) const ;
  virtual Bool_t forceAnalyticalInt(const RooAbsArg& dep) const { return kFALSE ; }
  virtual void forceNumInt(Bool_t flag=kTRUE) { _forceNumInt = flag ; }

  RooAbsReal* createIntegral(const RooArgSet& iset, const RooArgSet& nset) const { return createIntegral(iset,&nset) ; }
  RooAbsReal* createIntegral(const RooArgSet& iset, const RooArgSet& nset, RooIntegratorConfig& cfg) const { return createIntegral(iset,&nset,&cfg) ; }
  RooAbsReal* createIntegral(const RooArgSet& iset, const RooIntegratorConfig& cfg) const { return createIntegral(iset,0,&cfg) ; }
  virtual RooAbsReal* createIntegral(const RooArgSet& iset, const RooArgSet* nset=0, const RooIntegratorConfig* cfg=0) const ;  

  // Plotting options
  inline Double_t getPlotMin() const { return _plotMin; }
  inline Double_t getPlotMax() const { return _plotMax; }
  virtual Int_t getPlotBins() const { return _plotBins; }
  void setPlotMin(Double_t value) ;
  void setPlotMax(Double_t value) ;
  void setPlotRange(Double_t min, Double_t max) ;
  void setPlotBins(Int_t value) ; 
  void setPlotLabel(const char *label);
  const char *getPlotLabel() const;
  virtual Bool_t inPlotRange(Double_t value) const;

  virtual Double_t defaultErrorLevel() const { return 1.0 ; }

  const RooIntegratorConfig* getIntegratorConfig() const ;
  static RooIntegratorConfig* defaultIntegratorConfig()  ;
  RooIntegratorConfig* specialIntegratorConfig() const ;
  static void setDefaultIntegratorConfig(const RooIntegratorConfig& config) ;
  void setIntegratorConfig() ;
  void setIntegratorConfig(const RooIntegratorConfig& config) ;


public:

  // User entry point for plotting
  enum ScaleType { Raw, Relative, NumEvent, RelativeExpected } ;
  virtual RooPlot* plotOn(RooPlot* frame, 
			  const RooCmdArg& arg1            , const RooCmdArg& arg2=RooCmdArg(),
			  const RooCmdArg& arg3=RooCmdArg(), const RooCmdArg& arg4=RooCmdArg(),
			  const RooCmdArg& arg5=RooCmdArg(), const RooCmdArg& arg6=RooCmdArg(),
			  const RooCmdArg& arg7=RooCmdArg(), const RooCmdArg& arg8=RooCmdArg()) const ;

  // Plot implementation functions
  virtual RooPlot *plotOn(RooPlot *frame, Option_t* drawOptions="L", Double_t scaleFactor=1.0, 
			  ScaleType stype=Relative, const RooAbsData* projData=0, const RooArgSet* projSet=0,
			  Double_t precision=1e-3, Bool_t shiftToZero=kFALSE, const RooArgSet* projDataSet=0,
			  Double_t rangeLo=0, Double_t rangeHi=0, RooCurve::WingMode wmode=RooCurve::Extended) const;
  virtual RooPlot *plotAsymOn(RooPlot *frame, const RooAbsCategoryLValue& asymCat, Option_t* drawOptions="L", 
			      Double_t scaleFactor=1.0, const RooAbsData* projData=0, const RooArgSet* projSet=0,
			      Double_t precision=1e-3, const RooArgSet* projDataSet=0, 
			      Double_t rangeLo=0, Double_t rangeHi=0, RooCurve::WingMode wmode=RooCurve::Extended) const;

  // Forwarder function for backward compatibility
  virtual RooPlot *plotSliceOn(RooPlot *frame, const RooArgSet& sliceSet, Option_t* drawOptions="L", 
			       Double_t scaleFactor=1.0, ScaleType stype=Relative, const RooAbsData* projData=0) const;

  // Fill an existing histogram
  TH1 *fillHistogram(TH1 *hist, const RooArgList &plotVars,
		     Double_t scaleFactor= 1, const RooArgSet *projectedVars= 0) const;

  // I/O streaming interface (machine readable)
  virtual Bool_t readFromStream(istream& is, Bool_t compact, Bool_t verbose=kFALSE) ;
  virtual void writeToStream(ostream& os, Bool_t compact) const ;

  // Printing interface (human readable)
  virtual void printToStream(ostream& stream, PrintOption opt=Standard, TString indent= "") const ;

  const RooAbsReal* createProjection(const RooArgSet& depVars, const RooArgSet& projVars) const ;
  const RooAbsReal* createProjection(const RooArgSet& depVars, const RooArgSet& projVars, RooArgSet*& cloneSet) const ;

protected:

  // PlotOn with command list
  virtual RooPlot* plotOn(RooPlot* frame, RooLinkedList& cmdList) const ;

  // Hook for objects with normalization-dependent parameters interperetation
  virtual void selectNormalization(const RooArgSet* depSet=0, Bool_t force=kFALSE) {} ;

  // Helper functions for plotting
  Bool_t plotSanityChecks(RooPlot* frame) const ;
  void makeProjectionSet(const RooAbsArg* plotVar, const RooArgSet* allVars, 
			 RooArgSet& projectedVars, Bool_t silent) const ;

  const RooAbsReal *createProjection(const RooArgSet &dependentVars, const RooArgSet *projectedVars,
				     RooArgSet *&cloneSet) const;

  // Support interface for subclasses to advertise their analytic integration
  // and generator capabilities in their analticalIntegral() and generateEvent()
  // implementations.
  Bool_t matchArgs(const RooArgSet& allDeps, RooArgSet& numDeps, 
		   const RooArgProxy& a) const ;
  Bool_t matchArgs(const RooArgSet& allDeps, RooArgSet& numDeps, 
		   const RooArgProxy& a, const RooArgProxy& b) const ;
  Bool_t matchArgs(const RooArgSet& allDeps, RooArgSet& numDeps, 
		   const RooArgProxy& a, const RooArgProxy& b, const RooArgProxy& c) const ;
  Bool_t matchArgs(const RooArgSet& allDeps, RooArgSet& numDeps, 
		   const RooArgProxy& a, const RooArgProxy& b, 		   
		   const RooArgProxy& c, const RooArgProxy& d) const ;

  Bool_t matchArgs(const RooArgSet& allDeps, RooArgSet& numDeps, 
		   const RooArgSet& set) const ;

  // Internal consistency checking (needed by RooDataSet)
  virtual Bool_t isValid() const ;
  virtual Bool_t isValidReal(Double_t value, Bool_t printError=kFALSE) const ;

  // Function evaluation and error tracing
  Double_t traceEval(const RooArgSet* set) const ;
  virtual Bool_t traceEvalHook(Double_t value) const { return kFALSE ;}
  virtual Double_t evaluate() const = 0 ;

  // Hooks for RooDataSet interface
  friend class RooRealIntegral ;
  virtual void syncCache(const RooArgSet* set=0) { getVal(set) ; }
  virtual void copyCache(const RooAbsArg* source) ;
  virtual void attachToTree(TTree& t, Int_t bufSize=32000) ;
  virtual void setTreeBranchStatus(TTree& t, Bool_t active) ;
  virtual void fillTreeBranch(TTree& t) ;
  TString cleanBranchName() const ;
  UInt_t crc32(const char* data) const ;

  Double_t _plotMin ;       // Minimum of plot range
  Double_t _plotMax ;       // Maximum of plot range
  Int_t    _plotBins ;      // Number of plot bins
  mutable Double_t _value ; // Cache for current value of object
  TString  _unit ;          // Unit for objects value
  TString  _label ;         // Plot label for objects value
  Bool_t   _forceNumInt ;   // Force numerical integration if flag set

  friend class RooAbsPdf ;
  friend class RooConvolutedPdf ;
  friend class RooRealProxy ;

  RooIntegratorConfig* _specIntegratorConfig ; //!
  static RooIntegratorConfig* _defaultIntegratorConfig ;

private:

  Bool_t matchArgsByName(const RooArgSet &allArgs, RooArgSet &matchedArgs, const TList &nameList) const;

protected:

  ClassDef(RooAbsReal,1) // Abstract real-valued variable
};

// RooAbsReal::plotOn arguments
RooCmdArg DrawOption(const char* opt) ;
RooCmdArg Normalization(Double_t scaleFactor) ;
RooCmdArg Slice(const RooArgSet& sliceSet) ;
RooCmdArg Project(const RooArgSet& projSet) ;
RooCmdArg ProjWData(const RooAbsData& projData) ;
RooCmdArg ProjWData(const RooArgSet& projSet, const RooAbsData& projData) ;
RooCmdArg Asymmetry(const RooCategory& cat) ;
RooCmdArg Precision(Double_t prec) ;
RooCmdArg ShiftToZero() ;
RooCmdArg Range(Double_t lo, Double_t hi, Bool_t vlines=kFALSE) ;
RooCmdArg LineColor(Color_t color) ;
RooCmdArg LineStyle(Style_t style) ;
RooCmdArg LineWidth(Width_t width) ;
RooCmdArg FillColor(Color_t color) ;
RooCmdArg FillStyle(Style_t style) ;


#endif
