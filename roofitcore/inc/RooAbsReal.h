/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooAbsReal.rdl,v 1.22 2001/08/03 21:44:56 david Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/
#ifndef ROO_ABS_REAL
#define ROO_ABS_REAL

#include "RooFitCore/RooAbsArg.hh"

class RooArgSet ;
class RooDataSet ;
class RooPlot;
class RooRealVar;
class RooRealFunc1D;
class RooAbsFunc;
class RooRealFixedBinIter ;
class TH1F;

class RooAbsReal : public RooAbsArg {
public:

  // Constructors, assignment etc
  inline RooAbsReal() { }
  RooAbsReal(const char *name, const char *title, const char *unit= "") ;
  RooAbsReal(const char *name, const char *title, Double_t minVal, Double_t maxVal, 
	     const char *unit= "") ;
  RooAbsReal(const RooAbsReal& other, const char* name=0);
  virtual ~RooAbsReal();

  // Return value and unit accessors
  virtual Double_t getVal(const RooArgSet* set=0) const ;
  Bool_t operator==(Double_t value) const ;
  inline const Text_t *getUnit() const { return _unit.Data(); }
  inline void setUnit(const char *unit) { _unit= unit; }

  // Lightweight interface adaptors (caller takes ownership)
  RooAbsFunc *bindVars(const RooArgSet &vars) const;

  // Create a fundamental-type object that can hold our value.
  RooAbsArg *createFundamental() const;

  // Analytical integration support
  virtual Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars) const ;
  virtual Double_t analyticalIntegral(Int_t code) const ;
  virtual Bool_t forceAnalyticalInt(const RooAbsArg& dep) const { return kFALSE ; }
  virtual Bool_t selfNormalized() const { return kFALSE ; }
  
  // Plotting options
  inline Double_t getPlotMin() const { return _plotMin; }
  inline Double_t getPlotMax() const { return _plotMax; }
  inline Int_t getPlotBins() const { return _plotBins; }
  void setPlotMin(Double_t value) ;
  void setPlotMax(Double_t value) ;
  void setPlotRange(Double_t min, Double_t max) ;
  void setPlotBins(Int_t value) ; 
  void setPlotLabel(const char *label);
  const char *getPlotLabel() const;
  virtual Bool_t inPlotRange(Double_t value) const;

  // Plotting, binned fit interface
  virtual Int_t getPlotBin() const ;
  virtual RooAbsBinIter* createPlotBinIterator() const ;
  virtual Double_t plotBinCenter(Int_t i) const ;
  virtual Double_t plotBinLow(Int_t i) const ;
  virtual Double_t plotBinHigh(Int_t i) const ;

  // Create plots
  RooPlot *frame() const;
  RooPlot *plotOn(RooPlot *frame, Option_t* drawOptions="L", Double_t scaleFactor= 1.0) const;

  // Create histograms
  TH1F *createHistogram(const char *label, const char *axis, Int_t bins= 0) const;
  TH1F *createHistogram(const char *label, const char *axis, Double_t lo, Double_t hi, Int_t bins) const;

  // I/O streaming interface (machine readable)
  virtual Bool_t readFromStream(istream& is, Bool_t compact, Bool_t verbose=kFALSE) ;
  virtual void writeToStream(ostream& os, Bool_t compact) const ;

  // Printing interface (human readable)
  virtual void printToStream(ostream& stream, PrintOption opt=Standard, TString indent= "") const ;

protected:

  // Internal consistency checking (needed by RooDataSet)
  virtual Bool_t isValid() const ;
  virtual Bool_t isValidReal(Double_t value, Bool_t printError=kFALSE) const ;

  // Function evaluation and error tracing
  Double_t traceEval(const RooArgSet* set) const ;
  virtual Bool_t traceEvalHook(Double_t value) const { return kFALSE ;}
  virtual Double_t evaluate(const RooArgSet* set) const = 0 ;

  // Hooks for RooDataSet interface
  virtual void syncCache(const RooArgSet* set=0) { getVal(set) ; }
  virtual void copyCache(const RooAbsArg* source) ;
  virtual void attachToTree(TTree& t, Int_t bufSize=32000) ;

  friend class RooRealFixedBinIter ;
  void calcBinWidth() ;
  Double_t getPlotBinWidth() { return _plotBinW ; }

  Double_t _plotMin ;
  Double_t _plotMax ;
  Int_t    _plotBins ;
  Double_t _plotBinW ;
  mutable Double_t _value ;
  TString  _unit ;
  TString  _label ;

  friend class RooAbsPdf ;
  friend class RooConvolutedPdf ;

  ClassDef(RooAbsReal,1) // Abstract real-valued variable
};

#endif
