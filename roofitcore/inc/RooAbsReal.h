/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooAbsReal.rdl,v 1.1 2001/03/17 00:32:54 verkerke Exp $
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
class TH1F ;

class RooAbsReal : public RooAbsArg {
public:

  // Constructors, assignment etc
  inline RooAbsReal() { }
  RooAbsReal(const char *name, const char *title, const char *unit= "") ;
  RooAbsReal(const char *name, const char *title, Double_t minVal, Double_t maxVal, const char *unit= "") ;
  RooAbsReal(const RooAbsReal& other);
  RooAbsArg& operator=(RooAbsArg& other) ;
  virtual ~RooAbsReal();

  // Return value and unit accessors
  virtual Double_t getVal() ;
  Bool_t operator==(Double_t value) ;
  inline const Text_t *getUnit() const { return _unit.Data(); }
  inline void setUnit(const char *unit) { _unit= unit; }

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
  TH1F *createHistogram(const char *label, const char *axis, Int_t bins= 0);
  TH1F *createHistogram(const char *label, const char *axis, Double_t lo, Double_t hi, Int_t bins= 0);

  // I/O streaming interface (machine readable)
  virtual Bool_t readFromStream(istream& is, Bool_t compact, Bool_t verbose=kFALSE) ;
  virtual void writeToStream(ostream& os, Bool_t compact) ;

  // Printing interface (human readable)
  virtual void printToStream(ostream& stream, PrintOption opt=Standard) ;

protected:
  friend class RooDataSet ;

  // Function evaluation and error tracing
  Double_t traceEval() ;
  virtual Bool_t traceEvalHook(Double_t value) {}
  virtual Double_t evaluate() { return 0 ; }

  // Internal consistency checking (needed by RooDataSet)
  virtual Bool_t isValid() ;
  virtual Bool_t isValid(Double_t value) ;

  Double_t _plotMin ;
  Double_t _plotMax ;
  Int_t    _plotBins ;
  Double_t _value ;
  TString  _unit ;
  TString  _label ;

  ClassDef(RooAbsReal,1) // a real-valued variable and its value
};

#endif
