/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooRealVar.rdl,v 1.10 2001/04/08 00:06:49 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/
#ifndef ROO_REAL_VAR
#define ROO_REAL_VAR

#include <iostream.h>

#include "TString.h"

#include "RooFitCore/RooAbsReal.hh"
class RooArgSet ;

class RooRealVar : public RooAbsReal {
public:
  // Constructors, assignment etc.
  inline RooRealVar() { }
  RooRealVar(const char *name, const char *title,
  	   Double_t value, const char *unit= "") ;
  RooRealVar(const char *name, const char *title, Double_t minValue, 
	   Double_t maxValue, const char *unit= "");
  RooRealVar(const char *name, const char *title, Double_t value, 
	   Double_t minValue, Double_t maxValue, const char *unit= "") ;
  RooRealVar(const RooRealVar& other);
  RooRealVar(const char* name, const RooRealVar& other);
  virtual TObject* Clone() { return new RooRealVar(*this); }
  virtual ~RooRealVar();
  RooRealVar& operator=(const RooRealVar& other) ;
  
  // Parameter value and error accessors
  virtual operator Double_t&();
  virtual operator Double_t() const ;
  inline Double_t getError() const { return _error; }
  virtual void setVal(Double_t value);
  inline void setError(Double_t value) { _error= value; }
  virtual Double_t operator=(Double_t newValue);

  // Fit limits
  inline Double_t getFitMin() const { return _fitMin ; }
  inline Double_t getFitMax() const { return _fitMax ; }
  void setFitMin(Double_t value) ;
  void setFitMax(Double_t value) ;
  void setFitRange(Double_t min, Double_t max) ;
  Bool_t inFitRange(Double_t value, Double_t* clippedValue=0) const;
  Bool_t hasFitLimits() const { return kTRUE ; }

  // Constant and Projected flags 
  inline Bool_t isConstant() const { return getAttribute("Constant") ; }
  inline void setConstant(Bool_t value= kTRUE) { setAttribute("Constant",value); }
  inline Bool_t isProjected() const { return getAttribute("Projected") ; }
  inline void setProjected(Bool_t value= kTRUE) { setAttribute("Projected",value);}

  // I/O streaming interface (machine readable)
  virtual Bool_t readFromStream(istream& is, Bool_t compact, Bool_t verbose=kFALSE) ;
  virtual void writeToStream(ostream& os, Bool_t compact) const ;

  // Printing interface (human readable)
  virtual void printToStream(ostream& stream, PrintOption opt=Standard, TString indent= "") const ;
  TString* format(Int_t sigDigits, const char *options) ;

protected:

  virtual RooAbsArg& operator=(const RooAbsArg& other) ;

  virtual void attachToTree(TTree& t, Int_t bufSize=32000) ;
  Double_t chopAt(Double_t what, Int_t where) ;

  virtual Bool_t isValid(Double_t value, Bool_t verbose=kFALSE) const ;

  Double_t _fitMin ;
  Double_t _fitMax ;
  Double_t _error;

  ClassDef(RooRealVar,1) // a real-valued variable and its value
};

#endif
