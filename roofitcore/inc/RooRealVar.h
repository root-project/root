/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooRealVar.rdl,v 1.17 2001/05/14 05:22:55 verkerke Exp $
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
#include <math.h>
#include <float.h>
#include "TString.h"
#include "RooFitCore/RooAbsRealLValue.hh"

class RooArgSet ;

class RooRealVar : public RooAbsRealLValue {
public:
  // Constructors, assignment etc.
  inline RooRealVar() { }
  RooRealVar(const char *name, const char *title,
  	   Double_t value, const char *unit= "") ;
  RooRealVar(const char *name, const char *title, Double_t minValue, 
	   Double_t maxValue, const char *unit= "");
  RooRealVar(const char *name, const char *title, Double_t value, 
	   Double_t minValue, Double_t maxValue, const char *unit= "") ;
  RooRealVar(const RooRealVar& other, const char* name=0);
  virtual TObject* clone() const { return new RooRealVar(*this); }
  virtual ~RooRealVar();
  RooRealVar& operator=(const RooRealVar& other) ;
  
  // Parameter value and error accessors
  virtual Double_t getVal() const { return _value ; }
  virtual void setVal(Double_t value);
  inline Double_t getError() const { return _error; }
  inline void setError(Double_t value) { _error= value; }

  // Set/get finite fit range limits
  void setFitMin(Double_t value) ;
  void setFitMax(Double_t value) ;
  void setFitRange(Double_t min, Double_t max) ;
  virtual Double_t getFitMin() const { return _fitMin ; }
  virtual Double_t getFitMax() const { return _fitMax ; }

  // Set/get infinite fit range limits
  inline void removeFitMin() { _fitMin= -INFINITY; }
  inline void removeFitMax() { _fitMax= +INFINITY; }
  inline void removeFitRange() { _fitMin= -INFINITY; _fitMax= +INFINITY; }

  // I/O streaming interface (machine readable)
  virtual Bool_t readFromStream(istream& is, Bool_t compact, Bool_t verbose=kFALSE) ;
  virtual void writeToStream(ostream& os, Bool_t compact) const ;

  // Printing interface (human readable)
  virtual void printToStream(ostream& stream, PrintOption opt=Standard, TString indent= "") const ;
  TString* format(Int_t sigDigits, const char *options) ;

  virtual Bool_t isIntegrationSafeLValue(const RooDataSet* set) const { return kTRUE ; }

protected:

  virtual Double_t evaluate() const { return _value ; } // dummy because we overloaded getVal()

  Double_t chopAt(Double_t what, Int_t where) ;

  Double_t _fitMin ;
  Double_t _fitMax ;
  Double_t _error;

  ClassDef(RooRealVar,1) // Real-valued variable 
};

#endif
