/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooRealVar.rdl,v 1.2 2001/03/17 03:47:39 verkerke Exp $
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
class RooBlindBase ;

class RooRealVar : public RooAbsReal {
public:
  // Constructors, assignment etc.
  inline RooRealVar() { }
  RooRealVar(const char *name, const char *title,
  	   Double_t value, const char *unit= "", RooBlindBase* blinder=0) ;
  RooRealVar(const char *name, const char *title, Double_t minValue, 
	   Double_t maxValue, const char *unit= "", RooBlindBase* blinder=0);
  RooRealVar(const char *name, const char *title, Double_t value, 
	   Double_t minValue, Double_t maxValue, const char *unit= "", 
	   RooBlindBase* blinder=0);
  RooRealVar(const RooRealVar& other);
  virtual ~RooRealVar();
  virtual RooAbsArg& operator=(RooAbsArg& other) ;
  
  // Parameter value and error accessors
  virtual operator Double_t&();
  virtual operator Double_t() ;
  inline Double_t getError() const { return _error; }
  virtual Double_t getVal() { return _value ; } // overrides RooAbsReal::GetVar()
  virtual void setVal(Double_t value);
  inline void setError(Double_t value) { _error= value; }
  virtual Double_t operator=(Double_t newValue);

  // Integration limits
  inline Double_t getIntegMin() const { return _integMin ; }
  inline Double_t getIntegMax() const { return _integMax ; }
  void setIntegMin(Double_t value) ;
  void setIntegMax(Double_t value) ;
  void setIntegRange(Double_t min, Double_t max) ;
  Bool_t inIntegRange(Double_t value, Double_t* clippedValue=0) const;
  Bool_t hasIntegLimits() const { return kTRUE ; }

  // Constant and Projected flags 
  inline Bool_t isConstant() const { return getAttribute("Constant") ; }
  inline void setConstant(Bool_t value= kTRUE) { setAttribute("Constant",value); }
  inline Bool_t isProjected() const { return getAttribute("Projected") ; }
  inline void setProjected(Bool_t value= kTRUE) { setAttribute("Projected",value);}

  // I/O streaming interface (machine readable)
  virtual Bool_t readFromStream(istream& is, Bool_t compact, Bool_t verbose=kFALSE) ;
  virtual void writeToStream(ostream& os, Bool_t compact) ;

  // Printing interface (human readable)
  virtual void printToStream(ostream& stream, PrintOption opt=Standard) ;
  TString* format(Int_t sigDigits, const char *options) ;

protected:

  void updateIntegLimits() ;
  virtual void attachToTree(TTree& t, Int_t bufSize=32000) ;
  Double_t chopAt(Double_t what, Int_t where) ;

  virtual Bool_t isValid() ;
  virtual Bool_t isValid(Double_t value) ;

  Double_t _integMin ;
  Double_t _integMax ;
  Double_t _error;
  RooBlindBase* _blinder ; //! unowned ptr

  ClassDef(RooRealVar,1) // a real-valued variable and its value
};

#endif
