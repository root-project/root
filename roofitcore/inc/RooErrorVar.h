/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooErrorVar.rdl,v 1.1 2001/10/11 01:28:50 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   09-Oct-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/
#ifndef ROO_ERROR_VAR
#define ROO_ERROR_VAR

#include <iostream.h>
#include <math.h>
#include <float.h>
#include "TString.h"

#include "RooFitCore/RooAbsReal.hh"
#include "RooFitCore/RooRealVar.hh"
#include "RooFitCore/RooRealProxy.hh"

class RooErrorVar : public RooAbsRealLValue {
public:
  // Constructors, assignment etc.
  inline RooErrorVar() { }
  RooErrorVar(const char *name, const char *title, const RooRealVar& input) ;
  RooErrorVar(const RooErrorVar& other, const char* name=0);
  virtual TObject* clone(const char* newname) const { return new RooErrorVar(*this,newname); }
  virtual ~RooErrorVar() {} ;

  virtual Double_t getVal(const RooArgSet* set=0) const { 
    return evaluate();
  }
  virtual Double_t evaluate() const { 
    return ((RooRealVar&)_realVar.arg()).getError() ; // dummy because we overloaded getVal()
  } 

  virtual void setVal(Double_t value) {
    ((RooRealVar&)_realVar.arg()).setVal(value) ; // dummy because we overloaded getVal()
  }

  // Set/get finite fit range limits
  void setFitMin(Double_t value) ;
  void setFitMax(Double_t value) ;
  void setFitRange(Double_t min, Double_t max) ;
  void setFitBins(Int_t nBins) { _fitBins = nBins ; }
  virtual Double_t getFitMin() const { return _fitMin ; }
  virtual Double_t getFitMax() const { return _fitMax ; }
  virtual Int_t getFitBins() const { return _fitBins ; }

  // Set infinite fit range limits
  inline void removeFitMin() { _fitMin= -RooNumber::infinity; }
  inline void removeFitMax() { _fitMax= +RooNumber::infinity; }
  inline void removeFitRange() { _fitMin= -RooNumber::infinity; _fitMax= +RooNumber::infinity; }
protected:

  void syncCache(const RooArgSet* set=0) { _value = evaluate() ; }

  RooRealProxy _realVar ; // RealVar with the original error
  Double_t _fitMin ;  // Minimum of fit range
  Double_t _fitMax ;  // Maximum of fit range
  Int_t    _fitBins ; // Number of bins in fit range for binned fits

  ClassDef(RooErrorVar,1) // Abstract modifiable real-valued variable
};

#endif
