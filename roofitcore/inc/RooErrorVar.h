/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooErrorVar.rdl,v 1.3 2002/03/07 06:22:21 verkerke Exp $
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
  virtual ~RooErrorVar() ;

  virtual Double_t getVal(const RooArgSet* set=0) const { 
    return evaluate();
  }
  virtual Double_t evaluate() const { 
    return ((RooRealVar&)_realVar.arg()).getError() ; // dummy because we overloaded getVal()
  } 

  virtual void setVal(Double_t value) {
    ((RooRealVar&)_realVar.arg()).setVal(value) ; // dummy because we overloaded getVal()
  }

  // We implement a fundamental type of AbsArg that can be stored in a dataset
  inline virtual Bool_t isFundamental() const { return kTRUE ; }
//   virtual Bool_t isDerived() const { return kFALSE ; }

  // I/O streaming interface (machine readable)
  virtual Bool_t readFromStream(istream& is, Bool_t compact, Bool_t verbose=kFALSE) ;
  virtual void writeToStream(ostream& os, Bool_t compact) const ;

  // Set/get finite fit range limits
  void setFitMin(Double_t value) ;
  void setFitMax(Double_t value) ;
  void setFitRange(Double_t min, Double_t max) ;
  void setFitBins(Int_t nBins) { setBinning(RooUniformBinning(getFitMin(),getFitMax(),nBins)) ; }
  void setBinning(const RooAbsBinning& binning) ;
  const RooAbsBinning& getBinning() const { return *_binning ; }

  // Set infinite fit range limits
  inline void removeFitMin() { _binning->setMin(-RooNumber::infinity) ; }
  inline void removeFitMax() { _binning->setMax(RooNumber::infinity) ; }
  inline void removeFitRange() { _binning->setRange(-RooNumber::infinity,RooNumber::infinity) ; }

protected:

  void syncCache(const RooArgSet* set=0) { _value = evaluate() ; }

  RooRealProxy _realVar ; // RealVar with the original error
  RooAbsBinning* _binning ; //!

  ClassDef(RooErrorVar,1) // Abstract modifiable real-valued variable
};

#endif
