/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooLinearVar.rdl,v 1.4 2001/08/02 21:39:10 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/
#ifndef ROO_LINEAR_VAR
#define ROO_LINEAR_VAR

#include <iostream.h>
#include <math.h>
#include <float.h>
#include "TString.h"
#include "RooFitCore/RooAbsRealLValue.hh"
#include "RooFitCore/RooRealProxy.hh"
#include "RooFitCore/RooFormula.hh"

class RooArgSet ;

class RooLinearVar : public RooAbsRealLValue {
public:
  // Constructors, assignment etc.
  inline RooLinearVar() { }
  RooLinearVar(const char *name, const char *title, RooRealVar& variable, RooAbsReal& slope, RooAbsReal& offset, const char *unit= "") ;
  RooLinearVar(const RooLinearVar& other, const char* name=0);
  virtual TObject* clone(const char* newname) const { return new RooLinearVar(*this,newname); }
  virtual ~RooLinearVar() ;
  
  // Parameter value and error accessors
  virtual void setVal(Double_t value) ;

  // Jacobian and limits
  virtual Double_t getFitMin() const ;
  virtual Double_t getFitMax() const ;
  virtual Double_t jacobian() const ;
  virtual Bool_t isJacobianOK(const RooArgSet& depList) const ;

  // I/O streaming interface (machine readable)
  virtual Bool_t readFromStream(istream& is, Bool_t compact, Bool_t verbose=kFALSE) ;
  virtual void writeToStream(ostream& os, Bool_t compact) const ;

  // Printing interface (human readable)
  virtual void printToStream(ostream& stream, PrintOption opt=Standard, TString indent= "") const ;

protected:

  virtual Double_t evaluate() const ;
  
  RooRealProxy _var ;  
  RooRealProxy _slope ;
  RooRealProxy _offset ;

  ClassDef(RooLinearVar,1) //  Modifiable linear transformation variable
};

#endif
