/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooLinearVar.rdl,v 1.18 2005/06/20 15:44:54 wverkerke Exp $
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
#ifndef ROO_LINEAR_VAR
#define ROO_LINEAR_VAR

#include "Riostream.h"
#include <math.h>
#include <float.h>
#include "TString.h"
#include "RooAbsRealLValue.h"
#include "RooRealProxy.h"
#include "RooFormula.h"
#include "RooLinTransBinning.h"

class RooArgSet ;

class RooLinearVar : public RooAbsRealLValue {
public:
  // Constructors, assignment etc.
  RooLinearVar(const char *name, const char *title, RooAbsRealLValue& variable, const RooAbsReal& slope, const RooAbsReal& offset, const char *unit= "") ;
  RooLinearVar(const RooLinearVar& other, const char* name=0);
  virtual TObject* clone(const char* newname) const { return new RooLinearVar(*this,newname); }
  virtual ~RooLinearVar() ;
  
  // Parameter value and error accessors
  virtual void setVal(Double_t value) ;

  // Jacobian and limits
  virtual Bool_t hasBinning(const char* name) const ;
  virtual const RooAbsBinning& getBinning(const char* name=0, Bool_t verbose=kTRUE, Bool_t createOnTheFly=kFALSE) const ;
  virtual RooAbsBinning& getBinning(const char* name=0, Bool_t verbose=kTRUE, Bool_t createOnTheFly=kFALSE)  ;

  virtual Double_t jacobian() const ;
  virtual Bool_t isJacobianOK(const RooArgSet& depList) const ;

  // I/O streaming interface (machine readable)
  virtual Bool_t readFromStream(istream& is, Bool_t compact, Bool_t verbose=kFALSE) ;
  virtual void writeToStream(ostream& os, Bool_t compact) const ;

  // Printing interface (human readable)
  virtual void printToStream(ostream& stream, PrintOption opt=Standard, TString indent= "") const ;

  using RooAbsRealLValue::operator= ;

protected:

  virtual Double_t evaluate() const ;

  mutable RooLinTransBinning _binning ;
  RooLinkedList _altBinning ; //!
  RooRealProxy _var ;  
  RooRealProxy _slope ;
  RooRealProxy _offset ;

  ClassDef(RooLinearVar,1) //  Modifiable linear transformation variable
};

#endif
