/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooLinearVar.rdl,v 1.11 2002/09/05 04:33:37 verkerke Exp $
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2004, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
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
#include "RooFitCore/RooLinTransBinning.hh"

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
//   virtual Double_t getFitMin() const ;
//   virtual Double_t getFitMax() const ;
//   virtual Double_t fitBinCenter(Int_t i) const ;
//   virtual Double_t fitBinLow(Int_t i) const ;
//   virtual Double_t fitBinHigh(Int_t i) const ;
//   virtual Double_t fitBinWidth(Int_t i) const ;
//   virtual Int_t getFitBins() const ;
  virtual const RooAbsBinning& getBinning() const ;

  virtual Double_t jacobian() const ;
  virtual Bool_t isJacobianOK(const RooArgSet& depList) const ;

  // I/O streaming interface (machine readable)
  virtual Bool_t readFromStream(istream& is, Bool_t compact, Bool_t verbose=kFALSE) ;
  virtual void writeToStream(ostream& os, Bool_t compact) const ;

  // Printing interface (human readable)
  virtual void printToStream(ostream& stream, PrintOption opt=Standard, TString indent= "") const ;

protected:

  virtual Double_t evaluate() const ;

  mutable RooLinTransBinning _binning ;
  RooRealProxy _var ;  
  RooRealProxy _slope ;
  RooRealProxy _offset ;

  ClassDef(RooLinearVar,1) //  Modifiable linear transformation variable
};

#endif
