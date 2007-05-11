/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooErrorVar.rdl,v 1.15 2005/12/08 13:19:55 wverkerke Exp $
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
#ifndef ROO_ERROR_VAR
#define ROO_ERROR_VAR

#include "Riostream.h"
#include <math.h>
#include <float.h>

#include "RooAbsReal.h"
#include "RooRealVar.h"
#include "RooRealProxy.h"

class RooErrorVar : public RooAbsRealLValue {
public:
  // Constructors, assignment etc.
  inline RooErrorVar() { }
  RooErrorVar(const char *name, const char *title, const RooRealVar& input) ;
  RooErrorVar(const RooErrorVar& other, const char* name=0);
  virtual TObject* clone(const char* newname) const { return new RooErrorVar(*this,newname); }
  virtual ~RooErrorVar() ;

  virtual Double_t getVal(const RooArgSet* set=0) const ; 

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
  inline void setMin(Double_t value) { setMin(0,value) ; }
  inline void setMax(Double_t value) { setMax(0,value) ; }
  inline void setRange(Double_t min, Double_t max) { setRange(0,min,max) ; }
  void setMin(const char* name, Double_t value) ;
  void setMax(const char* name, Double_t value) ;
  void setRange(const char* name, Double_t min, Double_t max) ;

  void setBins(Int_t nBins) { setBinning(RooUniformBinning(getMin(),getMax(),nBins)) ; }
  void setBinning(const RooAbsBinning& binning, const char* name=0) ;
  const RooAbsBinning& getBinning(const char* name=0, Bool_t verbose=kTRUE, Bool_t createOnTheFly=kFALSE) const ;
  RooAbsBinning& getBinning(const char* name=0, Bool_t verbose=kTRUE, Bool_t createOnTheFly=kFALSE) ;
  Bool_t hasBinning(const char* name) const ;

  // Set infinite fit range limits
  inline void removeMin(const char* name=0) { getBinning(name).setMin(-RooNumber::infinity) ; }
  inline void removeMax(const char* name=0) { getBinning(name).setMax(RooNumber::infinity) ; }
  inline void removeRange(const char* name=0) { getBinning(name).setRange(-RooNumber::infinity,RooNumber::infinity) ; }

  using RooAbsRealLValue::operator= ;

protected:

  RooLinkedList _altBinning ;  //! Optional alternative ranges and binnings

  void syncCache(const RooArgSet* set=0) ;

  RooRealProxy _realVar ; // RealVar with the original error
  RooAbsBinning* _binning ; //!

  ClassDef(RooErrorVar,1) // Abstract modifiable real-valued variable
};

#endif
