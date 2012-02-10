/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id$
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
#ifndef ROO_HIST_FUNC
#define ROO_HIST_FUNC

#include "RooAbsReal.h"
#include "RooRealProxy.h"
#include "RooSetProxy.h"
#include "RooAICRegistry.h"

class RooRealVar;
class RooAbsReal;
class RooDataHist ;

class RooHistFunc : public RooAbsReal {
public:
  RooHistFunc() ; 
  RooHistFunc(const char *name, const char *title, const RooArgSet& vars, const RooDataHist& dhist, Int_t intOrder=0);
  RooHistFunc(const RooHistFunc& other, const char* name=0);
  virtual TObject* clone(const char* newname) const { return new RooHistFunc(*this,newname); }
  inline virtual ~RooHistFunc() { }

  RooDataHist& dataHist()  { 
    // Return RooDataHist that is represented
    return *_dataHist ; 
  }
  const RooDataHist& dataHist() const { 
    // Return RooDataHist that is represented
    return *_dataHist ; 
  }
  
  void setInterpolationOrder(Int_t order) { 
    // Set histogram interpolation order 
    _intOrder = order ; 
  }

  Int_t getInterpolationOrder() const { 
    // Return histogram interpolation order
    return _intOrder ; 
  }

  Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* rangeName=0) const ;
  Double_t analyticalIntegral(Int_t code, const char* rangeName=0) const ;

  void setCdfBoundaries(Bool_t flag) { 
    // Set use of special boundary conditions for c.d.f.s
    _cdfBoundaries = flag ; 
  }

  Bool_t getCdfBoundaries() const { 
    // If true, special boundary conditions for c.d.f.s are used
    return _cdfBoundaries ; 
  }

  virtual Int_t getMaxVal(const RooArgSet& vars) const;
  virtual Double_t maxVal(Int_t code) const;

  virtual std::list<Double_t>* binBoundaries(RooAbsRealLValue& /*obs*/, Double_t /*xlo*/, Double_t /*xhi*/) const ;
  virtual std::list<Double_t>* plotSamplingHint(RooAbsRealLValue& obs, Double_t xlo, Double_t xhi) const ; 
  virtual Bool_t isBinnedDistribution(const RooArgSet&) const { return _intOrder==0 ; }

protected:

  Double_t evaluate() const;
  Double_t totalVolume() const ;
  friend class RooAbsCachedReal ;
  Double_t totVolume() const ;

  RooSetProxy       _depList ;   // List of dependents defining dimensions of histogram
  RooDataHist*      _dataHist ;  // Unowned pointer to underlying histogram
  mutable RooAICRegistry _codeReg ; //! Auxiliary class keeping tracking of analytical integration code
  Int_t             _intOrder ; // Interpolation order
  Bool_t            _cdfBoundaries ; // Use boundary conditions for CDFs.
  mutable Double_t  _totVolume ; //! Total volume of space (product of ranges of observables)
  Bool_t            _unitNorm  ; //! Assume contents is unit normalized (for use as pdf cache)

  ClassDef(RooHistFunc,1) // Histogram based function
};

#endif
