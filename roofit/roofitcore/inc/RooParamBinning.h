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
#ifndef ROO_PARAM_BINNING
#define ROO_PARAM_BINNING

#include "Rtypes.h"
#include "RooAbsBinning.h"
#include "RooRealVar.h"
#include "RooListProxy.h"
class TIterator ;

class RooParamBinning : public RooAbsBinning {
public:

  RooParamBinning(const char* name=0) ;
  RooParamBinning(RooAbsReal& xlo, RooAbsReal& xhi, Int_t nBins, const char* name=0) ;
  RooParamBinning(const RooParamBinning& other, const char* name=0) ;
  RooAbsBinning* clone(const char* name=0) const { return new RooParamBinning(*this,name?name:GetName()) ; }
  virtual ~RooParamBinning() ;

  virtual void setRange(Double_t xlo, Double_t xhi) ;

  virtual Int_t numBoundaries() const { return _nbins + 1 ; }
  virtual Int_t binNumber(Double_t x) const  ;

  virtual Double_t lowBound() const { return xlo()->getVal() ; }
  virtual Double_t highBound() const { return xhi()->getVal() ; }

  virtual Double_t binCenter(Int_t bin) const ;
  virtual Double_t binWidth(Int_t bin) const ;
  virtual Double_t binLow(Int_t bin) const ;
  virtual Double_t binHigh(Int_t bin) const ;

  virtual Double_t averageBinWidth() const { return _binw ; }
  virtual Double_t* array() const ;

  void printMultiline(ostream &os, Int_t content, Bool_t verbose=kFALSE, TString indent="") const ;

  virtual void insertHook(RooAbsRealLValue&) const  ;
  virtual void removeHook(RooAbsRealLValue&) const  ;

  virtual Bool_t isShareable() const { return kFALSE ; } // parameterized binning cannot be shared across instances
  virtual Bool_t isParameterized() const { return kTRUE ; } // binning is parameterized, range will need special handling in integration
  virtual RooAbsReal* lowBoundFunc() const { return xlo() ; }
  virtual RooAbsReal* highBoundFunc() const { return xhi() ; }

protected:

  mutable Double_t* _array ; //! do not persist
  mutable RooAbsReal* _xlo ; //!
  mutable RooAbsReal* _xhi ; //!
  Int_t    _nbins ;
  Double_t _binw ;
  mutable RooListProxy* _lp ; //
  mutable RooAbsArg* _owner ; //

  RooAbsReal* xlo() const { return _lp ? ((RooAbsReal*)_lp->at(0)) : _xlo ; }
  RooAbsReal* xhi() const { return _lp ? ((RooAbsReal*)_lp->at(1)) : _xhi ; }

  ClassDef(RooParamBinning,2) // Binning specification with ranges parameterized by external RooAbsReal functions
};

#endif
