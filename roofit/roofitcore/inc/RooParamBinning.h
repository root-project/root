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

class RooParamBinning : public RooAbsBinning {
public:

  RooParamBinning(const char* name=nullptr) ;
  RooParamBinning(RooAbsReal& xlo, RooAbsReal& xhi, Int_t nBins, const char* name=nullptr) ;
  RooParamBinning(const RooParamBinning& other, const char* name=nullptr) ;
  RooAbsBinning* clone(const char* name=nullptr) const override { return new RooParamBinning(*this,name?name:GetName()) ; }
  ~RooParamBinning() override ;

  void setRange(double xlo, double xhi) override ;

  Int_t numBoundaries() const override { return _nbins + 1 ; }
  Int_t binNumber(double x) const override  ;

  double lowBound() const override { return xlo()->getVal() ; }
  double highBound() const override { return xhi()->getVal() ; }

  double binCenter(Int_t bin) const override ;
  double binWidth(Int_t bin) const override ;
  double binLow(Int_t bin) const override ;
  double binHigh(Int_t bin) const override ;

  double averageBinWidth() const override { return _binw ; }
  double* array() const override ;

  void printMultiline(std::ostream &os, Int_t content, bool verbose=false, TString indent="") const override ;

  void insertHook(RooAbsRealLValue&) const override  ;
  void removeHook(RooAbsRealLValue&) const override  ;

  bool isShareable() const override { return false ; } // parameterized binning cannot be shared across instances
  bool isParameterized() const override { return true ; } // binning is parameterized, range will need special handling in integration
  RooAbsReal* lowBoundFunc() const override { return xlo() ; }
  RooAbsReal* highBoundFunc() const override { return xhi() ; }

protected:

  mutable double* _array ; //! do not persist
  mutable RooAbsReal* _xlo ; //!
  mutable RooAbsReal* _xhi ; //!
  Int_t    _nbins ;
  double _binw ;
  mutable RooListProxy* _lp ; //
  mutable RooAbsArg* _owner ; //

  RooAbsReal* xlo() const { return _lp ? ((RooAbsReal*)_lp->at(0)) : _xlo ; }
  RooAbsReal* xhi() const { return _lp ? ((RooAbsReal*)_lp->at(1)) : _xhi ; }

  ClassDefOverride(RooParamBinning,3) // Binning specification with ranges parameterized by external RooAbsReal functions
};

#endif
