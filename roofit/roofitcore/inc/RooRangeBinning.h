/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooRangeBinning.h,v 1.4 2007/05/11 09:11:30 verkerke Exp $
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
#ifndef ROO_RANGE_BINNING
#define ROO_RANGE_BINNING

#include "RooAbsBinning.h"

class RooRangeBinning : public RooAbsBinning {
public:

  RooRangeBinning(const char* name=0) ;
  RooRangeBinning(Double_t xmin, Double_t xmax, const char* name=0) ;
  RooRangeBinning(const RooRangeBinning&, const char* name=0) ;
  RooAbsBinning* clone(const char* name=0) const override { return new RooRangeBinning(*this,name?name:GetName()) ; }
  ~RooRangeBinning() override ;

  Int_t numBoundaries() const override { return 2 ; }
  Int_t binNumber(Double_t) const override { return 0 ; }
  Double_t binCenter(Int_t) const override { return (_range[0] + _range[1]) / 2 ; }
  Double_t binWidth(Int_t) const override { return (_range[1] - _range[0]) ; }
  Double_t binLow(Int_t) const override { return _range[0] ; }
  Double_t binHigh(Int_t) const override { return _range[1] ; }

  void setRange(Double_t xlo, Double_t xhi) override ;
  void setMin(Double_t xlo) override { setRange(xlo,highBound()) ; }
  void setMax(Double_t xhi) override { setRange(lowBound(),xhi) ; }

  Double_t lowBound() const override { return _range[0] ; }
  Double_t highBound() const override { return _range[1] ; }
  Double_t averageBinWidth() const override { return binWidth(0) ; }

  Double_t* array() const override { return const_cast<Double_t*>(_range) ; }

protected:

  Double_t _range[2] ;

  ClassDefOverride(RooRangeBinning,1) // Binning that only defines the total range
};

#endif
