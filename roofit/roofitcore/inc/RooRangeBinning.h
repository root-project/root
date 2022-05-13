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
  RooRangeBinning(double xmin, double xmax, const char* name=0) ;
  RooRangeBinning(const RooRangeBinning&, const char* name=0) ;
  RooAbsBinning* clone(const char* name=0) const override { return new RooRangeBinning(*this,name?name:GetName()) ; }
  ~RooRangeBinning() override ;

  Int_t numBoundaries() const override { return 2 ; }
  Int_t binNumber(double) const override { return 0 ; }
  double binCenter(Int_t) const override { return (_range[0] + _range[1]) / 2 ; }
  double binWidth(Int_t) const override { return (_range[1] - _range[0]) ; }
  double binLow(Int_t) const override { return _range[0] ; }
  double binHigh(Int_t) const override { return _range[1] ; }

  void setRange(double xlo, double xhi) override ;
  void setMin(double xlo) override { setRange(xlo,highBound()) ; }
  void setMax(double xhi) override { setRange(lowBound(),xhi) ; }

  double lowBound() const override { return _range[0] ; }
  double highBound() const override { return _range[1] ; }
  double averageBinWidth() const override { return binWidth(0) ; }

  double* array() const override { return const_cast<double*>(_range) ; }

protected:

  double _range[2] ;

  ClassDefOverride(RooRangeBinning,1) // Binning that only defines the total range
};

#endif
