/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooUniformBinning.h,v 1.10 2007/05/11 09:11:30 verkerke Exp $
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
#ifndef ROO_UNIFORM_BINNING
#define ROO_UNIFORM_BINNING

#include "Rtypes.h"
#include "RooAbsBinning.h"

class RooUniformBinning : public RooAbsBinning {
public:

  RooUniformBinning(const char* name=0) ;
  RooUniformBinning(double xlo, double xhi, Int_t nBins, const char* name=0) ;
  RooUniformBinning(const RooUniformBinning& other, const char* name=0) ;
  RooAbsBinning* clone(const char* name=0) const override { return new RooUniformBinning(*this,name?name:GetName()) ; }
  ~RooUniformBinning() override ;

  void setRange(double xlo, double xhi) override ;

  Int_t numBoundaries() const override { return _nbins + 1 ; }
  Int_t binNumber(double x) const override  ;
  bool isUniform() const override { return true ; }

  double lowBound() const override { return _xlo ; }
  double highBound() const override { return _xhi ; }

  double binCenter(Int_t bin) const override ;
  double binWidth(Int_t bin) const override ;
  double binLow(Int_t bin) const override ;
  double binHigh(Int_t bin) const override ;

  double averageBinWidth() const override { return _binw ; }
  double* array() const override ;

protected:

  mutable double* _array ; ///<! do not persist
  double _xlo ;
  double _xhi ;
  Int_t    _nbins ;
  double _binw ;


  ClassDefOverride(RooUniformBinning,1) // Uniform binning specification
};

#endif
