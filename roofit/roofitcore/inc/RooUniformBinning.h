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
class TIterator ;

class RooUniformBinning : public RooAbsBinning {
public:

  RooUniformBinning(const char* name=0) ;
  RooUniformBinning(Double_t xlo, Double_t xhi, Int_t nBins, const char* name=0) ;
  RooUniformBinning(const RooUniformBinning& other, const char* name=0) ;
  RooAbsBinning* clone(const char* name=0) const override { return new RooUniformBinning(*this,name?name:GetName()) ; }
  ~RooUniformBinning() override ;

  void setRange(Double_t xlo, Double_t xhi) override ;

  Int_t numBoundaries() const override { return _nbins + 1 ; }
  Int_t binNumber(Double_t x) const override  ;
  Bool_t isUniform() const override { return kTRUE ; }

  Double_t lowBound() const override { return _xlo ; }
  Double_t highBound() const override { return _xhi ; }

  Double_t binCenter(Int_t bin) const override ;
  Double_t binWidth(Int_t bin) const override ;
  Double_t binLow(Int_t bin) const override ;
  Double_t binHigh(Int_t bin) const override ;

  Double_t averageBinWidth() const override { return _binw ; }
  Double_t* array() const override ;

protected:

  mutable Double_t* _array ; ///<! do not persist
  Double_t _xlo ;
  Double_t _xhi ;
  Int_t    _nbins ;
  Double_t _binw ;


  ClassDefOverride(RooUniformBinning,1) // Uniform binning specification
};

#endif
