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
  RooAbsBinning* clone(const char* name=0) const { return new RooUniformBinning(*this,name?name:GetName()) ; }
  virtual ~RooUniformBinning() ;

  virtual void setRange(Double_t xlo, Double_t xhi) ;

  virtual Int_t numBoundaries() const { return _nbins + 1 ; }
  virtual Int_t binNumber(Double_t x) const  ;
  virtual Bool_t isUniform() const { return kTRUE ; }

  virtual Double_t lowBound() const { return _xlo ; }
  virtual Double_t highBound() const { return _xhi ; }

  virtual Double_t binCenter(Int_t bin) const ;
  virtual Double_t binWidth(Int_t bin) const ;
  virtual Double_t binLow(Int_t bin) const ;
  virtual Double_t binHigh(Int_t bin) const ;

  virtual Double_t averageBinWidth() const { return _binw ; }
  virtual Double_t* array() const ;

protected:

  mutable Double_t* _array ; //! do not persist
  Double_t _xlo ;
  Double_t _xhi ;
  Int_t    _nbins ;
  Double_t _binw ;


  ClassDef(RooUniformBinning,1) // Uniform binning specification
};

#endif
