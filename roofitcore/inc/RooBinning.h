/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooBinning.rdl,v 1.8 2005/02/25 14:22:54 wverkerke Exp $
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
#ifndef ROO_BINNING
#define ROO_BINNING

#include "Rtypes.h"
#include "TList.h"
#include "RooDouble.h"
#include "RooAbsBinning.h"
#include "RooNumber.h"
class RooAbsPdf ;
class RooRealVar ;


class RooBinning : public RooAbsBinning {
public:

  RooBinning(Double_t xlo=-RooNumber::infinity, Double_t xhi=RooNumber::infinity, const char* name=0) ;
  RooBinning(Int_t nBins, Double_t xlo, Double_t xhi, const char* name=0) ;
  RooBinning(Int_t nBins, const Double_t* boundaries, const char* name=0) ;
  RooBinning(const RooBinning& other, const char* name=0) ;
  RooAbsBinning* clone(const char* name=0) const { return new RooBinning(*this,name?name:GetName()) ; }
  ~RooBinning() ;

  virtual Int_t numBoundaries() const { return _nbins+1 ; }
  virtual Int_t binNumber(Double_t x) const ;
  virtual Int_t rawBinNumber(Double_t x) const ;
  virtual Double_t nearestBoundary(Double_t x) const ;

  virtual void setRange(Double_t xlo, Double_t xhi) ;

  virtual Double_t lowBound() const { return _xlo ; }
  virtual Double_t highBound() const { return _xhi ; }
  virtual Double_t averageBinWidth() const { return (highBound()-lowBound())/numBins() ; }
  virtual Double_t* array() const ;

  virtual Double_t binCenter(Int_t bin) const ;
  virtual Double_t binWidth(Int_t bin) const ;
  virtual Double_t binLow(Int_t bin) const ;
  virtual Double_t binHigh(Int_t bin) const  ;
  
  Bool_t addBoundary(Double_t boundary) ;
  void addBoundaryPair(Double_t boundary, Double_t mirrorPoint=0) ;
  void addUniform(Int_t nBins, Double_t xlo, Double_t xhi) ;
  Bool_t removeBoundary(Double_t boundary) ;

  TIterator* binIterator() const ;
  Bool_t hasBoundary(Double_t boundary) ;
  
protected:

  Bool_t binEdges(Int_t bin, Double_t& xlo, Double_t& xhi) const ;
  void updateBinCount() ;

  Double_t _xlo ;
  Double_t _xhi ;
  Bool_t _ownBoundLo ;
  Bool_t _ownBoundHi ;
  Int_t _nbins ;

  TList _boundaries ;
  TIterator* _bIter ;        //! do not persist
  mutable Double_t* _array ; //! do not persist

  ClassDef(RooBinning,1) // Container class for binning specification
};

#endif
