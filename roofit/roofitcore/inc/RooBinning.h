/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooBinning.h,v 1.9 2007/05/11 09:11:30 verkerke Exp $
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
#include "RooAbsBinning.h"
#include "RooNumber.h"
#include <vector>
class RooAbsPdf;
class RooRealVar;


class RooBinning : public RooAbsBinning {
public:

  RooBinning(Double_t xlo = -RooNumber::infinity(), Double_t xhi = RooNumber::infinity(), const char* name = 0);
  RooBinning(Int_t nBins, Double_t xlo, Double_t xhi, const char* name = 0);
  RooBinning(Int_t nBins, const Double_t* boundaries, const char* name = 0);
  RooBinning(const RooBinning& other, const char* name = 0);
  RooAbsBinning* clone(const char* name = 0) const override { return new RooBinning(*this,name?name:GetName()); }
  ~RooBinning() override;

  /// Return the number boundaries
  Int_t numBoundaries() const override {
    return _nbins+1;
  }
  Int_t binNumber(Double_t x) const override;
  Int_t rawBinNumber(Double_t x) const override;
  virtual Double_t nearestBoundary(Double_t x) const;

  void setRange(Double_t xlo, Double_t xhi) override;

  /// Return the lower bound value
  Double_t lowBound() const override {
    return _xlo;
  }

  /// Return the upper bound value
  Double_t highBound() const override {
    return _xhi;
  }

  /// Return the average bin width
  Double_t averageBinWidth() const override {
    return (highBound() - lowBound()) / numBins();
  }
  Double_t* array() const override;

  Double_t binCenter(Int_t bin) const override;
  Double_t binWidth(Int_t bin) const override;
  Double_t binLow(Int_t bin) const override;
  Double_t binHigh(Int_t bin) const override;

  bool addBoundary(Double_t boundary);
  void addBoundaryPair(Double_t boundary, Double_t mirrorPoint = 0);
  void addUniform(Int_t nBins, Double_t xlo, Double_t xhi);
  bool removeBoundary(Double_t boundary);

  bool hasBoundary(Double_t boundary);

protected:

  bool binEdges(Int_t bin, Double_t& xlo, Double_t& xhi) const;
  void updateBinCount();

  Double_t _xlo;          ///< Lower bound
  Double_t _xhi;          ///< Upper bound
  bool _ownBoundLo;     ///< Does the lower bound coincide with a bin boundary
  bool _ownBoundHi;     ///< Does the upper bound coincide with a bin boundary
  Int_t _nbins;           ///< Numer of bins

  std::vector<Double_t> _boundaries; ///< Boundaries
  mutable Double_t* _array;          ///<! Array of boundaries
  mutable Int_t _blo;                ///<! bin number for _xlo

  ClassDefOverride(RooBinning,3) // Generic binning specification
};

#endif
