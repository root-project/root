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

  RooBinning(double xlo = -RooNumber::infinity(), double xhi = RooNumber::infinity(), const char* name = 0);
  RooBinning(Int_t nBins, double xlo, double xhi, const char* name = 0);
  RooBinning(Int_t nBins, const double* boundaries, const char* name = 0);
  RooBinning(const RooBinning& other, const char* name = 0);
  RooAbsBinning* clone(const char* name = 0) const override { return new RooBinning(*this,name?name:GetName()); }
  ~RooBinning() override;

  /// Return the number boundaries
  Int_t numBoundaries() const override {
    return _nbins+1;
  }
  Int_t binNumber(double x) const override;
  Int_t rawBinNumber(double x) const;
  virtual double nearestBoundary(double x) const;

  void setRange(double xlo, double xhi) override;

  /// Return the lower bound value
  double lowBound() const override {
    return _xlo;
  }

  /// Return the upper bound value
  double highBound() const override {
    return _xhi;
  }

  /// Return the average bin width
  double averageBinWidth() const override {
    return (highBound() - lowBound()) / numBins();
  }
  double* array() const override;

  double binCenter(Int_t bin) const override;
  double binWidth(Int_t bin) const override;
  double binLow(Int_t bin) const override;
  double binHigh(Int_t bin) const override;

  bool addBoundary(double boundary);
  void addBoundaryPair(double boundary, double mirrorPoint = 0);
  void addUniform(Int_t nBins, double xlo, double xhi);
  bool removeBoundary(double boundary);

protected:

  bool binEdges(Int_t bin, double& xlo, double& xhi) const;
  void updateBinCount();

  double _xlo;          ///< Lower bound
  double _xhi;          ///< Upper bound
  bool _ownBoundLo;     ///< Does the lower bound coincide with a bin boundary
  bool _ownBoundHi;     ///< Does the upper bound coincide with a bin boundary
  Int_t _nbins;           ///< Numer of bins

  std::vector<double> _boundaries; ///< Boundaries
  mutable double* _array;          ///<! Array of boundaries
  mutable Int_t _blo;                ///<! bin number for _xlo

  ClassDefOverride(RooBinning,3) // Generic binning specification
};

#endif
