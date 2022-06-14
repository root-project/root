/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooLinTransBinning.h,v 1.9 2007/05/11 09:11:30 verkerke Exp $
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
#ifndef ROO_LIN_TRANS_BINNING
#define ROO_LIN_TRANS_BINNING

#include "Rtypes.h"
#include "RooAbsBinning.h"

class RooLinTransBinning : public RooAbsBinning {
public:

  RooLinTransBinning(const char* name=0) : RooAbsBinning(name) { }
  RooLinTransBinning(const RooAbsBinning& input, double slope=1.0, double offset=0.0, const char* name=0);
  RooLinTransBinning(const RooLinTransBinning&, const char* name=0);
  RooAbsBinning* clone(const char* name=0) const override { return new RooLinTransBinning(*this,name) ; }
  ~RooLinTransBinning() override ;

  Int_t numBoundaries() const override { return _input->numBoundaries() ; }
  Int_t binNumber(double x) const override { return _input->binNumber(invTrans(x)) ; }
  double binCenter(Int_t bin) const override { return trans(_input->binCenter(binTrans(bin))) ; }
  double binWidth(Int_t bin) const override { return _slope*_input->binWidth(binTrans(bin)) ; }
  double binLow(Int_t bin) const override { if (_slope>0) return trans(_input->binLow(binTrans(bin))) ; else return trans(_input->binHigh(binTrans(bin))) ; }
  double binHigh(Int_t bin) const override { if (_slope>0) return trans(_input->binHigh(binTrans(bin))) ; else return trans(_input->binLow(binTrans(bin))) ; }

  void setRange(double xlo, double xhi) override ;
  void setMin(double xlo) override { setRange(xlo,highBound()) ; }
  void setMax(double xhi) override { setRange(lowBound(),xhi) ; }

  double lowBound() const override { if (_slope>0) return trans(_input->lowBound()) ; else return trans(_input->highBound()) ; }
  double highBound() const override { if (_slope>0) return trans(_input->highBound()) ; else return trans(_input->lowBound()) ; }
  double averageBinWidth() const override { return _slope*_input->averageBinWidth() ; }

  double* array() const override ;

  void updateInput(const RooAbsBinning& input, double slope=1.0, double offset=0.0) ;

protected:

  inline Int_t binTrans(Int_t bin) const { if (_slope>0) return bin ; else return numBins()-bin-1 ; }
  inline double trans(double x) const { return x*_slope + _offset ; }
  inline double invTrans(double x) const { if (_slope==0.) return 0 ; return (x-_offset)/_slope ; }

  double _slope{0.};   ///< Slope of transformation
  double _offset{0.};  ///< Offset of transformation
  RooAbsBinning* _input{nullptr};    ///< Input binning
  mutable double *_array{nullptr}; ///<! Array of transformed bin boundaries

  ClassDefOverride(RooLinTransBinning,1) // Linear transformation of binning specification
};

#endif
