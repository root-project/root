/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
 *    File: $Id: RooVoigtian.h,v 1.7 2007/07/12 20:30:49 wouter Exp $
 * Authors:                                                                  *
 *   TS, Thomas Schietinger, SLAC,          schieti@slac.stanford.edu        *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

#ifndef ROO_VOIGTIAN
#define ROO_VOIGTIAN

#include <RooAbsPdf.h>
#include <RooRealProxy.h>

class RooVoigtian : public RooAbsPdf {
public:
  RooVoigtian() {}
  RooVoigtian(const char *name, const char *title,
         RooAbsReal& _x, RooAbsReal& _mean,
              RooAbsReal& _width, RooAbsReal& _sigma,
              bool doFast = false);
  RooVoigtian(const RooVoigtian& other, const char* name=nullptr) ;
  TObject* clone(const char* newname) const override { return new RooVoigtian(*this,newname); }

  /// Enable the fast evaluation of the complex error function using look-up
  /// tables (default is the "slow" CERNlib algorithm).
  inline void selectFastAlgorithm()    { _doFast = true;  }

  /// Disable the fast evaluation of the complex error function using look-up
  /// tables (default is the "slow" CERNlib algorithm).
  inline void selectDefaultAlgorithm() { _doFast = false; }

protected:

  RooRealProxy x ;
  RooRealProxy mean ;
  RooRealProxy width ;
  RooRealProxy sigma ;

  double evaluate() const override ;
  void computeBatch(cudaStream_t*, double* output, size_t nEvents, RooFit::Detail::DataMap const&) const override;
  inline bool canComputeBatchWithCuda() const override { return true; }

private:

  bool _doFast = false;
  ClassDefOverride(RooVoigtian,2) // Voigtian PDF (Gauss (x) BreitWigner)
};

#endif

