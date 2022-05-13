/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
 *    File: $Id: RooLandau.h,v 1.5 2007/07/12 20:30:49 wouter Exp $
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
#ifndef ROO_LANDAU
#define ROO_LANDAU

#include "RooAbsPdf.h"
#include "RooRealProxy.h"

class RooRealVar;

class RooLandau : public RooAbsPdf {
public:
  RooLandau() {} ;
  RooLandau(const char *name, const char *title, RooAbsReal& _x, RooAbsReal& _mean, RooAbsReal& _sigma);
  RooLandau(const RooLandau& other, const char* name=0);
  TObject* clone(const char* newname) const override { return new RooLandau(*this,newname); }
  inline ~RooLandau() override { }

  Int_t getGenerator(const RooArgSet& directVars, RooArgSet &generateVars, bool staticInitOK=true) const override;
  void generateEvent(Int_t code) override;

protected:

  RooRealProxy x ;
  RooRealProxy mean ;
  RooRealProxy sigma ;

  double evaluate() const override ;
  void computeBatch(cudaStream_t*, double* output, size_t nEvents, RooBatchCompute::DataMap&) const override;
  inline bool canComputeBatchWithCuda() const override { return true; }

private:

  ClassDefOverride(RooLandau,1) // Landau Distribution PDF
};

#endif
