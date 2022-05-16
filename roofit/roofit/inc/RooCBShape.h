/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
 *    File: $Id: RooCBShape.h,v 1.11 2007/07/12 20:30:49 wouter Exp $
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
#ifndef ROO_CB_SHAPE
#define ROO_CB_SHAPE

#include "RooAbsPdf.h"
#include "RooRealProxy.h"

class RooRealVar;

class RooCBShape : public RooAbsPdf {
public:
  RooCBShape() {} ;
  RooCBShape(const char *name, const char *title, RooAbsReal& _m,
        RooAbsReal& _m0, RooAbsReal& _sigma,
        RooAbsReal& _alpha, RooAbsReal& _n);

  RooCBShape(const RooCBShape& other, const char* name = 0);
  TObject* clone(const char* newname) const override { return new RooCBShape(*this,newname); }

  inline ~RooCBShape() override { }

  Int_t getAnalyticalIntegral( RooArgSet& allVars,  RooArgSet& analVars, const char* rangeName=0 ) const override;
  double analyticalIntegral( Int_t code, const char* rangeName=0 ) const override;

  // Optimized accept/reject generator support
  Int_t getMaxVal(const RooArgSet& vars) const override ;
  double maxVal(Int_t code) const override ;

protected:

  double ApproxErf(double arg) const ;

  RooRealProxy m;
  RooRealProxy m0;
  RooRealProxy sigma;
  RooRealProxy alpha;
  RooRealProxy n;

  double evaluate() const override;
  void computeBatch(cudaStream_t*, double* output, size_t nEvents, RooFit::Detail::DataMap const&) const override;
  inline bool canComputeBatchWithCuda() const override { return true; }


private:

  ClassDefOverride(RooCBShape,1) // Crystal Ball lineshape PDF
};

#endif
