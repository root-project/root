/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
 *    File: $Id$
 * Authors:                                                                  *
 *   GR, Gerhard Raven,   Nikhef & VU, Gerhard.Raven@nikhef.nl
 *                                                                           *
 * Copyright (c) 2010, Nikhef & VU. All rights reserved.
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/
#ifndef ROO_LEGENDRE
#define ROO_LEGENDRE

#include "RooAbsReal.h"
#include "RooRealProxy.h"

class RooLegendre : public RooAbsReal {
public:
  RooLegendre() ;
  // an (associated) Legendre polynomial, P_l^m(x)
  // note: P_l(x) == P_l^0(x)
  RooLegendre(const char *name, const char *title, RooAbsReal& ctheta, int l, int m=0);
  // product of two associated Legendre polynomials, P_l1^m1(ctheta) * P_l2^m2(ctheta)
  RooLegendre(const char *name, const char *title, RooAbsReal& ctheta, int l1, int m1, int l2, int m2);

  RooLegendre(const RooLegendre& other, const char* name = 0);
  TObject* clone(const char* newname) const override { return new RooLegendre(*this, newname); }
  inline ~RooLegendre() override { }

  Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* rangeName=0) const override ;
  Double_t analyticalIntegral(Int_t code, const char* rangeName=0) const override ;

  Int_t getMaxVal( const RooArgSet& vars) const override;
  Double_t maxVal( Int_t code) const override;

protected: // allow RooSpHarmonic access...
  RooRealProxy _ctheta;
  int _l1,_m1;
  int _l2,_m2;

  Double_t evaluate() const override;
  RooSpan<double> evaluateSpan(RooBatchCompute::RunContext& evalData, const RooArgSet* normSet) const override;

  ClassDefOverride(RooLegendre,1) // Legendre polynomial
};

#endif
