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
#ifndef ROO_SPHARMONIC
#define ROO_SPHARMONIC

#include "RooLegendre.h"
#include "RooRealProxy.h"

class RooSpHarmonic : public RooLegendre {
public:
  RooSpHarmonic() ;
  RooSpHarmonic(const char *name, const char *title, RooAbsReal& ctheta, RooAbsReal& phi, int l, int m);
  RooSpHarmonic(const char *name, const char *title, RooAbsReal& ctheta, RooAbsReal& phi, int l1, int m1, int l2, int m2);

  RooSpHarmonic(const RooSpHarmonic& other, const char* name = 0);
  TObject* clone(const char* newname) const override { return new RooSpHarmonic(*this, newname); }
  inline ~RooSpHarmonic() override { }

  Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* rangeName=0) const override ;
  Double_t analyticalIntegral(Int_t code, const char* rangeName=0) const override ;

  Int_t getMaxVal( const RooArgSet& vars) const override;
  Double_t maxVal( Int_t code) const override;

private:
  RooRealProxy _phi;
  double _n;
  int _sgn1,_sgn2;

  Double_t evaluate() const override;

  ClassDefOverride(RooSpHarmonic,1) // SpHarmonic polynomial
};

#endif
