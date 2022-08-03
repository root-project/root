/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
 *    File: $Id$
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
#ifndef ROO_UNIFORM
#define ROO_UNIFORM

#include "RooAbsPdf.h"
#include "RooListProxy.h"

class RooRealVar;

class RooUniform : public RooAbsPdf {
public:
  RooUniform() {} ;
  RooUniform(const char *name, const char *title, const RooArgSet& _x);
  RooUniform(const RooUniform& other, const char* name=nullptr) ;
  TObject* clone(const char* newname) const override { return new RooUniform(*this,newname); }
  inline ~RooUniform() override { }

  Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* rangeName=nullptr) const override ;
  double analyticalIntegral(Int_t code, const char* rangeName=nullptr) const override ;

  Int_t getGenerator(const RooArgSet& directVars, RooArgSet &generateVars, bool staticInitOK=true) const override;
  void generateEvent(Int_t code) override;

protected:

  RooListProxy x ;

  double evaluate() const override ;
  RooSpan<double> evaluateSpan(RooBatchCompute::RunContext& evalData, const RooArgSet* /*normSet*/ = nullptr) const override;


private:

  ClassDefOverride(RooUniform,1) // Flat PDF in N dimensions
};

#endif
