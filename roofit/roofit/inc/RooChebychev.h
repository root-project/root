/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
 *    File: $Id: RooChebychev.h,v 1.6 2007/05/11 09:13:07 verkerke Exp $
 * Authors:                                                                  *
 *   GR, Gerhard Raven,   UC San Diego, Gerhard.Raven@slac.stanford.edu
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/
#ifndef ROO_CHEBYCHEV
#define ROO_CHEBYCHEV

#include "RooAbsPdf.h"
#include "RooRealProxy.h"
#include "RooListProxy.h"

class RooRealVar;
class RooArgList ;

class RooChebychev : public RooAbsPdf {
public:

  RooChebychev() ;
  RooChebychev(const char *name, const char *title,
               RooAbsReal& _x, const RooArgList& _coefList) ;

  RooChebychev(const RooChebychev& other, const char* name = 0);
  TObject* clone(const char* newname) const override { return new RooChebychev(*this, newname); }
  inline ~RooChebychev() override { }

  Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* rangeName=0) const override ;
  double analyticalIntegral(Int_t code, const char* rangeName=0) const override ;

  void selectNormalizationRange(const char* rangeName=0, bool force=false) override ;

private:

  RooRealProxy _x;
  RooListProxy _coefList ;
  mutable TNamed* _refRangeName ;

  double evaluate() const override;
  void computeBatch(cudaStream_t*, double* output, size_t nEvents, RooBatchCompute::DataMap&) const override;
  inline bool canComputeBatchWithCuda() const override { return true; }

  double evalAnaInt(const double a, const double b) const;

  ClassDefOverride(RooChebychev,2) // Chebychev polynomial PDF
};

#endif
