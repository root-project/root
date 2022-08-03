/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
 *    File: $Id: RooPolynomial.h,v 1.8 2007/05/11 09:13:07 verkerke Exp $
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
#ifndef ROO_POLYNOMIAL
#define ROO_POLYNOMIAL

#include <vector>

#include "RooAbsPdf.h"
#include "RooRealProxy.h"
#include "RooListProxy.h"

class RooRealVar;
class RooArgList ;

class RooPolynomial : public RooAbsPdf {
public:

  RooPolynomial() ;
  RooPolynomial(const char* name, const char* title, RooAbsReal& x) ;
  RooPolynomial(const char *name, const char *title,
      RooAbsReal& _x, const RooArgList& _coefList, Int_t lowestOrder=1) ;

  RooPolynomial(const RooPolynomial& other, const char* name = 0);
  TObject* clone(const char* newname) const override { return new RooPolynomial(*this, newname); }
  ~RooPolynomial() override ;

  Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* rangeName=nullptr) const override ;
  double analyticalIntegral(Int_t code, const char* rangeName=nullptr) const override ;

protected:

  RooRealProxy _x;
  RooListProxy _coefList ;
  Int_t _lowestOrder ;

  mutable std::vector<double> _wksp; //! do not persist

  /// Evaluation
  double evaluate() const override;
  //void computeBatch(cudaStream_t*, double* output, size_t nEvents, RooFit::DataMap&) const;
  //inline bool canComputeBatchWithCuda() const { return true; }

  ClassDefOverride(RooPolynomial,1) // Polynomial PDF
};

#endif
