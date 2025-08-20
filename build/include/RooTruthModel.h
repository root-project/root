/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooTruthModel.h,v 1.18 2007/05/11 10:14:56 verkerke Exp $
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
#ifndef ROO_TRUTH_MODEL
#define ROO_TRUTH_MODEL

#include "RooResolutionModel.h"

class RooTruthModel : public RooResolutionModel {
public:
  // Constructors, assignment etc
  RooTruthModel() = default;
  RooTruthModel(const char *name, const char *title, RooAbsRealLValue& x) ;
  RooTruthModel(const RooTruthModel& other, const char* name=nullptr) : RooResolutionModel{other, name} {}
  TObject* clone(const char* newname=nullptr) const override { return new RooTruthModel(*this,newname) ; }

  Int_t basisCode(const char* name) const override ;

  RooAbsGenContext* modelGenContext(const RooAbsAnaConvPdf& convPdf, const RooArgSet &vars,
                                            const RooDataSet *prototype=nullptr, const RooArgSet* auxProto=nullptr,
                                            bool verbose= false) const override;

  Int_t getGenerator(const RooArgSet& directVars, RooArgSet &generateVars, bool staticInitOK=true) const override;
  void generateEvent(Int_t code) override;

  Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* rangeName=nullptr) const override ;
  double analyticalIntegral(Int_t code, const char* rangeName=nullptr) const override ;

  void doEval(RooFit::EvalContext &) const override;
  inline bool canComputeBatchWithCuda() const override { return true; }

protected:
  double evaluate() const override ;
  void changeBasis(RooFormulaVar* basis) override ;

  ClassDefOverride(RooTruthModel,1) // Truth resolution model (delta function)
};

#endif
