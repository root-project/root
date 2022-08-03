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

  enum RooTruthBasis { noBasis=0, expBasisMinus= 1, expBasisSum= 2, expBasisPlus= 3,
                                  sinBasisMinus=11, sinBasisSum=12, sinBasisPlus=13,
                                  cosBasisMinus=21, cosBasisSum=22, cosBasisPlus=23,
                                                          linBasisPlus=33,
                                                         quadBasisPlus=43,
              coshBasisMinus=51,coshBasisSum=52,coshBasisPlus=53,
                 sinhBasisMinus=61,sinhBasisSum=62,sinhBasisPlus=63,
                       genericBasis=100 } ;

  enum BasisType { none=0, expBasis=1, sinBasis=2, cosBasis=3,
                   linBasis=4, quadBasis=5, coshBasis=6, sinhBasis=7 } ;
  enum BasisSign { Both=0, Plus=+1, Minus=-1 } ;

  // Constructors, assignment etc
  inline RooTruthModel() { }
  RooTruthModel(const char *name, const char *title, RooAbsRealLValue& x) ;
  RooTruthModel(const RooTruthModel& other, const char* name=nullptr);
  TObject* clone(const char* newname) const override { return new RooTruthModel(*this,newname) ; }
  ~RooTruthModel() override;

  Int_t basisCode(const char* name) const override ;

  RooAbsGenContext* modelGenContext(const RooAbsAnaConvPdf& convPdf, const RooArgSet &vars,
                                            const RooDataSet *prototype=nullptr, const RooArgSet* auxProto=nullptr,
                                            bool verbose= false) const override;

  Int_t getGenerator(const RooArgSet& directVars, RooArgSet &generateVars, bool staticInitOK=true) const override;
  void generateEvent(Int_t code) override;

  Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* rangeName=nullptr) const override ;
  double analyticalIntegral(Int_t code, const char* rangeName=nullptr) const override ;

protected:
  double evaluate() const override ;
  void changeBasis(RooFormulaVar* basis) override ;

  ClassDefOverride(RooTruthModel,1) // Truth resolution model (delta function)
};

#endif
