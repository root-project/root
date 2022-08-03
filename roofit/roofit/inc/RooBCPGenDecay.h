/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
 *    File: $Id: RooBCPGenDecay.h,v 1.13 2007/05/11 09:13:07 verkerke Exp $
 * Authors:                                                                  *
 *   JS, Jim Smith    , University of Colorado, jgsmith@pizero.colorado.edu  *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California,         *
 *                          University of Colorado                           *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/
#ifndef ROO_BCPGEN_DECAY
#define ROO_BCPGEN_DECAY

#include "RooAbsAnaConvPdf.h"
#include "RooRealProxy.h"
#include "RooCategoryProxy.h"

class RooBCPGenDecay : public RooAbsAnaConvPdf {
public:

  enum DecayType { SingleSided, DoubleSided, Flipped };

  // Constructors, assignment etc
  inline RooBCPGenDecay() { }
  RooBCPGenDecay(const char *name, const char *title,
       RooRealVar& t, RooAbsCategory& tag,
       RooAbsReal& tau, RooAbsReal& dm,
       RooAbsReal& avgMistag,
       RooAbsReal& a, RooAbsReal& b,
       RooAbsReal& delMistag,
                 RooAbsReal& mu,
       const RooResolutionModel& model, DecayType type=DoubleSided) ;

  RooBCPGenDecay(const RooBCPGenDecay& other, const char* name=nullptr);
  TObject* clone(const char* newname) const override { return new RooBCPGenDecay(*this,newname) ; }
  ~RooBCPGenDecay() override;

  double coefficient(Int_t basisIndex) const override ;

  Int_t getCoefAnalyticalIntegral(Int_t coef, RooArgSet& allVars, RooArgSet& analVars, const char* rangeName=nullptr) const override ;
  double coefAnalyticalIntegral(Int_t coef, Int_t code, const char* rangeName=nullptr) const override ;

  Int_t getGenerator(const RooArgSet& directVars, RooArgSet &generateVars, bool staticInitOK=true) const override;
  void initGenerator(Int_t code) override ;
  void generateEvent(Int_t code) override ;

protected:

  RooRealProxy _avgC ;
  RooRealProxy _avgS ;
  RooRealProxy _avgMistag ;
  RooRealProxy _delMistag ;
  RooRealProxy _mu ;
  RooRealProxy _t ;
  RooRealProxy _tau ;
  RooRealProxy _dm ;
  RooCategoryProxy _tag ;
  double _genB0Frac ;

  DecayType _type ;
  Int_t _basisExp ;
  Int_t _basisSin ;
  Int_t _basisCos ;

  ClassDefOverride(RooBCPGenDecay,1)  // B decay time distribution with CP violation
};

#endif
