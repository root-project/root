/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
 *    File: $Id: RooBMixDecay.h,v 1.14 2007/05/11 09:13:07 verkerke Exp $
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
#ifndef ROO_BMIX_DECAY
#define ROO_BMIX_DECAY

#include "RooAbsAnaConvPdf.h"
#include "RooRealProxy.h"
#include "RooCategoryProxy.h"

class RooBMixDecay : public RooAbsAnaConvPdf {
public:

  enum DecayType { SingleSided, DoubleSided, Flipped };

  // Constructors, assignment etc
  inline RooBMixDecay() { }
  RooBMixDecay(const char *name, const char *title,
          RooRealVar& t, RooAbsCategory& mixState, RooAbsCategory& tagFlav,
          RooAbsReal& tau, RooAbsReal& dm,
          RooAbsReal& mistag, RooAbsReal& delMistag, const RooResolutionModel& model,
          DecayType type=DoubleSided) ;

  RooBMixDecay(const RooBMixDecay& other, const char* name=nullptr);
  TObject* clone(const char* newname) const override { return new RooBMixDecay(*this,newname) ; }
  ~RooBMixDecay() override;

  double coefficient(Int_t basisIndex) const override ;

  Int_t getCoefAnalyticalIntegral(Int_t coef, RooArgSet& allVars, RooArgSet& analVars, const char* rangeName=nullptr) const override ;
  double coefAnalyticalIntegral(Int_t coef, Int_t code, const char* rangeName=nullptr) const override ;

  Int_t getGenerator(const RooArgSet& directVars, RooArgSet &generateVars, bool staticInitOK=true) const override;
  void initGenerator(Int_t code) override ;
  void generateEvent(Int_t code) override;

protected:

  DecayType        _type ;
  RooRealProxy     _mistag ;
  RooRealProxy     _delMistag ;
  RooCategoryProxy _mixState ;
  RooCategoryProxy _tagFlav ;
  RooRealProxy     _tau ;
  RooRealProxy     _dm ;
  RooRealProxy     _t ;
  Int_t _basisExp ;
  Int_t _basisCos ;

  double _genMixFrac ;   //! do not persist
  double _genFlavFrac ;  //!
  double _genFlavFracMix ;   //!
  double _genFlavFracUnmix ; //!

  ClassDefOverride(RooBMixDecay,1) // B Mixing decay PDF
};

#endif
