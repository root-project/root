/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
 *    File: $Id: RooBCPGenDecay.rdl,v 1.12 2005/06/20 15:51:06 wverkerke Exp $
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

  RooBCPGenDecay(const RooBCPGenDecay& other, const char* name=0);
  virtual TObject* clone(const char* newname) const { return new RooBCPGenDecay(*this,newname) ; }
  virtual ~RooBCPGenDecay();

  virtual Double_t coefficient(Int_t basisIndex) const ;

  virtual Int_t getCoefAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* rangeName=0) const ;
  virtual Double_t coefAnalyticalIntegral(Int_t coef, Int_t code, const char* rangeName=0) const ;

  Int_t getGenerator(const RooArgSet& directVars, RooArgSet &generateVars, Bool_t staticInitOK=kTRUE) const;
  void initGenerator(Int_t code) ;
  void generateEvent(Int_t code) ;
  
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
  Double_t _genB0Frac ;
  
  DecayType _type ;
  Int_t _basisExp ;
  Int_t _basisSin ;
  Int_t _basisCos ;

  ClassDef(RooBCPGenDecay,1) 
};

#endif
