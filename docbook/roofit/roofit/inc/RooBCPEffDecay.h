/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
 *    File: $Id: RooBCPEffDecay.h,v 1.13 2007/05/11 09:13:07 verkerke Exp $
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
#ifndef ROO_BCP_EFF_DECAY
#define ROO_BCP_EFF_DECAY

#include "RooAbsAnaConvPdf.h"
#include "RooRealProxy.h"
#include "RooCategoryProxy.h"

class RooBCPEffDecay : public RooAbsAnaConvPdf {
public:

  enum DecayType { SingleSided, DoubleSided, Flipped };

  // Constructors, assignment etc
  inline RooBCPEffDecay() { }
  RooBCPEffDecay(const char *name, const char *title, 
		 RooRealVar& t, RooAbsCategory& tag,
		 RooAbsReal& tau, RooAbsReal& dm,
		 RooAbsReal& avgMistag, RooAbsReal& CPeigenval,
		 RooAbsReal& a, RooAbsReal& b,
		 RooAbsReal& effRatio, RooAbsReal& delMistag,
		 const RooResolutionModel& model, DecayType type=DoubleSided) ;

  RooBCPEffDecay(const RooBCPEffDecay& other, const char* name=0);
  virtual TObject* clone(const char* newname) const { return new RooBCPEffDecay(*this,newname) ; }
  virtual ~RooBCPEffDecay();

  virtual Double_t coefficient(Int_t basisIndex) const ;

  virtual Int_t getCoefAnalyticalIntegral(Int_t coef, RooArgSet& allVars, RooArgSet& analVars, const char* rangeName=0) const ;
  virtual Double_t coefAnalyticalIntegral(Int_t coef, Int_t code, const char* rangeName=0) const ;

  Int_t getGenerator(const RooArgSet& directVars, RooArgSet &generateVars, Bool_t staticInitOK=kTRUE) const;
  void initGenerator(Int_t code) ;
  void generateEvent(Int_t code) ;
  
protected:

  RooRealProxy _absLambda ;
  RooRealProxy _argLambda ;
  RooRealProxy _effRatio ;
  RooRealProxy _CPeigenval ;
  RooRealProxy _avgMistag ;
  RooRealProxy _delMistag ;
  RooRealProxy _t ;
  RooRealProxy _tau ;
  RooRealProxy _dm ;
  RooCategoryProxy _tag ;
  Double_t _genB0Frac ;
  
  DecayType _type ;
  Int_t _basisExp ;
  Int_t _basisSin ;
  Int_t _basisCos ;

  ClassDef(RooBCPEffDecay,1) // B Mixing decay PDF
};

#endif
