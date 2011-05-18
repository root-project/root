/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
 *    File: $Id: RooBDecay.h,v 1.7 2007/05/11 09:13:07 verkerke Exp $
 * Authors:                                                                  *
 *   PL, Parker C Lund,   UC Irvine                                          *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

#ifndef ROO_BDECAY
#define ROO_BDECAY

#include "RooAbsAnaConvPdf.h"
#include "RooRealProxy.h"

class RooBDecay : public RooAbsAnaConvPdf
{

public:

  enum DecayType { SingleSided, DoubleSided, Flipped };

  //Constructors, assignment etc
  inline RooBDecay() { }
  RooBDecay(const char *name, const char *title, RooRealVar& t,
			RooAbsReal& tau, RooAbsReal& dgamma,
			RooAbsReal& f0,
			RooAbsReal& f1, RooAbsReal& f2, 
			RooAbsReal& f3, RooAbsReal& dm, 
			const RooResolutionModel& model,
			DecayType type);
  RooBDecay(const RooBDecay& other, const char* name=0);
  virtual TObject* clone(const char* newname) const 
  { 
    return new RooBDecay(*this,newname);
  }
  virtual ~RooBDecay();

  virtual Double_t coefficient(Int_t basisIndex) const;
  RooArgSet* coefVars(Int_t coefIdx) const ;

  Int_t getCoefAnalyticalIntegral(Int_t coef, RooArgSet& allVars, RooArgSet& analVars, const char* rangeName=0) const ;
  Double_t coefAnalyticalIntegral(Int_t coef, Int_t code, const char* rangeName=0) const ;

  Int_t getGenerator(const RooArgSet& directVars, RooArgSet &generateVars, Bool_t staticInitOK=kTRUE) const;
  void generateEvent(Int_t code);

protected:

  RooRealProxy _t;
  RooRealProxy _tau;
  RooRealProxy _dgamma;
  RooRealProxy _f0;
  RooRealProxy _f1;
  RooRealProxy _f2;
  RooRealProxy _f3;
  RooRealProxy _dm;	
  Int_t _basisCosh;
  Int_t _basisSinh;
  Int_t _basisCos;
  Int_t _basisSin;
  Int_t _basisB;
  DecayType _type;

  ClassDef(RooBDecay, 1) // P.d.f of general description of B decay time distribution
    };

#endif

