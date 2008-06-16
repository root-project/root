/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
 *    File: $Id: RooDecay.h,v 1.11 2007/05/11 09:13:07 verkerke Exp $
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
#ifndef ROO_DECAY
#define ROO_DECAY

#include "RooAbsAnaConvPdf.h"
#include "RooRealProxy.h"

class RooDecay : public RooAbsAnaConvPdf {
public:

  enum DecayType { SingleSided, DoubleSided, Flipped };

  // Constructors, assignment etc
  inline RooDecay() { }
  RooDecay(const char *name, const char *title, RooRealVar& t, 
	   RooAbsReal& tau, const RooResolutionModel& model, DecayType type) ;
  RooDecay(const RooDecay& other, const char* name=0);
  virtual TObject* clone(const char* newname) const { return new RooDecay(*this,newname) ; }
  virtual ~RooDecay();

  virtual Double_t coefficient(Int_t basisIndex) const ;

  Int_t getGenerator(const RooArgSet& directVars, RooArgSet &generateVars, Bool_t staticInitOK=kTRUE) const;
  void generateEvent(Int_t code);
  
protected:
  
  RooRealProxy _t ;
  RooRealProxy _tau ;
  DecayType    _type ;
  Int_t        _basisExp ;

  ClassDef(RooDecay,1) // General decay function p.d.f 
};

#endif
