/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
 *    File: $Id: RooChebychev.rdl,v 1.2 2002/09/10 02:01:31 verkerke Exp $
 * Authors:                                                                  *
 *   GR, Gerhard Raven,   UC San Diego, Gerhard.Raven@slac.stanford.edu
 *                                                                           *
 * Copyright (c) 2000-2004, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/
#ifndef ROO_CHEBYCHEV
#define ROO_CHEBYCHEV

#include "RooFitCore/RooAbsPdf.hh"
#include "RooFitCore/RooRealProxy.hh"
#include "RooFitCore/RooListProxy.hh"

class RooRealVar;
class RooArgList ;

class RooChebychev : public RooAbsPdf {
public:

  RooChebychev() ;
  RooChebychev(const char *name, const char *title,
               RooAbsReal& _x, const RooArgList& _coefList) ;

  RooChebychev(const RooChebychev& other, const char* name = 0);
  virtual TObject* clone(const char* newname) const { return new RooChebychev(*this, newname); }
  inline virtual ~RooChebychev() { }

  Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars) const ;
  Double_t analyticalIntegral(Int_t code) const ;

private:

  RooRealProxy _x;
  RooListProxy _coefList ;

  Double_t evaluate() const;

  ClassDef(RooChebychev,1) // Chebychev PDF
};

#endif
