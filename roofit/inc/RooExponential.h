/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
 *    File: $Id: RooExponential.rdl,v 1.6 2004/04/05 22:38:34 wverkerke Exp $
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2004, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/
#ifndef ROO_EXPONENTIAL
#define ROO_EXPONENTIAL

#include "RooFitCore/RooAbsPdf.hh"
#include "RooFitCore/RooRealProxy.hh"

class RooRealVar;
class RooAbsReal;

class RooExponential : public RooAbsPdf {
public:
  RooExponential(const char *name, const char *title,
		 RooAbsReal& _x, RooAbsReal& _c);
  RooExponential(const RooExponential& other, const char* name=0);
  virtual TObject* clone(const char* newname) const { return new RooExponential(*this,newname); }
  inline virtual ~RooExponential() { }

  Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* rangeName=0) const ;
  Double_t analyticalIntegral(Int_t code, const char* rangeName=0) const ;

protected:
  RooRealProxy x;
  RooRealProxy c;

  Double_t evaluate() const;

private:
  ClassDef(RooExponential,0) // Exponential PDF
};

#endif
