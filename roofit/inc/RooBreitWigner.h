/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
 *    File: $Id: RooBreitWigner.rdl,v 1.4 2002/09/10 02:01:31 verkerke Exp $
 * Authors:                                                                  *
 *   AS, Abi Soffer, Colorado State University, abi@slac.stanford.edu        *
 *   TS, Thomas Schietinger, SLAC, schieti@slac.stanford.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2004, Regents of the University of California          *
 *                          Colorado State University                        *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/
#ifndef ROO_BREITWIGNER
#define ROO_BREITWIGNER

#include "RooFitCore/RooAbsPdf.hh"
#include "RooFitCore/RooRealProxy.hh"

class RooRealVar;

class RooBreitWigner : public RooAbsPdf {
public:
  RooBreitWigner(const char *name, const char *title,
	      RooAbsReal& _x, RooAbsReal& _mean, RooAbsReal& _width);
  RooBreitWigner(const RooBreitWigner& other, const char* name=0) ;
  virtual TObject* clone(const char* newname) const { return new RooBreitWigner(*this,newname); }
  inline virtual ~RooBreitWigner() { }

  Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars) const ;
  Double_t analyticalIntegral(Int_t code) const ;

protected:

  RooRealProxy x ;
  RooRealProxy mean ;
  RooRealProxy width ;
  
  Double_t evaluate() const ;

//   void initGenerator();
//   Int_t generateDependents();

private:

  ClassDef(RooBreitWigner,0) // Breit Wigner PDF
};

#endif
