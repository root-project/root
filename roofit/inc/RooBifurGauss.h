// -*- C++ -*-
/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitModels
 *    File: $Id: RooBifurGauss.rdl,v 1.4 2001/08/23 01:23:34 verkerke Exp $
 * Authors:
 *   Abi Soffer, Coloraro State University, abi@slac.stanford.edu
 * History:
 *   5-Dec-2000 Abi, Created.
 *  19-Jun-2001 JB, Ported to RooFitModels
 *
 * Copyright (C) 2000 Coloraro State University
 *****************************************************************************/
#ifndef ROO_BIFUR_GAUSS
#define ROO_BIFUR_GAUSS

#include "RooFitCore/RooAbsPdf.hh"
#include "RooFitCore/RooRealProxy.hh"

class RooRealVar;

class RooBifurGauss : public RooAbsPdf {
public:
  RooBifurGauss(const char *name, const char *title, RooAbsReal& _x, 
		RooAbsReal& _mean, RooAbsReal& _sigmaL, RooAbsReal& _sigmaR);

  RooBifurGauss(const RooBifurGauss& other, const char* name=0) ;
  virtual TObject* clone(const char* newname) const { return new RooBifurGauss(*this,newname); }
  inline virtual ~RooBifurGauss() { }

  Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const RooArgSet* normSet=0) const ;
  Double_t analyticalIntegral(Int_t code) const ;


protected:

  RooRealProxy x;
  RooRealProxy mean;
  RooRealProxy sigmaL;
  RooRealProxy sigmaR;

  Double_t evaluate() const;

private:

  ClassDef(RooBifurGauss,0) // Bifurcated Gaussian PDF
};

#endif
