/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitTools
 *    File: $Id$
 * Authors:
 *   GR, Gerhard Raven, UC San Diego, Gerhard.Raven@slac.stanford.edu
 * History:
 *   20-Oct-2001 GR Initial RFC version
 *
 * Copyright (C) 2001 University of California
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
