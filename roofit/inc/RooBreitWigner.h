/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitModels
 *    File: $Id: RooBreitWigner.rdl,v 1.6 2001/08/23 01:23:35 verkerke Exp $
 * Authors:
 *   AS, Abi Soffer, Colorado State University, abi@slac.stanford.edu
 *   TS, Thomas Schietinger, SLAC, schieti@slac.stanford.edu
 * History:
 *   13-Mar-2001 AS Created.
 *   14-Sep-2001 TS Port to RooFitModels/RooFitCore
 *
 * Copyright (C) 2001 Colorado State University
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
