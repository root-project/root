/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitTools
 *    File: $Id: RooExponential.rdl,v 1.2 2001/01/23 19:36:16 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   AB, Adrian Bevan, Liverpool University, bevan@slac.stanford.edu
 * History:
 *   05-Jan-2000 DK Created initial version from RooLifetimeProb
 *   21-Aug-2001 AB Portto RooFitCore/RooFitModels
 *
 * Copyright (C) 2000 Stanford University
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

  Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars) const ;
  Double_t analyticalIntegral(Int_t code) const ;

protected:
  RooRealProxy x;
  RooRealProxy c;

  Double_t evaluate(const RooArgSet * nset) const;

//  void useParametersImpl();
//  void initGenerator();
//  Int_t generateDependents();
//  Double_t  _exp1, _exp2;
//  RooRealVar *_xptr;

private:
  ClassDef(RooExponential,0) // Exponential PDF
};

#endif
