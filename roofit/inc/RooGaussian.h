/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitTools
 *    File: $Id$
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   05-Jan-2000 DK Created initial version from RooGaussianProb
 *   02-May-2001 WV Port to RooFitModels/RooFitCore
 *
 * Copyright (C) 2000 Stanford University
 *****************************************************************************/
#ifndef ROO_GAUSSIAN
#define ROO_GAUSSIAN

#include "RooFitCore/RooAbsPdf.hh"
#include "RooFitCore/RooRealProxy.hh"

class RooRealVar;

class RooGaussian : public RooAbsPdf {
public:
  RooGaussian(const char *name, const char *title,
	      RooAbsReal& _x, RooAbsReal& _mean, RooAbsReal& _sigma);
  RooGaussian(const RooGaussian& other, const char* name=0) ;
  virtual TObject* clone() const { return new RooGaussian(*this); }
  inline virtual ~RooGaussian() { }

  Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& numVars) const ;
  Double_t analyticalIntegral(Int_t code) const ;

protected:

  RooRealProxy x ;
  RooRealProxy mean ;
  RooRealProxy sigma ;
  
  Double_t evaluate() const ;

//   void initGenerator();
//   Int_t generateDependents();

private:

  ClassDef(RooGaussian,0) // Gaussian PDF
};

#endif
