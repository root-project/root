/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels
 *
 * authors: Stefan A. Schmitz, Gregory Schott
 * implementation of the Gamma distribution (class structure derived
 * from the class RooGaussian by Wouter Verkerke and David Kirkby)
 *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

#ifndef ROO_GAMMA
#define ROO_GAMMA

#include "RooAbsPdf.h"
#include "RooRealProxy.h"

class RooGamma : public RooAbsPdf {
public:
  RooGamma() {} ;
  RooGamma(const char *name, const char *title,
         RooAbsReal& _x, RooAbsReal& _gamma, RooAbsReal& _beta, RooAbsReal& _mu);
  RooGamma(const RooGamma& other, const char* name=nullptr) ;
  TObject* clone(const char* newname) const override { return new RooGamma(*this,newname); }

  Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* rangeName=nullptr) const override ;
  double analyticalIntegral(Int_t code, const char* rangeName=nullptr) const override ;

  Int_t getGenerator(const RooArgSet& directVars, RooArgSet &generateVars, bool staticInitOK=true) const override;
  void generateEvent(Int_t code) override;

  RooAbsReal const &getX() const { return *x; }
  RooAbsReal const &getGamma() const { return *gamma; }
  RooAbsReal const &getBeta() const { return *beta; }
  RooAbsReal const &getMu() const { return *mu; }

protected:

  RooRealProxy x ;
  RooRealProxy gamma ;
  RooRealProxy beta ;
  RooRealProxy mu ;

  double evaluate() const override ;
  void doEval(RooFit::EvalContext &) const override;
  inline bool canComputeBatchWithCuda() const override { return true; }

private:

  ClassDefOverride(RooGamma,1) // Gaussian PDF
};

#endif
