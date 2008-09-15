 /***************************************************************************** 
  * Project: RooFit                                                           * 
  *                                                                           * 
  * Simple Poisson PDF
  * author: Kyle Cranmer <cranmer@cern.ch>
  *                                                                           * 
  *****************************************************************************/ 

#ifndef ROOPOISSON
#define ROOPOISSON

#include "RooAbsPdf.h"
#include "RooRealProxy.h"
#include "RooCategoryProxy.h"
#include "RooAbsReal.h"
#include "RooAbsCategory.h"
 
class RooPoisson : public RooAbsPdf {
public:
  RooPoisson() {} ;
  RooPoisson(const char *name, const char *title, RooAbsReal& _x, RooAbsReal& _mean);
  RooPoisson(const RooPoisson& other, const char* name=0) ;
  virtual TObject* clone(const char* newname) const { return new RooPoisson(*this,newname); }
  inline virtual ~RooPoisson() { }

  Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* rangeName=0) const ;
  Double_t analyticalIntegral(Int_t code, const char* rangeName=0) const ;

  Int_t getGenerator(const RooArgSet& directVars, RooArgSet &generateVars, Bool_t staticInitOK=kTRUE) const;
  void generateEvent(Int_t code);

protected:

  RooRealProxy x ;
  RooRealProxy mean ;
  
  Double_t evaluate() const ;
  Double_t evaluate(Double_t k) const;


private:

  ClassDef(RooPoisson,1) // A Poisson PDF
};
 
#endif
