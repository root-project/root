 /*****************************************************************************
  * Project: RooFit                                                           *
  * @(#)root/roofit:$Id$ *
  *                                                                           *
  * RooFit Lognormal PDF                                                      *
  *                                                                           *
  * Author: Gregory Schott and Stefan Schmitz                                 *
  *                                                                           *
  *****************************************************************************/

#ifndef ROO_LOGNORMAL
#define ROO_LOGNORMAL

#include "RooAbsPdf.h"
#include "RooRealProxy.h"

class RooRealVar;

class RooLognormal : public RooAbsPdf {
public:
  RooLognormal() {} ;
  RooLognormal(const char *name, const char *title,
         RooAbsReal& _x, RooAbsReal& _m0, RooAbsReal& _k);
  RooLognormal(const RooLognormal& other, const char* name=nullptr) ;
  TObject* clone(const char* newname) const override { return new RooLognormal(*this,newname); }
  inline ~RooLognormal() override { }

  Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* rangeName=nullptr) const override ;
  double analyticalIntegral(Int_t code, const char* rangeName=nullptr) const override ;

  Int_t getGenerator(const RooArgSet& directVars, RooArgSet &generateVars, bool staticInitOK=true) const override;
  void generateEvent(Int_t code) override;

protected:

  RooRealProxy x ;
  RooRealProxy m0 ;
  RooRealProxy k ;

  double evaluate() const override ;
  void computeBatch(cudaStream_t*, double* output, size_t nEvents, RooFit::Detail::DataMap const&) const override;
  inline bool canComputeBatchWithCuda() const override { return true; }

private:

  ClassDefOverride(RooLognormal,1) // log-normal PDF
};

#endif
