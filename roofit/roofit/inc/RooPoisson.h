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
#include "RooTrace.h"

class RooPoisson : public RooAbsPdf {
public:
  RooPoisson() { _noRounding = false ;   } ;
  RooPoisson(const char *name, const char *title, RooAbsReal& _x, RooAbsReal& _mean, bool noRounding=false);
  RooPoisson(const RooPoisson& other, const char* name=0) ;
  TObject* clone(const char* newname) const override { return new RooPoisson(*this,newname); }
  inline ~RooPoisson() override {  }

  Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* rangeName=0) const override;
  double analyticalIntegral(Int_t code, const char* rangeName=0) const override;

  Int_t getGenerator(const RooArgSet& directVars, RooArgSet &generateVars, bool staticInitOK=true) const override;
  void generateEvent(Int_t code) override;

  /// Switch off/on rounding of `x` to the nearest integer.
  void setNoRounding(bool flag = true) {_noRounding = flag;}
  /// Switch on or off protection against negative means.
  void protectNegativeMean(bool flag = true) {_protectNegative = flag;}

  /// Get the x variable.
  RooAbsReal const& getX() const { return x.arg(); }

  /// Get the mean parameter.
  RooAbsReal const& getMean() const { return mean.arg(); }

protected:

  RooRealProxy x ;
  RooRealProxy mean ;
  bool  _noRounding ;
  bool  _protectNegative{true};

  double evaluate() const override;
  void computeBatch(cudaStream_t*, double* output, size_t nEvents, RooFit::Detail::DataMap const&) const override;
  inline bool canComputeBatchWithCuda() const override { return true; }

  ClassDefOverride(RooPoisson,3) // A Poisson PDF
};

#endif
