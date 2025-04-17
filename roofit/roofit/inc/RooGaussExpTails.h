#ifndef RooFit_RooFit_RooGaussExpTails_h
#define RooFit_RooFit_RooGaussExpTails_h

#include "RooAbsPdf.h"
#include "RooRealProxy.h"

class RooAbsReal;

class RooGaussExpTails : public RooAbsPdf {
public:
  RooGaussExpTails() { };
  RooGaussExpTails(const char *name, const char *title,
                         RooAbsReal::Ref _x,
                         RooAbsReal::Ref _x0,
                         RooAbsReal::Ref _sigma,
                         RooAbsReal::Ref _kL,
                         RooAbsReal::Ref _kH
                         );
  RooGaussExpTails(const RooGaussExpTails& other, const char* name=nullptr);
  TObject* clone(const char* newname) const override {
    return new RooGaussExpTails(*this,newname);
  }

  Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* rangeName=nullptr) const override;
  double analyticalIntegral(Int_t code, const char* rangeName=nullptr) const override;

  RooAbsReal::Ref x() const { return *x_; }
  RooAbsReal::Ref x0() const { return *x0_; }
  RooAbsReal::Ref sigma() const { return *sigma_; }
  RooAbsReal::Ref kL() const { return *kL_; }
  RooAbsReal::Ref kH() const { return *kH_; }

protected:
  double evaluate() const override;

private:
  RooRealProxy x_;
  RooRealProxy x0_;
  RooRealProxy sigma_;
  RooRealProxy kL_;
  RooRealProxy kH_;

private:

  ClassDefOverride(RooGaussExpTails,1) // Gaussian with double-sided exponential tails PDF, see https://arxiv.org/abs/1603.08591v1
};

#endif
