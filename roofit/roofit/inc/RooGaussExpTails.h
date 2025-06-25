#ifndef RooFit_RooFit_RooGaussExpTails_h
#define RooFit_RooFit_RooGaussExpTails_h

#include "RooAbsPdf.h"
#include "RooRealProxy.h"

class RooAbsReal;

class RooGaussExpTails : public RooAbsPdf {
public:
   RooGaussExpTails() {}
   RooGaussExpTails(const char *name, const char *title, RooAbsReal::Ref x, RooAbsReal::Ref x0, RooAbsReal::Ref sigma,
                    RooAbsReal::Ref kL, RooAbsReal::Ref kH);
   RooGaussExpTails(const RooGaussExpTails &other, const char *name = nullptr);
   TObject *clone(const char *newname) const override { return new RooGaussExpTails(*this, newname); }

   Int_t getAnalyticalIntegral(RooArgSet &allVars, RooArgSet &analVars, const char *rangeName = nullptr) const override;
   double analyticalIntegral(Int_t code, const char *rangeName = nullptr) const override;

   RooAbsReal const &x() const { return *_x; }
   RooAbsReal const &x0() const { return *_x0; }
   RooAbsReal const &sigma() const { return *_sigma; }
   RooAbsReal const &kL() const { return *_kL; }
   RooAbsReal const &kH() const { return *_kH; }

protected:
   double evaluate() const override;

private:
   RooRealProxy _x;
   RooRealProxy _x0;
   RooRealProxy _sigma;
   RooRealProxy _kL;
   RooRealProxy _kH;

private:
   ClassDefOverride(RooGaussExpTails, 1) // Gaussian with double-sided exponential tails PDF, see https://arxiv.org/abs/1603.08591v1
};

#endif
