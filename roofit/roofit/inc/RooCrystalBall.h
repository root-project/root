// Author: Jonas Rembser, CERN  02/2021

#ifndef RooFit_RooFit_RooCrystalBall_h
#define RooFit_RooFit_RooCrystalBall_h

#include "RooAbsPdf.h"
#include "RooRealProxy.h"

#include <memory>

class RooRealVar;

class RooCrystalBall final : public RooAbsPdf {
public:

   RooCrystalBall(){};

   RooCrystalBall(const char *name, const char *title, RooAbsReal &x, RooAbsReal &x0, RooAbsReal &sigmaL,
                  RooAbsReal &sigmaR, RooAbsReal &alphaL, RooAbsReal &nL, RooAbsReal &alphaR, RooAbsReal &nR);
   RooCrystalBall(const char *name, const char *title, RooAbsReal &x, RooAbsReal &x0, RooAbsReal &sigmaLR,
                  RooAbsReal &alphaL, RooAbsReal &nL, RooAbsReal &alphaR, RooAbsReal &nR);
   RooCrystalBall(const char *name, const char *title, RooAbsReal &x, RooAbsReal &x0, RooAbsReal &sigmaLR,
                  RooAbsReal &alpha, RooAbsReal &n, bool doubleSided = false);

   RooCrystalBall(const RooCrystalBall &other, const char *name = 0);
   TObject *clone(const char *newname) const override { return new RooCrystalBall(*this, newname); }

   inline ~RooCrystalBall() override {}

   Int_t getAnalyticalIntegral(RooArgSet &allVars, RooArgSet &analVars, const char *rangeName = 0) const override;
   double analyticalIntegral(Int_t code, const char *rangeName = 0) const override;

   // Optimized accept/reject generator support
   Int_t getMaxVal(const RooArgSet &vars) const override;
   double maxVal(Int_t code) const override;

protected:
   double evaluate() const override;

private:
   RooRealProxy x_;
   RooRealProxy x0_;
   RooRealProxy sigmaL_;
   RooRealProxy sigmaR_;
   RooRealProxy alphaL_;
   RooRealProxy nL_;

   // optional parameters
   std::unique_ptr<RooRealProxy> alphaR_ = nullptr;
   std::unique_ptr<RooRealProxy> nR_ = nullptr;

   ClassDefOverride(RooCrystalBall, 1)
};

#endif
