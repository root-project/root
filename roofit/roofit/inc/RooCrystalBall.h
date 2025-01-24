// Author: Jonas Rembser, CERN  02/2021

#ifndef RooFit_RooFit_RooCrystalBall_h
#define RooFit_RooFit_RooCrystalBall_h

#include "RooAbsPdf.h"
#include "RooRealProxy.h"

#include <memory>

class RooCrystalBall final : public RooAbsPdf {
public:
   RooCrystalBall() {};

   RooCrystalBall(const char *name, const char *title, RooAbsReal &x, RooAbsReal &x0, RooAbsReal &sigmaL,
                  RooAbsReal &sigmaR, RooAbsReal &alphaL, RooAbsReal &nL, RooAbsReal &alphaR, RooAbsReal &nR);
   RooCrystalBall(const char *name, const char *title, RooAbsReal &x, RooAbsReal &x0, RooAbsReal &sigmaLR,
                  RooAbsReal &alphaL, RooAbsReal &nL, RooAbsReal &alphaR, RooAbsReal &nR);
   RooCrystalBall(const char *name, const char *title, RooAbsReal &x, RooAbsReal &x0, RooAbsReal &sigmaLR,
                  RooAbsReal &alpha, RooAbsReal &n, bool doubleSided = false);

   RooCrystalBall(const RooCrystalBall &other, const char *name = nullptr);
   TObject *clone(const char *newname) const override { return new RooCrystalBall(*this, newname); }

   Int_t getAnalyticalIntegral(RooArgSet &allVars, RooArgSet &analVars, const char *rangeName = nullptr) const override;
   double analyticalIntegral(Int_t code, const char *rangeName = nullptr) const override;

   // Optimized accept/reject generator support
   Int_t getMaxVal(const RooArgSet &vars) const override;
   double maxVal(Int_t code) const override;

   // Getters for non-optional parameters
   RooAbsReal const &x() const { return *x_; }
   RooAbsReal const &x0() const { return *x0_; }
   RooAbsReal const &sigmaL() const { return *sigmaL_; }
   RooAbsReal const &sigmaR() const { return *sigmaR_; }
   RooAbsReal const &alphaL() const { return *alphaL_; }
   RooAbsReal const &nL() const { return *nL_; }

   // Getters for optional parameter: return nullptr if parameter is not set
   RooAbsReal const *alphaR() const { return alphaR_ ? &**alphaR_ : nullptr; }
   RooAbsReal const *nR() const { return nR_ ? &**nR_ : nullptr; }

   // Convenience functions to check if optional parameters are set
   bool hasAlphaR() const { return alphaR_ != nullptr; }
   bool hasNR() const { return nR_ != nullptr; }

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

   ClassDefOverride(RooCrystalBall, 2)
};

#endif
