#ifndef RooFit_RooFit_RooStudentT_h
#define RooFit_RooFit_RooStudentT_h

#include "RooAbsPdf.h"
#include "RooRealProxy.h"

class RooStudentT : public RooAbsPdf {
public:
   RooStudentT() {};
   RooStudentT(const char *name, const char *title, RooAbsReal::Ref x, RooAbsReal::Ref mean, RooAbsReal::Ref sigma,
               RooAbsReal::Ref ndf);
   RooStudentT(const RooStudentT &other, const char *name = nullptr);
   TObject *clone(const char *newname = nullptr) const override { return new RooStudentT(*this, newname); }

   Int_t getAnalyticalIntegral(RooArgSet &allVars, RooArgSet &analVars, const char *rangeName = nullptr) const override;
   double analyticalIntegral(Int_t code, const char *rangeName = nullptr) const override;

   /// Get the _x variable.
   RooAbsReal const &x() const { return _x.arg(); }

   /// Get the _mean parameter.
   RooAbsReal const &mean() const { return _mean.arg(); }

   /// Get the standard deviation parameter.
   RooAbsReal const &sigma() const { return _sigma.arg(); }

   /// Get the degrees of freedom parameter.
   RooAbsReal const &ndf() const { return _ndf.arg(); }

   Int_t getMaxVal(const RooArgSet &vars) const override;
   double maxVal(Int_t code) const override;

protected:
   double evaluate() const override;

private:
   RooRealProxy _x;     ///< variable
   RooRealProxy _mean;  ///< mean
   RooRealProxy _sigma; ///< standard deviation
   RooRealProxy _ndf;   ///< degrees of freedom

   ClassDefOverride(RooStudentT, 1) // Location-scale Student's t-distribution PDF
};

#endif
