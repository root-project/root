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

#include <RooAbsPdf.h>
#include <RooRealProxy.h>

class RooLognormal : public RooAbsPdf {
public:
   RooLognormal() {}
   RooLognormal(const char *name, const char *title, RooAbsReal &_x, RooAbsReal &_m0, RooAbsReal &_k,
                bool useStandardParametrization = false);
   RooLognormal(const RooLognormal &other, const char *name = nullptr);
   TObject *clone(const char *newname) const override { return new RooLognormal(*this, newname); }

   Int_t getAnalyticalIntegral(RooArgSet &allVars, RooArgSet &analVars, const char *rangeName = nullptr) const override;
   double analyticalIntegral(Int_t code, const char *rangeName = nullptr) const override;

   Int_t getGenerator(const RooArgSet &directVars, RooArgSet &generateVars, bool staticInitOK = true) const override;
   void generateEvent(Int_t code) override;

   /// Get the x variable.
   RooAbsReal const &getX() const { return x.arg(); }

   /// Get the median parameter.
   RooAbsReal const &getMedian() const { return m0.arg(); }

   /// Get the shape parameter.
   RooAbsReal const &getShapeK() const { return k.arg(); }

   bool useStandardParametrization() const { return _useStandardParametrization; }

protected:
   RooRealProxy x;  ///< the variable
   RooRealProxy m0; ///< the median, exp(mu)
   RooRealProxy k;  ///< the shape parameter, exp(sigma)
   bool _useStandardParametrization = false;

   double evaluate() const override;
   void doEval(RooFit::EvalContext &) const override;
   inline bool canComputeBatchWithCuda() const override { return true; }

private:
   ClassDefOverride(RooLognormal, 2) // log-normal PDF
};

#endif
