/*
 * Project: RooFit
 * Authors:
 *   Jonas Rembser, CERN 2022
 *
 * Copyright (c) 2022, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#ifndef RooFit_RooNormalizedPdf_h
#define RooFit_RooNormalizedPdf_h

#include <RooAbsPdf.h>
#include <RooRealProxy.h>

namespace RooFit {
namespace Detail {

class RooNormalizedPdf : public RooAbsPdf {
public:
   RooNormalizedPdf(RooAbsPdf &pdf, RooArgSet const &normSet)
      : _pdf("numerator", "numerator", this, pdf),
        _normIntegral(
           "denominator", "denominator", this,
           std::unique_ptr<RooAbsReal>{pdf.createIntegral(normSet, *pdf.getIntegratorConfig(), pdf.normRange())}, true,
           false),
        _normSet{normSet}
   {
      auto name = std::string(pdf.GetName()) + "_over_" + _normIntegral->GetName();
      SetName(name.c_str());
      SetTitle(name.c_str());
      _normRange = pdf.normRange(); // so that e.g. RooAddPdf can query over what we are normalized
   }

   RooNormalizedPdf(const RooNormalizedPdf &other, const char *name)
      : RooAbsPdf(other, name),
        _pdf("numerator", this, other._pdf),
        _normIntegral("denominator", this, other._normIntegral),
        _normSet{other._normSet}
   {
   }

   TObject *clone(const char *newname) const override { return new RooNormalizedPdf(*this, newname); }

   bool selfNormalized() const override { return true; }

   inline double getCorrection() const override { return _pdf->getCorrection(); }

   bool forceAnalyticalInt(const RooAbsArg & /*dep*/) const override { return true; }
   /// Forward determination of analytical integration capabilities to input p.d.f
   Int_t getAnalyticalIntegralWN(RooArgSet &allVars, RooArgSet &analVars, const RooArgSet * /*normSet*/,
                                 const char *rangeName = nullptr) const override
   {
      return _pdf->getAnalyticalIntegralWN(allVars, analVars, &_normSet, rangeName);
   }
   /// Forward calculation of analytical integrals to input p.d.f
   double
   analyticalIntegralWN(Int_t code, const RooArgSet * /*normSet*/, const char *rangeName = nullptr) const override
   {
      return _pdf->analyticalIntegralWN(code, &_normSet, rangeName);
   }

   ExtendMode extendMode() const override { return static_cast<RooAbsPdf &>(*_pdf).extendMode(); }
   double expectedEvents(const RooArgSet * /*nset*/) const override { return _pdf->expectedEvents(&_normSet); }

   std::unique_ptr<RooAbsReal> createExpectedEventsFunc(const RooArgSet * /*nset*/) const override
   {
      return _pdf->createExpectedEventsFunc(&_normSet);
   }

   bool canComputeBatchWithCuda() const override { return true; }

   RooAbsPdf const &pdf() const { return *_pdf; }
   RooAbsReal const &normIntegral() const { return *_normIntegral; }

protected:
   void doEval(RooFit::EvalContext &) const override;
   double evaluate() const override
   {
      // The evaluate() function should not be called in the BatchMode, but we
      // still need it to support printing of the object.
      return getValV(nullptr);
   }
   double getValV(const RooArgSet * /*normSet*/) const override
   {
      return normalizeWithNaNPacking(_pdf->getVal(), _normIntegral->getVal());
   };

private:
   RooTemplateProxy<RooAbsPdf> _pdf;
   RooRealProxy _normIntegral;
   RooArgSet _normSet;

   ClassDefOverride(RooFit::Detail::RooNormalizedPdf, 0);
};

} // namespace Detail
} // namespace RooFit

#endif
