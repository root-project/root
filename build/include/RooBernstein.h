/*
 * Project: RooFit
 *
 * Copyright (c) 2024, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#ifndef RooFit_RooBernstein_h
#define RooFit_RooBernstein_h

#include <RooAbsPdf.h>
#include <RooAbsRealLValue.h>
#include <RooListProxy.h>
#include <RooRealVar.h>
#include <RooTemplateProxy.h>

#include <string>

class RooBernstein : public RooAbsPdf {
public:
   RooBernstein() = default;
   RooBernstein(const char *name, const char *title, RooAbsRealLValue &_x, const RooArgList &_coefList);

   RooBernstein(const RooBernstein &other, const char *name = nullptr);

   TObject *clone(const char *newname) const override { return new RooBernstein(*this, newname); }

   Int_t getAnalyticalIntegral(RooArgSet &allVars, RooArgSet &analVars, const char *rangeName = nullptr) const override;
   double analyticalIntegral(Int_t code, const char *rangeName = nullptr) const override;
   void selectNormalizationRange(const char *rangeName = nullptr, bool force = false) override;

   RooAbsRealLValue const &x() const { return *_x; }
   RooArgList const &coefList() const { return _coefList; }

   // Implementation detail. Do not use.
   void fillBuffer() const;
   // Implementation detail. Do not use.
   inline double xmin() const { return _buffer[_coefList.size()]; }
   // Implementation detail. Do not use.
   inline double xmax() const { return _buffer[_coefList.size() + 1]; }

private:

   RooTemplateProxy<RooAbsRealLValue> _x;
   RooListProxy _coefList;
   std::string _refRangeName;
   mutable std::vector<double> _buffer; ///<!

   double evaluate() const override;
   void doEval(RooFit::EvalContext &) const override;
   inline bool canComputeBatchWithCuda() const override { return true; }

   ClassDefOverride(RooBernstein, 2) // Bernstein polynomial PDF
};

#endif
