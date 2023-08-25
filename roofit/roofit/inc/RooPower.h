/*
 * Project: RooFit
 *
 * Copyright (c) 2022, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#ifndef RooFit_RooPower_h
#define RooFit_RooPower_h

#include <RooAbsPdf.h>
#include <RooRealProxy.h>
#include <RooListProxy.h>

#include <vector>

class RooPower : public RooAbsPdf {
public:
   RooPower() {}
   RooPower(const char *name, const char *title, RooAbsReal &x, const RooArgList &coefList, const RooArgList &expList);

   RooPower(const RooPower &other, const char *name = nullptr);
   TObject *clone(const char *newname) const override { return new RooPower(*this, newname); }

   /// Get the base of the exponentiated terms (aka. x variable).
   RooAbsReal const &base() const { return *_x; }

   /// Get the list of coefficients.
   RooArgList const &coefList() const { return _coefList; }

   /// Get the list of exponents.
   RooArgList const &expList() const { return _expList; }

   std::string getFormulaExpression(bool expand) const;

   int getAnalyticalIntegral(RooArgSet &allVars, RooArgSet &analVars, const char *rangeName = nullptr) const override;
   double analyticalIntegral(int code, const char *rangeName = nullptr) const override;

protected:
   RooRealProxy _x;
   RooListProxy _coefList;
   RooListProxy _expList;

   mutable std::vector<double> _wksp; //! do not persist

   // CUDA support
   void computeBatch(double *output, size_t size, RooFit::Detail::DataMap const &) const override;
   inline bool canComputeBatchWithCuda() const override { return true; }

   /// Evaluation
   double evaluate() const override;

   ClassDefOverride(RooPower, 1) // Power PDF
};

#endif
