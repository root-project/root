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
   RooPower(const char *name, const char *title, RooAbsReal &x);
   RooPower(const char *name, const char *title, RooAbsReal &x, const RooArgList &coefList, const RooArgList &expList);

   RooPower(const RooPower &other, const char *name = nullptr);
   TObject *clone(const char *newname) const override { return new RooPower(*this, newname); }

   std::string getFormulaExpression(bool expand) const;

   int getAnalyticalIntegral(RooArgSet &allVars, RooArgSet &analVars, const char *rangeName = nullptr) const override;
   double analyticalIntegral(int code, const char *rangeName = nullptr) const override;

protected:
   RooRealProxy _x;
   RooListProxy _coefList;
   RooListProxy _expList;

   mutable std::vector<double> _wksp; //! do not persist

   /// Evaluation
   double evaluate() const override;

   ClassDefOverride(RooPower, 1) // Power PDF
};

#endif
