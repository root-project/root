/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
 *    File: $Id: RooPolynomial.h,v 1.8 2007/05/11 09:13:07 verkerke Exp $
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/
#ifndef ROO_POLYNOMIAL
#define ROO_POLYNOMIAL

#include <RooAbsPdf.h>
#include <RooRealProxy.h>
#include <RooListProxy.h>

#include <vector>

class RooPolynomial : public RooAbsPdf {
public:
   RooPolynomial() {}
   RooPolynomial(const char *name, const char *title, RooAbsReal &x);
   RooPolynomial(const char *name, const char *title, RooAbsReal &_x, const RooArgList &_coefList,
                 Int_t lowestOrder = 1);

   RooPolynomial(const RooPolynomial &other, const char *name = nullptr);
   TObject *clone(const char *newname) const override { return new RooPolynomial(*this, newname); }

   Int_t getAnalyticalIntegral(RooArgSet &allVars, RooArgSet &analVars, const char *rangeName = nullptr) const override;
   double analyticalIntegral(Int_t code, const char *rangeName = nullptr) const override;

   /// Get the x variable.
   RooAbsReal const &x() const { return _x.arg(); }

   /// Get the coefficient list.
   RooArgList const &coefList() const { return _coefList; }

   /// Return the order for the first coefficient in the list.
   int lowestOrder() const { return _lowestOrder; }

   // If this polynomial has no terms it's a uniform distribution, and a uniform
   // pdf is a reducer node because it doesn't depend on the observables.
   bool isReducerNode() const override { return _coefList.empty(); }

protected:
   RooRealProxy _x;
   RooListProxy _coefList;
   Int_t _lowestOrder = 1;

   mutable std::vector<double> _wksp; //! do not persist

   /// Evaluation
   double evaluate() const override;
   void doEval(RooFit::EvalContext &) const override;

   // It doesn't make sense to use the GPU if the polynomial has no terms.
   inline bool canComputeBatchWithCuda() const override { return !_coefList.empty(); }

private:
   ClassDefOverride(RooPolynomial, 1); // Polynomial PDF
};

#endif
