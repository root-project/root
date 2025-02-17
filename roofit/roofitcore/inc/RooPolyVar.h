/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooPolyVar.h,v 1.7 2007/05/11 09:11:30 verkerke Exp $
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
#ifndef ROO_POLY_VAR
#define ROO_POLY_VAR

#include <RooAbsReal.h>
#include <RooRealProxy.h>
#include <RooListProxy.h>

#include <vector>

class RooPolyVar : public RooAbsReal {
public:
   RooPolyVar() {}
   RooPolyVar(const char *name, const char *title, RooAbsReal &x);
   RooPolyVar(const char *name, const char *title, RooAbsReal &_x, const RooArgList &_coefList, Int_t lowestOrder = 0);

   RooPolyVar(const RooPolyVar &other, const char *name = nullptr);
   TObject *clone(const char *newname) const override { return new RooPolyVar(*this, newname); }

   Int_t getAnalyticalIntegral(RooArgSet &allVars, RooArgSet &analVars, const char *rangeName = nullptr) const override;
   double analyticalIntegral(Int_t code, const char *rangeName = nullptr) const override;

   RooRealProxy const &x() const { return _x; }
   RooArgList const &coefList() const { return _coefList; }
   int lowestOrder() const { return _lowestOrder; }

protected:
   RooRealProxy _x;
   RooListProxy _coefList;
   Int_t _lowestOrder = 0;

   mutable std::vector<double> _wksp; ///<! do not persist

   double evaluate() const override;
   void doEval(RooFit::EvalContext &) const override;

   // It doesn't make sense to use the GPU if the polynomial has no terms.
   inline bool canComputeBatchWithCuda() const override { return !_coefList.empty(); }

private:
   friend class RooPolynomial;

   static void doEvalImpl(RooAbsArg const* caller, RooFit::EvalContext &,
                          RooAbsReal const &x, RooArgList const &coefs, int lowestOrder);

   static void fillCoeffValues(std::vector<double> &wksp, RooListProxy const &coefList);

   ClassDefOverride(RooPolyVar, 1) // Polynomial function
};

#endif
