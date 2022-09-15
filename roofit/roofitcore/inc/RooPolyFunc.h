/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
 *    File: $Id: RooPolyFunc.h,v 1.8 2007/05/11 09:13:07 verkerke Exp $
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2021:                                                       *
 *      CERN, Switzerland                                                    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/
#ifndef RooFit_RooFit_RooPolyFunc_h
#define RooFit_RooFit_RooPolyFunc_h

#include "RooAbsReal.h"
#include "RooListProxy.h"

#include <vector>

class RooRealVar;
class RooArgList;

class RooPolyFunc : public RooAbsReal {
public:
   RooPolyFunc();
   RooPolyFunc(const char *name, const char *title, RooAbsReal &x, const RooAbsCollection &coefList);
   RooPolyFunc(const char *name, const char *title, RooAbsReal &x, RooAbsReal &y, const RooAbsCollection &coefList);
   RooPolyFunc(const char *name, const char *title, const RooAbsCollection &vars);
   RooPolyFunc(const RooPolyFunc &other, const char *name = nullptr);
   RooPolyFunc &operator=(const RooPolyFunc &other) = delete;
   RooPolyFunc &operator=(RooPolyFunc &&other) = delete;
   TObject *clone(const char *newname) const override { return new RooPolyFunc(*this, newname); }

   void addTerm(double coefficient);
   void addTerm(double coefficient, const RooAbsCollection &exponents);
   void addTerm(double coefficient, const RooAbsReal &var1, int exp1);
   void addTerm(double coefficient, const RooAbsReal &var1, int exp1, const RooAbsReal &var2, int exp2);

   static std::unique_ptr<RooAbsReal>
   taylorExpand(const char *name, const char *title, RooAbsReal &func, const RooAbsCollection &observables,
                std::vector<double> const &observableValues, int order = 1, double eps1 = 1e-6, double eps2 = 1e-3);
   static std::unique_ptr<RooAbsReal> taylorExpand(const char *name, const char *title, RooAbsReal &func,
                                                   const RooAbsCollection &observables, double observablesValue = 0.0,
                                                   int order = 1, double eps = 1e-6, double eps2 = 1e-3);

protected:
   void setCoordinate(const RooAbsCollection &observables, std::vector<double> const &observableValues);
   RooListProxy _vars;
   std::vector<std::unique_ptr<RooListProxy>> _terms;

   /// Evaluation
   double evaluate() const override;

   ClassDefOverride(RooPolyFunc, 1) // Polynomial Function
};

#endif
