/*
 * Project: RooFit
 * Author:
 *   Rahul Balasubramanian, Nikhef 01 Apr 2021
 *
 * Copyright (c) 2023, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#ifndef RooFit_RooRatio_h
#define RooFit_RooRatio_h

#include <RooAbsReal.h>
#include <RooRealProxy.h>

class RooArgList;

class RooRatio : public RooAbsReal {
public:
   RooRatio();
   RooRatio(const char *name, const char *title, double numerator, double denominator);
   RooRatio(const char *name, const char *title, double numerator, RooAbsReal &denominator);
   RooRatio(const char *name, const char *title, RooAbsReal &numerator, double denominator);
   RooRatio(const char *name, const char *title, RooAbsReal &numerator, RooAbsReal &denominator);
   RooRatio(const char *name, const char *title, const RooArgList &num, const RooArgList &denom);

   RooRatio(const RooRatio &other, const char *name = nullptr);
   TObject *clone(const char *newname) const override { return new RooRatio(*this, newname); }
   ~RooRatio() override;

   RooAbsReal const &numerator() const { return *_numerator; }
   RooAbsReal const &denominator() const { return *_denominator; }

protected:
   double evaluate() const override;
   void doEval(RooFit::EvalContext &) const override;
   inline bool canComputeBatchWithCuda() const override { return true; }

   RooRealProxy _numerator;
   RooRealProxy _denominator;

   ClassDefOverride(RooRatio, 2) // Ratio of two RooAbsReal and/or numbers
};

#endif
