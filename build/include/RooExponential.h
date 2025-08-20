/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
 *    File: $Id: RooExponential.h,v 1.10 2007/07/12 20:30:49 wouter Exp $
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
#ifndef ROO_EXPONENTIAL
#define ROO_EXPONENTIAL

#include <RooAbsPdf.h>
#include <RooRealProxy.h>

class RooExponential : public RooAbsPdf {
public:
   RooExponential() {}
   RooExponential(const char *name, const char *title, RooAbsReal &variable, RooAbsReal &coefficient,
                  bool negateCoefficient = false);
   RooExponential(const RooExponential &other, const char *name = nullptr);
   TObject *clone(const char *newname) const override { return new RooExponential(*this, newname); }

   Int_t getAnalyticalIntegral(RooArgSet &allVars, RooArgSet &analVars, const char *rangeName = nullptr) const override;
   double analyticalIntegral(Int_t code, const char *rangeName = nullptr) const override;

   /// Get the x variable.
   RooAbsReal const &variable() const { return x.arg(); }

   /// Get the coefficient "c".
   RooAbsReal const &coefficient() const { return c.arg(); }

   bool negateCoefficient() const { return _negateCoefficient; }

protected:
   RooRealProxy x;
   RooRealProxy c;
   bool _negateCoefficient = false;

   double evaluate() const override;
   void doEval(RooFit::EvalContext &) const override;
   inline bool canComputeBatchWithCuda() const override { return true; }

private:
   ClassDefOverride(RooExponential, 2) // Exponential PDF
};

#endif
