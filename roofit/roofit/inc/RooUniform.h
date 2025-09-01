/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
 *    File: $Id$
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
#ifndef ROO_UNIFORM
#define ROO_UNIFORM

#include "RooAbsPdf.h"
#include "RooListProxy.h"
#include "RooRealProxy.h"

class RooRealVar;
class RooAbsReal;

class RooUniform : public RooAbsPdf {
public:
   RooUniform() {};
   RooUniform(const char *name, const char *title, const RooArgSet &_x);
   /// Constructor for a 1D uniform PDF with fittable bounds.
   RooUniform(const char *name, const char *title, RooAbsReal &x, RooAbsReal &x_low, RooAbsReal &x_up);
   RooUniform(const RooUniform &other, const char *name = nullptr);
   TObject *clone(const char *newname = nullptr) const override { return new RooUniform(*this, newname); }

   Int_t getAnalyticalIntegral(RooArgSet &allVars, RooArgSet &analVars, const char *rangeName = nullptr) const override;
   double analyticalIntegral(Int_t code, const char *rangeName = nullptr) const override;

   Int_t getGenerator(const RooArgSet &directVars, RooArgSet &generateVars, bool staticInitOK = true) const override;
   void generateEvent(Int_t code) override;

protected:
   RooListProxy x;        ///< List of observables for N-dimensional uniform PDF
   RooRealProxy x_single; ///< Single observable for 1D bounded mode
   RooRealProxy x_low;    ///< Lower bound for 1D bounded mode
   RooRealProxy x_up;     ///< Upper bound for 1D bounded mode

   double evaluate() const override;

private:
   ClassDefOverride(RooUniform, 1) // Flat PDF in N dimensions
};

#endif
