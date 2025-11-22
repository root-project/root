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
   /// old constructor
   RooUniform(const char *name, const char *title, const RooArgSet& observables);
   /// Constructor for an N-dimensional uniform PDF with fittable bounds.
   RooUniform(const char *name, const char *title, const RooArgSet& observables, const RooArgSet& lowerBounds, const RooArgSet& upperBounds);

   RooUniform(const RooUniform &other, const char *name = nullptr);
   TObject *clone(const char *newname = nullptr) const override { return new RooUniform(*this, newname); }

   Int_t getAnalyticalIntegral(RooArgSet &allVars, RooArgSet &analVars, const char *rangeName = nullptr) const override;
   double analyticalIntegral(Int_t code, const char *rangeName = nullptr) const override;

   Int_t getGenerator(const RooArgSet &directVars, RooArgSet &generateVars, bool staticInitOK = true) const override;
   void generateEvent(Int_t code) override;

protected:
   RooListProxy _observables; ///< List of observables
   RooListProxy _lowerBounds; ///< List of lower bounds
   RooListProxy _upperBounds; ///< List of upper bounds

   double evaluate() const override;

private:
   ClassDefOverride(RooUniform, 2)
};

#endif
