/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
 *    File: $Id: RooUnblindPrecision.h,v 1.7 2007/05/11 10:15:52 verkerke Exp $
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
#ifndef ROO_UNBLIND_PRECISION
#define ROO_UNBLIND_PRECISION

#include "RooAbsHiddenReal.h"
#include "RooRealProxy.h"
#include "RooCategoryProxy.h"
#include "RooBlindTools.h"

class RooCategory ;

class RooUnblindPrecision : public RooAbsHiddenReal {
public:
  // Constructors, assignment etc
  RooUnblindPrecision() ;
  RooUnblindPrecision(const char *name, const char *title,
            const char *blindString, double centralValue, double scale, RooAbsReal& blindValue, bool sin2betaMode=false);
  RooUnblindPrecision(const char *name, const char *title,
            const char *blindString, double centralValue, double scale,
            RooAbsReal& blindValue, RooAbsCategory& blindState, bool sin2betaMode=false);
  RooUnblindPrecision(const RooUnblindPrecision& other, const char* name=nullptr);
  TObject* clone(const char* newname) const override { return new RooUnblindPrecision(*this,newname); }
  ~RooUnblindPrecision() override;

protected:

  // Function evaluation
  double evaluate() const override ;

  RooRealProxy _value ;          // Holder of the blind value
  RooBlindTools _blindEngine ;   // Blinding engine

  ClassDefOverride(RooUnblindPrecision,1) // Precision unblinding transformation
};

#endif
