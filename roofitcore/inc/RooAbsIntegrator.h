/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id$                                                             *
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2002, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/
#ifndef ROO_ABS_INTEGRATOR
#define ROO_ABS_INTEGRATOR

#include "RooFitCore/RooAbsFunc.hh"

class RooAbsIntegrator {
public:
  RooAbsIntegrator(const RooAbsFunc& function);
  inline virtual ~RooAbsIntegrator() { }
  inline Bool_t isValid() const { return _valid; }

  inline Double_t integrand(const Double_t x[]) const { return (*_function)(x); }
  inline const RooAbsFunc *integrand() const { return _function; }

  inline virtual Bool_t checkLimits() const { return kTRUE; }
  virtual Double_t integral()=0 ;

protected:
  const RooAbsFunc *_function;
  Bool_t _valid;

  ClassDef(RooAbsIntegrator,0) // Abstract interface for real-valued function integrators
};

#endif
