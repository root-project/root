/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooImproperIntegrator1D.rdl,v 1.7 2003/05/09 20:48:23 wverkerke Exp $
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2004, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/
#ifndef ROO_IMPROPER_INTEGRATOR_1D
#define ROO_IMPROPER_INTEGRATOR_1D

#include "RooFitCore/RooAbsIntegrator.hh"

class RooInvTransform;
class RooIntegrator1D;
class RooIntegratorConfig ;

class RooImproperIntegrator1D : public RooAbsIntegrator {
public:

  RooImproperIntegrator1D(const RooAbsFunc& function);
  RooImproperIntegrator1D(const RooAbsFunc& function, const RooIntegratorConfig& config);
  virtual ~RooImproperIntegrator1D();

  virtual Bool_t checkLimits() const;
  virtual Double_t integral(const Double_t* yvec=0) ;

protected:

  void initialize(const RooAbsFunc& function) ;

  enum LimitsCase { Invalid, ClosedBothEnds, OpenBothEnds, OpenBelowSpansZero, OpenBelow,
		    OpenAboveSpansZero, OpenAbove };
  LimitsCase limitsCase() const;
  LimitsCase _case;
  mutable Double_t _xmin, _xmax;

  RooInvTransform *_function;
  mutable RooIntegrator1D *_integrator1,*_integrator2,*_integrator3;
  
  ClassDef(RooImproperIntegrator1D,0) // 1-dimensional improper integration engine
};

#endif
