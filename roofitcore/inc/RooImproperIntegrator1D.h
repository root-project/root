/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooImproperIntegrator1D.rdl,v 1.3 2001/09/15 00:26:02 david Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Aug-2001 DK Created initial version
 *
 * Copyright (C) 2001 University of California
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
  virtual Double_t integral() ;

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
