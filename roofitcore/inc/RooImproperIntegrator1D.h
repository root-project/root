/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooIntegrator1D.rdl,v 1.6 2001/08/02 23:54:24 david Exp $
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

class RooImproperIntegrator1D : public RooAbsIntegrator {
public:

  RooImproperIntegrator1D(const RooAbsFunc& function);
  virtual ~RooImproperIntegrator1D();

  virtual Double_t integral() ;

protected:

  RooInvTransform *_function;
  RooIntegrator1D *_integrator1,*_integrator2,*_integrator3;

  ClassDef(RooImproperIntegrator1D,0) // 1-dimensional improper integration engine
};

#endif
