/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooAbsIntegrator.rdl,v 1.7 2001/08/08 23:11:23 david Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *   05-Aug-2001 DK Adapted to use RooAbsFunc interface
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/
#ifndef ROO_ABS_INTEGRATOR
#define ROO_ABS_INTEGRATOR

#include "RooFitCore/RooAbsFunc.hh"

class RooAbsFunc;

class RooAbsIntegrator {
public:
  RooAbsIntegrator(const RooAbsFunc& function);
  inline virtual ~RooAbsIntegrator() { }
  inline Bool_t isValid() const { return _valid; }

  inline Double_t integrand(const Double_t x[]) const { return (*_function)(x); }
  inline const RooAbsFunc *integrand() const { return _function; }

  virtual Double_t integral()=0 ;

protected:
  const RooAbsFunc *_function;
  Bool_t _valid;

  ClassDef(RooAbsIntegrator,0) // Abstract interface for real-valued function integrators
};

#endif
