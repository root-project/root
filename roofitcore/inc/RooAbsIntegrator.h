/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooAbsIntegrator.rdl,v 1.4 2001/05/14 22:54:19 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/
#ifndef ROO_ABS_INTEGRATOR
#define ROO_ABS_INTEGRATOR

#include "RooFitCore/RooAbsPdf.hh"


class RooAbsIntegrator : public TObject {
public:

  // Constructors, assignment etc
  inline RooAbsIntegrator() { }
  RooAbsIntegrator(const RooAbsPdf& function, Int_t mode) ;
  RooAbsIntegrator(const RooAbsIntegrator& other);
  virtual ~RooAbsIntegrator();
  virtual Double_t integral()=0 ;

protected:
  
  inline Double_t eval() const { return _function->analyticalIntegral(_mode) ; }

  RooAbsPdf* _function ;
  Int_t           _mode ;

private:
  RooAbsIntegrator& operator=(const RooAbsIntegrator& other) ; // not allowed

protected:
  ClassDef(RooAbsIntegrator,1) // Abstract interface for numerical integrators
};

#endif
