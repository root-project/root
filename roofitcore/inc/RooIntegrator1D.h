/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooIntegrator1D.rdl,v 1.8 2001/08/24 23:55:15 david Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *   05-Aug-2001 DK Adapted to use RooAbsFunc interface
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/
#ifndef ROO_INTEGRATOR_1D
#define ROO_INTEGRATOR_1D

#include "RooFitCore/RooAbsIntegrator.hh"

class RooIntegrator1D : public RooAbsIntegrator {
public:

  // Constructors, assignment etc
  enum SummationRule { Trapezoid, Midpoint };
  RooIntegrator1D(const RooAbsFunc& function, SummationRule rule= Trapezoid,
		  Int_t maxSteps= 0, Double_t eps= 0) ; 
  RooIntegrator1D(const RooAbsFunc& function, Double_t xmin, Double_t xmax,
		  SummationRule rule= Trapezoid, Int_t maxSteps= 0, Double_t eps= 0) ; 
  virtual ~RooIntegrator1D();

  virtual Bool_t checkLimits() const;
  virtual Double_t integral() ;

  Bool_t setLimits(Double_t xmin, Double_t xmax);

protected:

  Bool_t initialize();

  Bool_t _useIntegrandLimits;

  // Integrator configuration
  SummationRule _rule;
  Int_t _maxSteps ;
  Double_t _eps ;
  enum { _nPoints = 5 };

  // Numerical integrator support functions
  Double_t addTrapezoids(Int_t n) ;
  Double_t addMidpoints(Int_t n) ;
  void extrapolate(Int_t n) ;
  
  // Numerical integrator workspace
  mutable Double_t _xmin;              //! do not persist
  mutable Double_t _xmax;              //! do not persist
  mutable Double_t _range;             //! do not persist
  Double_t _extrapValue;               //! do not persist
  Double_t _extrapError;               //! do not persist
  Double_t *_h ;                       //! do not persist
  Double_t *_s ;                       //! do not persist
  Double_t *_c ;                       //! do not persist
  Double_t *_d ;                       //! do not persist
  Double_t _savedResult;               //! do not persist

  ClassDef(RooIntegrator1D,0) // 1-dimensional numerical integration engine
};

#endif
