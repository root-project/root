/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooIntegrator1D.rdl,v 1.2 2001/04/21 02:42:43 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/
#ifndef ROO_INTEGRATOR_1D
#define ROO_INTEGRATOR_1D

#include "RooFitCore/RooAbsIntegrator.hh"
class RooRealVar ;


class RooIntegrator1D : public RooAbsIntegrator {
public:

  // Constructors, assignment etc
  inline RooIntegrator1D() { }
  RooIntegrator1D(const RooAbsReal& function, Int_t mode, RooRealVar& var, Int_t maxSteps=20, Double_t eps=1e-6) ; 
  RooIntegrator1D(const RooIntegrator1D& other);
  virtual ~RooIntegrator1D();

  virtual Double_t integral() ;

protected:

  void initialize() ;

  // Integrator configuration
  Int_t _maxSteps ;
  Double_t _eps ;
  enum { _nPoints = 5 };

  // Numerical integrator support functions
  Double_t evalAt(Double_t x) const ;
  Double_t addTrapezoids(Int_t n) ;
  void extrapolate(Int_t n) ;
  
  // Numerical integrator workspace
  RooRealVar* _var ;                   
  Double_t _xmin;                      //! do not persist
  Double_t _xmax;                      //! do not persist
  Double_t _range;                     //! do not persist
  Double_t _extrapValue;               //! do not persist
  Double_t _extrapError;               //! do not persist
  Double_t *_h ;                       //! do not persist
  Double_t *_s ;                       //! do not persist
  Double_t *_c ;                       //! do not persist
  Double_t *_d ;                       //! do not persist
  Double_t _savedResult;               //! do not persist

private:
  RooIntegrator1D& operator=(const RooIntegrator1D& other) ;

protected:
  ClassDef(RooIntegrator1D,1) // a real-valued variable and its value
};

#endif
