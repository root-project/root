/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id$
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/
#ifndef ROO_REAL_INTEGRAL
#define ROO_REAL_INTEGRAL

#include "RooFitCore/RooDerivedReal.hh"
#include "RooFitCore/RooArgSet.hh"

class RooArgSet ;
class TH1F ;
class RooAbsCategory ;
class RooRealVar ;

class RooRealIntegral : public RooDerivedReal {
public:

  // Constructors, assignment etc
  inline RooRealIntegral() { }
  RooRealIntegral(const char *name, const char *title, RooDerivedReal& function, 
		  RooArgSet& depList, Int_t maxSteps=20, Double_t eps=1e-6) ;
  RooRealIntegral(const RooRealIntegral& other);
  RooRealIntegral(const char* name, const RooRealIntegral& other);
  RooRealIntegral& operator=(const RooRealIntegral& other) ;
  virtual TObject* Clone() { return new RooRealIntegral(*this); }
  virtual ~RooRealIntegral();

  virtual void printToStream(ostream& stream, PrintOption opt=Standard) const ;

protected:

  RooAbsArg& operator=(const RooAbsArg& other) ;

  virtual Double_t sum(const RooArgSet& sumList, const RooArgSet& intList) const ;
  virtual Double_t integrate(const RooArgSet& intList) const ;
  virtual Bool_t engineInit() ;
  virtual Bool_t engineCleanup() ;

  // Evaluation and validation implementation
  Double_t evaluate() const ;
  virtual Bool_t isValid(Double_t value) const ;

  // Post-processing of server redirection
  virtual Bool_t redirectServersHook(RooArgSet& newServerList, Bool_t mustReplaceAll=kFALSE) ;  

  // Function pointer and integrands list
  RooDerivedReal* _function ;
  mutable RooArgSet _sumList ;
  mutable RooArgSet _intList ;

  // Integrator configuration
  Int_t _mode ;
  Int_t _maxSteps ;
  Double_t _eps ;
  enum { _nPoints = 5 };

  // Numerical integrator support functions
  Double_t evalAt(Double_t x) const ;
  Double_t addTrapezoids(Int_t n) const ;
  void extrapolate(Int_t n) const ;
  
  // Numerical integrator workspace
  mutable RooRealVar* _var ;                   //! do not persist
  mutable Double_t _xmin;                      //! do not persist
  mutable Double_t _xmax;                      //! do not persist
  mutable Double_t _range;                     //! do not persist
  mutable Double_t _extrapValue;               //! do not persist
  mutable Double_t _extrapError;               //! do not persist
  mutable Double_t *_h ;                       //! do not persist
  mutable Double_t *_s ;                       //! do not persist
  mutable Double_t *_c ;                       //! do not persist
  mutable Double_t *_d ;                       //! do not persist
  mutable Double_t _savedResult;               //! do not persist

  ClassDef(RooRealIntegral,1) // a real-valued variable and its value
};

#endif
