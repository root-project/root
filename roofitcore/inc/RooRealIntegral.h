/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooRealIntegral.rdl,v 1.1 2001/04/08 00:06:49 verkerke Exp $
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
class RooAbsIntegrator ;

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

  Bool_t _init ;
  void deferredInit() ;
  void initNumIntegrator() ;
  RooAbsArg& operator=(const RooAbsArg& other) ;

  virtual Double_t sum() const ;
  virtual Double_t integrate() const ;

  // Evaluation and validation implementation
  Double_t evaluate() const ;
  virtual Bool_t isValid(Double_t value) const ;

  // Post-processing of server redirection
  virtual Bool_t redirectServersHook(RooArgSet& newServerList, Bool_t mustReplaceAll=kFALSE) ;  

  // Function pointer and integrands list
  RooDerivedReal* _function ;
  mutable RooArgSet _depList ;
  mutable RooArgSet _sumList ;
  mutable RooArgSet _intList ;
  Int_t _mode ;

  RooAbsIntegrator* _numIntEngine ;

  ClassDef(RooRealIntegral,1) // a real-valued variable and its value
};

#endif
