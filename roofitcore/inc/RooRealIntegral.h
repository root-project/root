/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooRealIntegral.rdl,v 1.17 2001/08/03 02:04:33 verkerke Exp $
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

#include "RooFitCore/RooAbsReal.hh"
#include "RooFitCore/RooArgSet.hh"
#include "RooFitCore/RooAbsPdf.hh"
#include "RooFitCore/RooRealProxy.hh"
#include "RooFitCore/RooSetProxy.hh"

class RooArgSet ;
class TH1F ;
class RooAbsCategory ;
class RooRealVar ;
class RooAbsIntegrator ;

class RooRealIntegral : public RooAbsReal {
public:

  // Constructors, assignment etc
  inline RooRealIntegral() : _numIntEngine(0),_numIntegrand(0),_valid(kFALSE) { }
  RooRealIntegral(const char *name, const char *title, const RooAbsReal& function, RooArgSet& depList) ;
  RooRealIntegral(const RooRealIntegral& other, const char* name=0);
  virtual TObject* clone(const char* newname) const { return new RooRealIntegral(*this,newname); }
  virtual ~RooRealIntegral();

  Bool_t isValid() const { return _valid; }

  virtual void printToStream(ostream& stream, PrintOption opt=Standard, TString indent="") const ;

protected:

  mutable Bool_t _valid;

  enum OperMode { Hybrid, Analytic, PassThrough } ;
  //friend class RooAbsPdf ;

  Bool_t initNumIntegrator() const;

  virtual Double_t sum() const ;
  virtual Double_t integrate() const ;
  virtual Double_t jacobianProduct() const ;

  // Evaluation and validation implementation
  Double_t evaluate(const RooArgSet* nset) const ;
  virtual Bool_t isValidReal(Double_t value, Bool_t printError=kFALSE) const ;

  // Post-processing of server redirection
  virtual Bool_t redirectServersHook(const RooArgSet& newServerList, Bool_t mustReplaceAll=kFALSE) ;  

  // Function pointer and integrands list
  RooRealProxy       _function ;
  mutable RooSetProxy _sumList ;
  mutable RooSetProxy _intList ;
  mutable RooSetProxy _anaList ;
  mutable RooSetProxy _jacList ;
  mutable RooSetProxy _facList ;

  Int_t _mode ;
  OperMode _operMode ;

  mutable RooAbsIntegrator* _numIntEngine ;  //! do not persist
  mutable RooAbsFunc *_numIntegrand;         //! do not persist


  ClassDef(RooRealIntegral,1) // Real-valued variable representing an integral over a RooAbsReal object
};

#endif
