/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitTools
 *    File: $Id: RooVoigtian.rdl,v 1.6 2001/08/23 01:23:35 verkerke Exp $
 * Authors:
 *   TS, Thomas Schietinger, SLAC, schieti@slac.stanford.edu
 * History:
 *   09-Aug-2001 TS Created initial version from RooGaussian
 *   27-Aug-2001 TS Port to RooFitModels/RooFitCore
 *
 * Copyright (C) 2001 Stanford Linear Accelerator Center
 *****************************************************************************/
#ifndef ROO_VOIGTIAN
#define ROO_VOIGTIAN

#include "RooFitCore/RooAbsPdf.hh"
#include "RooFitCore/RooRealProxy.hh"

class RooRealVar;

class RooVoigtian : public RooAbsPdf {
public:
  RooVoigtian(const char *name, const char *title,
	      RooAbsReal& _x, RooAbsReal& _mean, 
              RooAbsReal& _width, RooAbsReal& _sigma);
  RooVoigtian(const RooVoigtian& other, const char* name=0) ;
  virtual TObject* clone(const char* newname) const { return new RooVoigtian(*this,newname); }
  inline virtual ~RooVoigtian() { }

// These methods allow the user to select the fast evaluation
// of the complex error function using look-up tables
// (default is the "slow" CERNlib algorithm)

  inline void selectFastAlgorithm()    { _doFast = kTRUE;  }
  inline void selectDefaultAlgorithm() { _doFast = kFALSE; }

protected:

  RooRealProxy x ;
  RooRealProxy mean ;
  RooRealProxy width ;
  RooRealProxy sigma ;

  Double_t evaluate() const ;

private:

  Double_t _invRootPi;
  Bool_t _doFast;
  ClassDef(RooVoigtian,0) // Voigtian PDF
};

#endif

