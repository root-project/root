/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooGaussModel.rdl,v 1.1 2001/06/09 05:14:11 verkerke Exp $
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   05-Jun-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/
#ifndef ROO_GAUSS_MODEL
#define ROO_GAUSS_MODEL

#include "RooFitCore/RooResolutionModel.hh"
#include "RooFitCore/RooRealProxy.hh"
#include "RooFitCore/RooComplex.hh"

class RooGaussModel : public RooResolutionModel {
public:

  enum RooGaussBasis { noBasis=0, expBasisPlus=1, expBasisMinus=2,
                                  sinBasisPlus=3, sinBasisMinus=4,
                                  cosBasisPlus=5, cosBasisMinus=6 } ;

  // Constructors, assignment etc
  inline RooGaussModel() { }
  RooGaussModel(const char *name, const char *title, RooRealVar& x, 
		RooAbsReal& mean, RooAbsReal& sigma) ; 
  RooGaussModel(const RooGaussModel& other, const char* name=0);
  virtual TObject* clone() const { return new RooGaussModel(*this) ; }
  virtual TObject* clone(const char* newname) const { return new RooGaussModel(*this,newname) ; }
  virtual ~RooGaussModel();
  
  virtual Int_t basisCode(const char* name) const ;
  virtual Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars) const ;
  virtual Double_t analyticalIntegral(Int_t code) const ;

protected:
  virtual Double_t evaluate(const RooDataSet* dset) const ;

  RooComplex evalCerf(Double_t swt, Double_t u, Double_t c) const ;

  RooRealProxy mean ;
  RooRealProxy sigma ;

  ClassDef(RooGaussModel,1) // Gaussian Resolution Model
};

#endif
