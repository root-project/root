/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id$
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   19-Jun-2001 WV Initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

#ifndef ROO_ADD_MODEL
#define ROO_ADD_MODEL

#include "RooFitCore/RooResolutionModel.hh"
#include "TList.h"

class RooAddModel : public RooResolutionModel {
public:
  RooAddModel(const char *name, const char *title, RooRealVar& convVar);
  RooAddModel(const char *name, const char *title,
	    RooResolutionModel& model1, RooResolutionModel& model2, RooAbsReal& coef1) ;
  RooAddModel(const char *name, const char *title,
	    RooResolutionModel& model1, RooResolutionModel& model2, RooResolutionModel& model3, 
            RooAbsReal& coef1, RooAbsReal& coef2) ;
  RooAddModel(const RooAddModel& other, const char* name=0) ;
  virtual TObject* clone(const char* newname) const { return new RooAddModel(*this,newname) ; }
  virtual TObject* clone() const { return new RooAddModel(*this) ; }
  virtual ~RooAddModel() ;


  RooResolutionModel* convolution(RooFormulaVar* basis) const ;

  void addModel(RooResolutionModel& model, RooAbsReal& coef) ;
  void addLastModel(RooResolutionModel& model) ;

  virtual Double_t evaluate(const RooDataSet* dset) const ;
  virtual Double_t getNorm(const RooDataSet* dset=0) const ;
  virtual Bool_t checkDependents(const RooDataSet* set) const ;	

  virtual Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& numVars) const ;
  virtual Double_t analyticalIntegral(Int_t code) const ;
  virtual Int_t basisCode(const char* name) const ;

  Bool_t selfNormalized(const RooArgSet& dependents) const { return kTRUE ; }

protected:

  TList _modelProxyList ;
  TList _coefProxyList ;

private:

  ClassDef(RooAddModel,0) // Resolution model consisting of a sum of resolution models
};

#endif
