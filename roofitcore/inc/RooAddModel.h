/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooAddModel.rdl,v 1.4 2001/08/01 01:24:08 verkerke Exp $
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
  RooAddModel(const char *name, const char *title, RooResolutionModel& model1) ;
  RooAddModel(const char *name, const char *title,
	    RooResolutionModel& model1, RooResolutionModel& model2, RooAbsReal& coef1) ;
  RooAddModel(const char *name, const char *title,
	    RooResolutionModel& model1, RooResolutionModel& model2, RooResolutionModel& model3, 
            RooAbsReal& coef1, RooAbsReal& coef2) ;
  RooAddModel(const RooAddModel& other, const char* name=0) ;
  virtual TObject* clone(const char* newname) const { return new RooAddModel(*this,newname) ; }
  virtual ~RooAddModel() ;


  virtual RooResolutionModel* convolution(RooFormulaVar* basis, RooAbsArg* owner) const ;

  void addModel(RooResolutionModel& model, RooAbsReal& coef) ;
  void addLastModel(RooResolutionModel& model) ;

  virtual Double_t evaluate(const RooArgSet* nset) const ;
  virtual Double_t getNorm(const RooArgSet* nset=0) const ;
  virtual Bool_t checkDependents(const RooArgSet* nset) const ;	

  virtual Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& numVars) const ;
  virtual Double_t analyticalIntegral(Int_t code) const ;
  virtual Int_t basisCode(const char* name) const ;

  virtual Bool_t selfNormalized() const { return kTRUE ; }

  virtual void syncNormalization(const RooArgSet* nset) const ;
  virtual void normLeafServerList(RooArgSet& list) const ;


protected:

  Bool_t _isCopy ;
  TList _modelProxyList ;
  TList _coefProxyList ;
  TIterator* _modelProxyIter ;
  TIterator* _coefProxyIter ;

private:

  ClassDef(RooAddModel,0) // Resolution model consisting of a sum of resolution models
};

#endif
