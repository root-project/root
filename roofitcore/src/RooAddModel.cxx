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

// -- CLASS DESCRIPTION --

#include "TIterator.h"
#include "TList.h"
#include "RooFitCore/RooAddModel.hh"
#include "RooFitCore/RooRealProxy.hh"

ClassImp(RooAddModel)
;


RooAddModel::RooAddModel(const char *name, const char *title, RooRealVar& convVar) :
  RooResolutionModel(name,title,convVar)
{
  // Dummy constructor 
}


RooAddModel::RooAddModel(const char *name, const char *title,
			 RooResolutionModel& model1, 
			 RooResolutionModel& model2, 
			 RooAbsReal& coef1) : 
  RooResolutionModel(name,title,model1.convVar())
{
  // Check for consistency of convolution variables
  if (&model1.convVar() != &model2.convVar()) {
    cout << "RooAddModel::RooAddModel(" << GetName() << "): ERROR, input models " 
	 << model1.GetName() << " and " << model2.GetName() 
	 << " do not have the same convolution variable" << endl ;
    assert(0) ;
  }

  // Constructor with two models
  addModel(model1,coef1) ;
  addLastModel(model2) ;    

}


RooAddModel::RooAddModel(const char *name, const char *title,
			 RooResolutionModel& model1, 
			 RooResolutionModel& model2, 
			 RooResolutionModel& model3, 
			 RooAbsReal& coef1,
			 RooAbsReal& coef2) : 
  RooResolutionModel(name,title,model1.convVar())
{
  // Check for consistency of convolution variables
  if (&model1.convVar() != &model2.convVar()) {
    cout << "RooAddModel::RooAddModel(" << GetName() << "): ERROR, input models " 
	 << model1.GetName() << " and " << model2.GetName() 
	 << " do not have the same convolution variable" << endl ;
    assert(0) ;
  }

  if (&model2.convVar() != &model3.convVar()) {
    cout << "RooAddModel::RooAddModel(" << GetName() << "): ERROR, input models " 
	 << model1.GetName() << " and " << model3.GetName() 
	 << " do not have the same convolution variable" << endl ;
    assert(0) ;
  }

  // Constructor with 3 models
  addModel(model1,coef1) ;
  addModel(model2,coef2) ;
  addLastModel(model3) ;    

}


RooAddModel::RooAddModel(const RooAddModel& other, const char* name) :
  RooResolutionModel(other,name)
{
  // Copy constructor

  // If we own the components convolutions we should clone them here

  // Copy proxy lists
  TIterator *iter = other._coefProxyList.MakeIterator() ;
  RooRealProxy* proxy ;
  while(proxy=(RooRealProxy*)iter->Next()) {
    _coefProxyList.Add(new RooRealProxy("coef",this,*proxy)) ;
  }
  delete iter ;

  iter = other._modelProxyList.MakeIterator() ;
  while(proxy=(RooRealProxy*)iter->Next()) {
    if (_basis) {
      removeServer(proxy->arg()) ;
      _modelProxyList.Add(new RooRealProxy("model","model",this,*(RooResolutionModel*)(proxy->arg().Clone()) )) ;
    } else {
      _modelProxyList.Add(new RooRealProxy("model",this,*proxy)) ;
    }
  }
  delete iter ;
}


RooAddModel::~RooAddModel()
{
  // Destructor

  // If we are a convolution object, we own the component convolutions
  TList ownedList ;
  if (_basis) {
    TIterator* mIter = _modelProxyList.MakeIterator() ;
    RooRealProxy* modelProxy ;
    while (modelProxy=((RooRealProxy*)mIter->Next())) {
      ownedList.Add(modelProxy->absArg()) ;
    }
  }
  
  // Delete all owned proxies 
  _coefProxyList.Delete() ;
  _modelProxyList.Delete() ;

  // Delete owned objects only after referring proxies have been deleted
  if (_basis) {
    ownedList.Delete() ;
  }
}



void RooAddModel::addModel(RooResolutionModel& model, RooAbsReal& coef) 
{  
  // Add a model/coefficient pair to the model sum

  RooRealProxy *modelProxy = new RooRealProxy("model","model",this,model) ;
  RooRealProxy *coefProxy = new RooRealProxy("coef","coef",this,coef) ;
  
  _modelProxyList.Add(modelProxy) ;
  _coefProxyList.Add(coefProxy) ;
}


void RooAddModel::addLastModel(RooResolutionModel& model) 
{
  // Specify the last model, whose coefficient is automatically 
  // calculated from the normalization requirement
  RooRealProxy *modelProxy = new RooRealProxy("model","model",this,model) ;
  _modelProxyList.Add(modelProxy) ;
}


RooResolutionModel* RooAddModel::convolution(RooFormulaVar* basis) const
{
  // Check that primary variable of basis functions is our convolution variable  
  if (basis->findServer(0) != x.absArg()) {
    cout << "RooResolutionModel::convolution(" << GetName() 
	 << " convolution parameter of basis function and PDF don't match" << endl ;
    return 0 ;
  }

  TString newName(GetName()) ;
  newName.Append("_conv_") ;
  newName.Append(basis->GetName()) ;

  TString newTitle(GetTitle()) ;
  newTitle.Append(" convoluted with basis function ") ;
  newTitle.Append(basis->GetName()) ;

  RooAddModel* convSum = new RooAddModel(newName,newTitle,convVar()) ;

  TIterator *mIter = _modelProxyList.MakeIterator() ;
  TIterator *cIter = _coefProxyList.MakeIterator() ;
  RooRealProxy* coef ;
  RooRealProxy* model ;
  while(coef=(RooRealProxy*)cIter->Next()) {
    model = (RooRealProxy*)mIter->Next() ;
    
    // Create component convolution
    RooResolutionModel* conv = ((RooResolutionModel*)(model->absArg()))->convolution(basis) ;    
    convSum->addModel(*conv,(RooAbsReal&)coef->arg()) ;
  }

  // Create last component convolution
  model = (RooRealProxy*)mIter->Next() ;    
  RooResolutionModel* conv = ((RooResolutionModel*)(model->absArg()))->convolution(basis) ;    
  convSum->addLastModel(*conv) ;
  
  convSum->changeBasis(basis) ;
  return convSum ;
}



Int_t RooAddModel::basisCode(const char* name) const 
{
  // Assign code of first component, or return 0 if any of the components return 0
  TIterator* mIter = _modelProxyList.MakeIterator() ;
  RooRealProxy* model ;
  Bool_t first(kTRUE), code(0) ;
    while(model = (RooRealProxy*)mIter->Next()) {
      Int_t subCode = ((RooResolutionModel&)model->arg()).basisCode(name) ;
      if (first) {
	code = subCode ;
      } else if (subCode==0) {
	code = 0 ;
      }
  }
  delete mIter ;

  return code ;
}



Double_t RooAddModel::evaluate(const RooDataSet* dset) const 
{
  // Calculate the current value of this object
  TIterator *pIter = _modelProxyList.MakeIterator() ;
  TIterator *cIter = _coefProxyList.MakeIterator() ;
  
  Double_t value(0) ;
  Double_t lastCoef(1) ;

  // Do running sum of coef/model pairs, calculate lastCoef.
  RooRealProxy* coef ;
  RooRealProxy* model ;
  while(coef=(RooRealProxy*)cIter->Next()) {
    model = (RooRealProxy*)pIter->Next() ;
    value += (*model)*(*coef) ;
    lastCoef -= (*coef) ;
  }

  // Add last model with correct coefficient
  model = (RooRealProxy*) pIter->Next() ;
  value += (*model)*lastCoef ;

  // Warn about coefficient degeneration
  if (lastCoef<0 || lastCoef>1) {
    cout << "RooAddModel::evaluate(" << GetName() 
	 << " WARNING: sum of model coefficients not in range [0-1], value=" 
	 << 1-lastCoef << endl ;
  } 

  delete pIter ;
  delete cIter ;

  return value ;
}


Double_t RooAddModel::getNorm(const RooDataSet* dset) const
{
  // Operate as regular PDF if we have no basis function
  if (!_basis) return RooAbsPdf::getNorm(dset) ;

  // Return sum of component normalizations
  TIterator *pIter = _modelProxyList.MakeIterator() ;
  TIterator *cIter = _coefProxyList.MakeIterator() ;

  Double_t norm(0) ;
  Double_t lastCoef(1) ;

  // Do running norm of coef/model pairs, calculate lastCoef.
  RooRealProxy* coef ;
  RooResolutionModel* model ;
  while(coef=(RooRealProxy*)cIter->Next()) {
    model = (RooResolutionModel*)((RooRealProxy*)pIter->Next())->absArg() ;
    norm += model->getNorm(dset)*(*coef) ;
    lastCoef -= (*coef) ;
  }

  // Add last model with correct coefficient
  model = (RooResolutionModel*)((RooRealProxy*)pIter->Next())->absArg() ;
  norm += model->getNorm(dset)*lastCoef ;

  // Warn about coefficient degeneration
  if (lastCoef<0 || lastCoef>1) {
    cout << "RooAddModel::evaluate(" << GetName() 
	 << " WARNING: sum of model coefficients not in range [0-1], value=" 
	 << 1-lastCoef << endl ;
  } 

  delete pIter ;
  delete cIter ;

  return norm ;
}



Bool_t RooAddModel::checkDependents(const RooDataSet* set) const 
{
  // Check if model is valid with dependent configuration given by specified data set

  // Special, more lenient dependent checking: Coeffient and model should
  // be non-overlapping, but coef/model pairs can
  Bool_t ret(kFALSE) ;

  TIterator *pIter = _modelProxyList.MakeIterator() ;
  TIterator *cIter = _coefProxyList.MakeIterator() ;

  RooRealProxy* coef ;
  RooRealProxy* model ;
  while(coef=(RooRealProxy*)cIter->Next()) {
    model = (RooRealProxy*)pIter->Next() ;
    ret |= model->arg().checkDependents(set) ;
    ret |= coef->arg().checkDependents(set) ;
    if (model->arg().dependentOverlaps(set,coef->arg())) {
      cout << "RooAddModel::checkDependents(" << GetName() << "): ERROR: coefficient " << coef->arg().GetName() 
	   << " and model " << model->arg().GetName() << " have one or more dependents in common" << endl ;
      ret = kTRUE ;
    }
  }
  
  
  delete pIter ;
  delete cIter ;

  return ret ;
}


Int_t RooAddModel::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& numVars) const 
{
  // Determine which part (if any) of given integral can be performed analytically.
  // If any analytical integration is possible, return integration scenario code

  // This model is by construction normalized
  return 0 ;
}


Double_t RooAddModel::analyticalIntegral(Int_t code) const 
{
  // Return analytical integral defined by given scenario code

  // This model is by construction normalized
  return 1.0 ;
}
