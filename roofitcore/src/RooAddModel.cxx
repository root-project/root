/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooAddModel.cc,v 1.5 2001/08/09 01:02:13 verkerke Exp $
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
  RooResolutionModel(name,title,convVar), 
  _modelProxyIter(_modelProxyList.MakeIterator()),
  _coefProxyIter(_coefProxyList.MakeIterator()),
  _isCopy(kFALSE) 
{
  // Dummy constructor 
}


RooAddModel::RooAddModel(const char *name, const char *title,
			 RooResolutionModel& model1) :
  RooResolutionModel(name,title,model1.convVar()),
  _modelProxyIter(_modelProxyList.MakeIterator()),
  _coefProxyIter(_coefProxyList.MakeIterator()),
  _isCopy(kFALSE) 
{
  // Constructor with one model
  addLastModel(model1) ;    

}


RooAddModel::RooAddModel(const char *name, const char *title,
			 RooResolutionModel& model1, 
			 RooResolutionModel& model2, 
			 RooAbsReal& coef1) : 
  RooResolutionModel(name,title,model1.convVar()),
  _modelProxyIter(_modelProxyList.MakeIterator()),
  _coefProxyIter(_coefProxyList.MakeIterator()),
  _isCopy(kFALSE)
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
  RooResolutionModel(name,title,model1.convVar()),
  _modelProxyIter(_modelProxyList.MakeIterator()),
  _coefProxyIter(_coefProxyList.MakeIterator()),
  _isCopy(kFALSE) 
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
  RooResolutionModel(other,name),
  _modelProxyIter(_modelProxyList.MakeIterator()),
  _coefProxyIter(_coefProxyList.MakeIterator()),
  _isCopy(kTRUE) 
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
//     if (_basis) {
//       removeServer(*proxy->absArg()) ;
//       _modelProxyList.Add(new RooRealProxy("model","model",this,*(RooResolutionModel*)(proxy->arg().Clone()) )) ;
//     } else {
      _modelProxyList.Add(new RooRealProxy("model",this,*proxy)) ;
//     }
  }
  delete iter ;
}


RooAddModel::~RooAddModel()
{
  // Destructor
  cout << "RooAddModel::dtor(" << GetName() << "," << this << ") _basis= " << _basis << ", isCopy = " << (_isCopy?"T":"F") << endl ;

  // If we are a non-copied convolution object, we own the component convolutions
  TList ownedList ;
  if (_basis && !_isCopy) {
    TIterator* mIter = _modelProxyList.MakeIterator() ;
    RooRealProxy* modelProxy ;
    while (modelProxy=((RooRealProxy*)mIter->Next())) {
      ownedList.Add(modelProxy->absArg()) ;
    }
  }

  delete _coefProxyIter ;
  delete _modelProxyIter ;
  
  // Delete all owned proxies 
  _coefProxyList.Delete() ;
  _modelProxyList.Delete() ;
  


  // Delete owned objects only after referring proxies have been deleted
  if (_basis && !_isCopy) {
    cout << "RooAddModel::dtor(" << GetName() << ") deleting owned components" << endl ;
    ownedList.Print() ;
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


RooResolutionModel* RooAddModel::convolution(RooFormulaVar* basis, RooAbsArg* owner) const
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
  newName.Append("_[") ;
  newName.Append(owner->GetName()) ;
  newName.Append("]") ;

  TString newTitle(GetTitle()) ;
  newTitle.Append(" convoluted with basis function ") ;
  newTitle.Append(basis->GetName()) ;

  RooAddModel* convSum = new RooAddModel(newName,newTitle,convVar()) ;

  _coefProxyIter->Reset() ;
  _modelProxyIter->Reset() ;
  RooRealProxy* coef ;
  RooRealProxy* model ;
  while(coef=(RooRealProxy*)_coefProxyIter->Next()) {
    model = (RooRealProxy*)_modelProxyIter->Next() ;
    
    // Create component convolution
    RooResolutionModel* conv = ((RooResolutionModel*)(model->absArg()))->convolution(basis,owner) ;    
    convSum->addModel(*conv,(RooAbsReal&)coef->arg()) ;
  }
  
  // Create last component convolution
  model = (RooRealProxy*)_modelProxyIter->Next() ;    
  RooResolutionModel* conv = ((RooResolutionModel*)(model->absArg()))->convolution(basis,owner) ;    
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



Double_t RooAddModel::evaluate(const RooArgSet* nset) const 
{
  // Calculate the current value of this object
  _coefProxyIter->Reset() ;
  _modelProxyIter->Reset() ;
  
  Double_t value(0) ;
  Double_t lastCoef(1) ;

  // Do running sum of coef/model pairs, calculate lastCoef.
  RooRealProxy* coef ;
  RooRealProxy* model ;
  while(coef=(RooRealProxy*)_coefProxyIter->Next()) {
    model = (RooRealProxy*)_modelProxyIter->Next() ;
    value += (*model)*(*coef) ;
    lastCoef -= (*coef) ;
  }

  // Add last model with correct coefficient
  model = (RooRealProxy*) _modelProxyIter->Next() ;
  value += (*model)*lastCoef ;


  // Warn about coefficient degeneration
  if ((lastCoef<0.0 || lastCoef>1.0) && ++_errorCount<=10) {
    cout << "RooAddModel::evaluate(" << GetName() 
	 << " WARNING: sum of model coefficients not in range [0-1], value=" 
	 << 1-lastCoef << endl ;
    if(_errorCount == 10) cout << "(no more will be printed) ";
  } 

//   cout << "RooAddModel::evaluate(" << GetName() << "): result = " << value << endl ;
  return value ;
}


Double_t RooAddModel::getNorm(const RooArgSet* nset) const
{
  // Operate as regular PDF if we have no basis function
  if (!_basis) return RooAbsPdf::getNorm(nset) ;

  // Return sum of component normalizations
  _coefProxyIter->Reset() ;
  _modelProxyIter->Reset() ;

  Double_t norm(0) ;
  Double_t lastCoef(1) ;

  // Do running norm of coef/model pairs, calculate lastCoef.
  RooRealProxy* coef ;
  RooResolutionModel* model ;
  while(coef=(RooRealProxy*)_coefProxyIter->Next()) {
    model = (RooResolutionModel*)((RooRealProxy*)_modelProxyIter->Next())->absArg() ;
    if (_verboseEval>1) cout << "RooAddModel::getNorm(" << GetName() << "): norm x coef = " 
			     << model->getNorm(nset) << " x " << (*coef) << " = " 
			     << model->getNorm(nset)*(*coef) << endl ;

    norm += model->getNorm(nset)*(*coef) ;
    lastCoef -= (*coef) ;
  }

  // Add last model with correct coefficient
  model = (RooResolutionModel*)((RooRealProxy*)_modelProxyIter->Next())->absArg() ;
  norm += model->getNorm(nset)*lastCoef ;
  if (_verboseEval>1) cout << "RooAddModel::getNorm(" << GetName() << "): norm x coef = " 
			   << model->getNorm(nset) << " x " << lastCoef << " = " 
			   << model->getNorm(nset)*lastCoef << endl ;

  // Warn about coefficient degeneration
  if (lastCoef<0 || lastCoef>1) {
    cout << "RooAddModel::evaluate(" << GetName() 
	 << " WARNING: sum of model coefficients not in range [0-1], value=" 
	 << 1-lastCoef << endl ;
  } 

//   cout << "RooAddModel::getNorm(" << GetName() << ") result = " << norm << endl ;
  return norm ;
}



Bool_t RooAddModel::checkDependents(const RooArgSet* set) const 
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


void RooAddModel::normLeafServerList(RooArgSet& list) const 
{
  // Fill list with leaf server nodes of normalization integral 

  TIterator *pIter = _modelProxyList.MakeIterator() ;
  RooRealProxy* proxy ;
  RooResolutionModel* model ;
  while(proxy = (RooRealProxy*) pIter->Next()) {
    model = (RooResolutionModel*) proxy->absArg() ;
    model->_norm->leafNodeServerList(&list) ;
  }
  delete pIter ;
}

void RooAddModel::syncNormalization(const RooArgSet* nset) const 
{
  // Fan out syncNormalization call to components
  if (_verboseEval>0) cout << "RooAddModel:syncNormalization(" << GetName() 
			 << ") forwarding sync request to components (" 
			 << _lastNormSet << " -> " << nset << ")" << endl ;

  // Update dataset pointers of proxies
  ((RooAbsPdf*) this)->setProxyNormSet(nset) ;

  TIterator *pIter = _modelProxyList.MakeIterator() ;
  RooRealProxy* proxy ;
  RooResolutionModel* model ;
  while(proxy = (RooRealProxy*)pIter->Next()) {
    model = (RooResolutionModel*) proxy->absArg() ;
    model->syncNormalization(nset) ;
  }
  delete pIter ;
  
}
