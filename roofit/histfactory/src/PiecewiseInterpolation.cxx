/*****************************************************************************

 *****************************************************************************/

//////////////////////////////////////////////////////////////////////////////
// 
// BEGIN_HTML
// PiecewiseInterpolation 
// END_HTML
//


#include "RooFit.h"

#include "Riostream.h"
#include "Riostream.h"
#include <math.h>

#include "RooStats/HistFactory/PiecewiseInterpolation.h"
#include "RooAbsReal.h"
#include "RooAbsPdf.h"
#include "RooErrorHandler.h"
#include "RooArgSet.h"
#include "RooNLLVar.h"
#include "RooChi2Var.h"
#include "RooMsgService.h"

ClassImp(PiecewiseInterpolation)
;


//_____________________________________________________________________________
PiecewiseInterpolation::PiecewiseInterpolation()
{
  _lowIter = _lowSet.createIterator() ;
  _highIter = _highSet.createIterator() ;
  _paramIter = _paramSet.createIterator() ;
  _positiveDefinite=false;
}



//_____________________________________________________________________________
PiecewiseInterpolation::PiecewiseInterpolation(const char* name, const char* title, const RooAbsReal& nominal,
					       const RooArgList& lowSet, 
					       const RooArgList& highSet,
					       const RooArgList& paramSet,
					       Bool_t takeOwnership) :
  RooAbsReal(name, title),
  _nominal("!nominal","nominal value", this, (RooAbsReal&)nominal),
  _lowSet("!lowSet","low-side variation",this),
  _highSet("!highSet","high-side variation",this),
  _paramSet("!paramSet","high-side variation",this),
  _positiveDefinite(false)
{
  // Constructor with two set of RooAbsReals. The value of the function will be
  //
  //  A = sum_i lowSet(i)*highSet(i) 
  //
  // If takeOwnership is true the PiecewiseInterpolation object will take ownership of the arguments in sumSet

  _lowIter = _lowSet.createIterator() ;
  _highIter = _highSet.createIterator() ;
  _paramIter = _paramSet.createIterator() ;

  // KC: check both sizes
  if (lowSet.getSize() != highSet.getSize()) {
    coutE(InputArguments) << "PiecewiseInterpolation::ctor(" << GetName() << ") ERROR: input lists should be of equal length" << endl ;
    RooErrorHandler::softAbort() ;    
  }

  TIterator* inputIter1 = lowSet.createIterator() ;
  RooAbsArg* comp ;
  while((comp = (RooAbsArg*)inputIter1->Next())) {
    if (!dynamic_cast<RooAbsReal*>(comp)) {
      coutE(InputArguments) << "PiecewiseInterpolation::ctor(" << GetName() << ") ERROR: component " << comp->GetName() 
			    << " in first list is not of type RooAbsReal" << endl ;
      RooErrorHandler::softAbort() ;
    }
    _lowSet.add(*comp) ;
    if (takeOwnership) {
      _ownedList.addOwned(*comp) ;
    }
  }
  delete inputIter1 ;


  TIterator* inputIter2 = highSet.createIterator() ;
  while((comp = (RooAbsArg*)inputIter2->Next())) {
    if (!dynamic_cast<RooAbsReal*>(comp)) {
      coutE(InputArguments) << "PiecewiseInterpolation::ctor(" << GetName() << ") ERROR: component " << comp->GetName() 
			    << " in first list is not of type RooAbsReal" << endl ;
      RooErrorHandler::softAbort() ;
    }
    _highSet.add(*comp) ;
    if (takeOwnership) {
      _ownedList.addOwned(*comp) ;
    }
  }
  delete inputIter2 ;


  TIterator* inputIter3 = paramSet.createIterator() ;
  while((comp = (RooAbsArg*)inputIter3->Next())) {
    if (!dynamic_cast<RooAbsReal*>(comp)) {
      coutE(InputArguments) << "PiecewiseInterpolation::ctor(" << GetName() << ") ERROR: component " << comp->GetName() 
			    << " in first list is not of type RooAbsReal" << endl ;
      RooErrorHandler::softAbort() ;
    }
    _paramSet.add(*comp) ;
    if (takeOwnership) {
      _ownedList.addOwned(*comp) ;
    }
  }
  delete inputIter3 ;
}



//_____________________________________________________________________________
PiecewiseInterpolation::PiecewiseInterpolation(const PiecewiseInterpolation& other, const char* name) :
  RooAbsReal(other, name), 
  _nominal("!nominal",this,other._nominal),
  _lowSet("!lowSet",this,other._lowSet),
  _highSet("!highSet",this,other._highSet),
  _paramSet("!paramSet",this,other._paramSet),
  _positiveDefinite(other._positiveDefinite)
{
  // Copy constructor

  _lowIter = _lowSet.createIterator() ;
  _highIter = _highSet.createIterator() ;
  _paramIter = _paramSet.createIterator() ;
  
  // Member _ownedList is intentionally not copy-constructed -- ownership is not transferred
}



//_____________________________________________________________________________
PiecewiseInterpolation::~PiecewiseInterpolation() 
{
  // Destructor

  if (_lowIter) delete _lowIter ;
  if (_highIter) delete _highIter ;
  if (_paramIter) delete _paramIter ;
}




//_____________________________________________________________________________
Double_t PiecewiseInterpolation::evaluate() const 
{
  // Calculate and return current value of self

  ///////////////////
  Double_t nominal = _nominal;
  Double_t sum(nominal) ;
  _lowIter->Reset() ;
  _highIter->Reset() ;
  _paramIter->Reset() ;
  

  RooAbsReal* param ;
  RooAbsReal* high ;
  RooAbsReal* low ;
  //  const RooArgSet* nset = _paramList.nset() ;
  int i=0;

  while((param=(RooAbsReal*)_paramIter->Next())) {
    low = (RooAbsReal*)_lowIter->Next() ;
    high = (RooAbsReal*)_highIter->Next() ;

    
    if(param->getVal()>0)
      sum += param->getVal()*(high->getVal() - nominal );
    else
      sum += param->getVal()*(nominal - low->getVal());

    ++i;
  }

  if(_positiveDefinite && (sum<0)){
     sum = 1e-6;
  }
  return sum;

}


//_____________________________________________________________________________
Int_t PiecewiseInterpolation::getAnalyticalIntegralWN(RooArgSet& allVars, RooArgSet& analVars, 
						      const RooArgSet* normSet, const char* /*rangeName*/) const 
{
  // Advertise that all integrals can be handled internally.

  /*
  cout << "---------------------------\nin PiecewiseInterpolation get analytic integral " <<endl;
  cout << "all vars = "<<endl;
  allVars.Print("v");
  cout << "anal vars = "<<endl;
  analVars.Print("v");
  cout << "normset vars = "<<endl;
  if(normSet2)
    normSet2->Print("v");
  */


  // Handle trivial no-integration scenario
  if (allVars.getSize()==0) return 0 ;
  if (_forceNumInt) return 0 ;


  // Select subset of allVars that are actual dependents
  analVars.add(allVars) ;
  //  RooArgSet* normSet = normSet2 ? getObservables(normSet2) : 0 ;
  //  RooArgSet* normSet = getObservables();
  //  RooArgSet* normSet = 0;


  // Check if this configuration was created before
  Int_t sterileIdx(-1) ;
  CacheElem* cache = (CacheElem*) _normIntMgr.getObj(normSet,&analVars,&sterileIdx,0) ;
  if (cache) {
    return _normIntMgr.lastIndex()+1 ;
  }
  
  // Create new cache element
  cache = new CacheElem ;

  // Make list of function projection and normalization integrals 
  RooAbsReal* param ;
  RooAbsReal *func ;
  //  const RooArgSet* nset = _paramList.nset() ;

  // do nominal
  func = (RooAbsReal*)(&_nominal.arg()) ;
  RooAbsReal* funcInt = func->createIntegral(analVars) ;
  cache->_funcIntList.addOwned(*funcInt) ;

  // do variations
  _lowIter->Reset() ;
  _highIter->Reset() ;
  _paramIter->Reset() ;
  int i=0;
  while(_paramIter->Next() ) {
    func = (RooAbsReal*)_lowIter->Next() ;
    funcInt = func->createIntegral(analVars) ;
    cache->_lowIntList.addOwned(*funcInt) ;

    func = (RooAbsReal*)_highIter->Next() ;
    funcInt = func->createIntegral(analVars) ;
    cache->_highIntList.addOwned(*funcInt) ;

    ++i;
  }

  // Store cache element
  Int_t code = _normIntMgr.setObj(normSet,&analVars,(RooAbsCacheElement*)cache,0) ;

  return code+1 ; 
}




//_____________________________________________________________________________
Double_t PiecewiseInterpolation::analyticalIntegralWN(Int_t code, const RooArgSet* /*normSet2*/,const char* /*rangeName*/) const 
{
  // Implement analytical integrations by doing appropriate weighting from  component integrals
  // functions to integrators of components

  CacheElem* cache = (CacheElem*) _normIntMgr.getObjByIndex(code-1) ;

  TIterator* funcIntIter = cache->_funcIntList.createIterator() ;
  TIterator* lowIntIter = cache->_lowIntList.createIterator() ;
  TIterator* highIntIter = cache->_highIntList.createIterator() ;
  RooAbsReal *funcInt(0), *low(0), *high(0), *param(0) ;
  Double_t value(0) ;
  Double_t nominal(0);

  // get nominal 
  int i=0;
  while(( funcInt = (RooAbsReal*)funcIntIter->Next())) {
    value += funcInt->getVal() ;
    nominal = value;
    i++;
  }
  if(i==0 || i>1)
    cout << "problem, wrong number of nominal functions"<<endl;

  // now get low/high variations
  i = 0;
  _paramIter->Reset() ;
  while((param=(RooAbsReal*)_paramIter->Next())) {
    low = (RooAbsReal*)lowIntIter->Next() ;
    high = (RooAbsReal*)highIntIter->Next() ;

    
    if(param->getVal()>0)
      value += param->getVal()*(high->getVal() - nominal );
    else
      value += param->getVal()*(nominal - low->getVal());

    ++i;
  }

  delete funcIntIter; 
  delete lowIntIter;
  delete highIntIter; 

  return value;

}


/*
//_____________________________________________________________________________
void PiecewiseInterpolation::printMetaArgs(ostream& os) const 
{
  // Customized printing of arguments of a PiecewiseInterpolation to more intuitively reflect the contents of the
  // product operator construction

  _lowIter->Reset() ;
  if (_highIter) {
    _highIter->Reset() ;
  }

  Bool_t first(kTRUE) ;
    
  RooAbsArg* arg1, *arg2 ;
  if (_highSet.getSize()!=0) { 

    while((arg1=(RooAbsArg*)_lowIter->Next())) {
      if (!first) {
	os << " + " ;
      } else {
	first = kFALSE ;
      }
      arg2=(RooAbsArg*)_highIter->Next() ;
      os << arg1->GetName() << " * " << arg2->GetName() ;
    }

  } else {
    
    while((arg1=(RooAbsArg*)_lowIter->Next())) {
      if (!first) {
	os << " + " ;
      } else {
	first = kFALSE ;
      }
      os << arg1->GetName() ; 
    }  

  }

  os << " " ;    
}

*/
