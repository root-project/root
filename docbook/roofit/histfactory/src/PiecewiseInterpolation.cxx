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
  _paramSet("!paramSet","high-side variation",this)
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
  _paramSet("!paramSet",this,other._paramSet)
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

  return sum;

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
