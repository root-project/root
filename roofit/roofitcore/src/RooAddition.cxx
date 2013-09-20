/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

//////////////////////////////////////////////////////////////////////////////
// 
// BEGIN_HTML
// RooAddition calculates the sum of a set of RooAbsReal terms, or
// when constructed with two sets, it sums the product of the terms
// in the two sets. This class does not (yet) do any smart handling of integrals, 
// i.e. all integrals of the product are handled numerically
// END_HTML
//


#include "RooFit.h"

#include "Riostream.h"
#include <math.h>
#include <memory>
#include <list>
#include <algorithm>
using namespace std ;

#include "RooAddition.h"
#include "RooProduct.h"
#include "RooAbsReal.h"
#include "RooErrorHandler.h"
#include "RooArgSet.h"
#include "RooNameReg.h"
#include "RooNLLVar.h"
#include "RooChi2Var.h"
#include "RooMsgService.h"

ClassImp(RooAddition)
;


//_____________________________________________________________________________
RooAddition::RooAddition()
  : _setIter( _set.createIterator() )
{
}



//_____________________________________________________________________________
RooAddition::RooAddition(const char* name, const char* title, const RooArgList& sumSet, Bool_t takeOwnership) 
  : RooAbsReal(name, title)
  , _set("!set","set of components",this)
  , _setIter( _set.createIterator() ) // yes, _setIter is defined _after_ _set ;-)
  , _cacheMgr(this,10)
{
  // Constructor with a single set of RooAbsReals. The value of the function will be
  // the sum of the values in sumSet. If takeOwnership is true the RooAddition object
  // will take ownership of the arguments in sumSet

  std::auto_ptr<TIterator> inputIter( sumSet.createIterator() );
  RooAbsArg* comp ;
  while((comp = (RooAbsArg*)inputIter->Next())) {
    if (!dynamic_cast<RooAbsReal*>(comp)) {
      coutE(InputArguments) << "RooAddition::ctor(" << GetName() << ") ERROR: component " << comp->GetName() 
			    << " is not of type RooAbsReal" << endl ;
      RooErrorHandler::softAbort() ;
    }
    _set.add(*comp) ;
    if (takeOwnership) _ownedList.addOwned(*comp) ;
  }

}



//_____________________________________________________________________________
RooAddition::RooAddition(const char* name, const char* title, const RooArgList& sumSet1, const RooArgList& sumSet2, Bool_t takeOwnership) 
    : RooAbsReal(name, title)
    , _set("!set","set of components",this)
    , _setIter( _set.createIterator() ) // yes, _setIter is defined _after_ _set ;-)
    , _cacheMgr(this,10)
{
  // Constructor with two set of RooAbsReals. The value of the function will be
  //
  //  A = sum_i sumSet1(i)*sumSet2(i) 
  //
  // If takeOwnership is true the RooAddition object will take ownership of the arguments in sumSet

  if (sumSet1.getSize() != sumSet2.getSize()) {
    coutE(InputArguments) << "RooAddition::ctor(" << GetName() << ") ERROR: input lists should be of equal length" << endl ;
    RooErrorHandler::softAbort() ;    
  }

  std::auto_ptr<TIterator> inputIter1( sumSet1.createIterator() );
  std::auto_ptr<TIterator> inputIter2( sumSet2.createIterator() );
  RooAbsArg *comp1(0),*comp2(0) ;
  while((comp1 = (RooAbsArg*)inputIter1->Next())) {
    if (!dynamic_cast<RooAbsReal*>(comp1)) {
      coutE(InputArguments) << "RooAddition::ctor(" << GetName() << ") ERROR: component " << comp1->GetName() 
			    << " in first list is not of type RooAbsReal" << endl ;
      RooErrorHandler::softAbort() ;
    }
    comp2 = (RooAbsArg*)inputIter2->Next();
    if (!dynamic_cast<RooAbsReal*>(comp2)) {
      coutE(InputArguments) << "RooAddition::ctor(" << GetName() << ") ERROR: component " << comp2->GetName() 
			    << " in first list is not of type RooAbsReal" << endl ;
      RooErrorHandler::softAbort() ;
    }
    // TODO: add flag to RooProduct c'tor to make it assume ownership...
    TString _name(name);
    _name.Append( "_[");
    _name.Append(comp1->GetName());
    _name.Append( "_x_");
    _name.Append(comp2->GetName());
    _name.Append( "]");
    RooProduct  *prod = new RooProduct( _name, _name , RooArgSet(*comp1, *comp2) /*, takeOwnership */ ) ;
    _set.add(*prod);
    _ownedList.addOwned(*prod) ;
    if (takeOwnership) {
        _ownedList.addOwned(*comp1) ;
        _ownedList.addOwned(*comp2) ;
    }
  }
}



//_____________________________________________________________________________
RooAddition::RooAddition(const RooAddition& other, const char* name) 
    : RooAbsReal(other, name)
    , _set("!set",this,other._set)
    , _setIter( _set.createIterator() ) // yes, _setIter is defined _after_ _set ;-)
    , _cacheMgr(other._cacheMgr,this)
{
  // Copy constructor
  
  // Member _ownedList is intentionally not copy-constructed -- ownership is not transferred
}


//_____________________________________________________________________________
RooAddition::~RooAddition() 
{ // Destructor
  delete _setIter ;
}

//_____________________________________________________________________________
Double_t RooAddition::evaluate() const 
{
  // Calculate and return current value of self
  Double_t sum(0);
  const RooArgSet* nset = _set.nset() ;

//   cout << "RooAddition::eval sum = " ;

  RooFIter setIter = _set.fwdIterator() ;
  RooAbsReal* comp ;
  while((comp=(RooAbsReal*)setIter.next())) {
    Double_t tmp = comp->getVal(nset) ;
//     cout << tmp << " " ;
    sum += tmp ;
  }
//   cout << " = " << sum << endl ;
  return sum ;
}


//_____________________________________________________________________________
Double_t RooAddition::defaultErrorLevel() const 
{
  // Return the default error level for MINUIT error analysis
  // If the addition contains one or more RooNLLVars and 
  // no RooChi2Vars, return the defaultErrorLevel() of
  // RooNLLVar. If the addition contains one ore more RooChi2Vars
  // and no RooNLLVars, return the defaultErrorLevel() of
  // RooChi2Var. If the addition contains neither or both
  // issue a warning message and return a value of 1

  RooAbsReal* nllArg(0) ;
  RooAbsReal* chi2Arg(0) ;

  RooAbsArg* arg ;

  RooArgSet* comps = getComponents() ;
  TIterator* iter = comps->createIterator() ;
  while((arg=(RooAbsArg*)iter->Next())) {
    if (dynamic_cast<RooNLLVar*>(arg)) {
      nllArg = (RooAbsReal*)arg ;
    }
    if (dynamic_cast<RooChi2Var*>(arg)) {
      chi2Arg = (RooAbsReal*)arg ;
    }
  }
  delete iter ;
  delete comps ;

  if (nllArg && !chi2Arg) {
    coutI(Fitting) << "RooAddition::defaultErrorLevel(" << GetName() 
		   << ") Summation contains a RooNLLVar, using its error level" << endl ;
    return nllArg->defaultErrorLevel() ;
  } else if (chi2Arg && !nllArg) {
    coutI(Fitting) << "RooAddition::defaultErrorLevel(" << GetName() 
		   << ") Summation contains a RooChi2Var, using its error level" << endl ;
    return chi2Arg->defaultErrorLevel() ;
  } else if (!nllArg && !chi2Arg) {
    coutI(Fitting) << "RooAddition::defaultErrorLevel(" << GetName() << ") WARNING: "
		   << "Summation contains neither RooNLLVar nor RooChi2Var server, using default level of 1.0" << endl ;
  } else {
    coutI(Fitting) << "RooAddition::defaultErrorLevel(" << GetName() << ") WARNING: "
		   << "Summation contains BOTH RooNLLVar and RooChi2Var server, using default level of 1.0" << endl ;
  }

  return 1.0 ;
}


//_____________________________________________________________________________
void RooAddition::enableOffsetting(Bool_t flag) 
{
  _setIter->Reset() ;

  RooAbsReal* arg;
  while((arg=(RooAbsReal*)_setIter->Next())) {
    arg->enableOffsetting(flag) ;
  }  
}



//_____________________________________________________________________________
Bool_t RooAddition::setData(RooAbsData& data, Bool_t cloneData) 
{
  _setIter->Reset() ;

  RooAbsReal* arg;
  while((arg=(RooAbsReal*)_setIter->Next())) {
    arg->setData(data,cloneData) ;
  }  
  return kTRUE ;
}



//_____________________________________________________________________________
void RooAddition::printMetaArgs(ostream& os) const 
{
  _setIter->Reset() ;

  Bool_t first(kTRUE) ;
  RooAbsArg* arg;
  while((arg=(RooAbsArg*)_setIter->Next())) {
    if (!first) { os << " + " ;
    } else { first = kFALSE ; 
    }
    os << arg->GetName() ; 
  }  
  os << " " ;    
}

//_____________________________________________________________________________
Int_t RooAddition::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* rangeName) const
{
  
  // we always do things ourselves -- actually, always delegate further down the line ;-)
  analVars.add(allVars);

  // check if we already have integrals for this combination of factors
  Int_t sterileIndex(-1);
  CacheElem* cache = (CacheElem*) _cacheMgr.getObj(&analVars,&analVars,&sterileIndex,RooNameReg::ptr(rangeName));
  if (cache!=0) {
    Int_t code = _cacheMgr.lastIndex();
    return code+1;
  }

  // we don't, so we make it right here....
  cache = new CacheElem;
  _setIter->Reset();
  RooAbsReal *arg(0);
  while( (arg=(RooAbsReal*)_setIter->Next())!=0 ) {  // checked in c'tor that this will work...
      RooAbsReal *I = arg->createIntegral(analVars,rangeName);
      cache->_I.addOwned(*I);
  }

  Int_t code = _cacheMgr.setObj(&analVars,&analVars,(RooAbsCacheElement*)cache,RooNameReg::ptr(rangeName));
  return 1+code;
}

//_____________________________________________________________________________
Double_t RooAddition::analyticalIntegral(Int_t code, const char* rangeName) const 
{
  // Calculate integral internally from appropriate integral cache

  // note: rangeName implicit encoded in code: see _cacheMgr.setObj in getPartIntList...
  CacheElem *cache = (CacheElem*) _cacheMgr.getObjByIndex(code-1);
  if (cache==0) {
    // cache got sterilized, trigger repopulation of this slot, then try again...
    std::auto_ptr<RooArgSet> vars( getParameters(RooArgSet()) );
    std::auto_ptr<RooArgSet> iset(  _cacheMgr.nameSet2ByIndex(code-1)->select(*vars) );
    RooArgSet dummy;
    Int_t code2 = getAnalyticalIntegral(*iset,dummy,rangeName);
    assert(code==code2); // must have revived the right (sterilized) slot...
    return analyticalIntegral(code2,rangeName);
  }
  assert(cache!=0);

  // loop over cache, and sum...
  std::auto_ptr<TIterator> iter( cache->_I.createIterator() );
  RooAbsReal *I;
  double result(0);
  while ( ( I=(RooAbsReal*)iter->Next() ) != 0 ) result += I->getVal();
  return result;

}



//_____________________________________________________________________________
std::list<Double_t>* RooAddition::binBoundaries(RooAbsRealLValue& obs, Double_t xlo, Double_t xhi) const
{
  std::list<Double_t>* sumBinB = 0 ;
  Bool_t needClean(kFALSE) ;
  
  RooFIter iter = _set.fwdIterator() ;
  RooAbsReal* func ;
  // Loop over components pdf
  while((func=(RooAbsReal*)iter.next())) {

    std::list<Double_t>* funcBinB = func->binBoundaries(obs,xlo,xhi) ;
    
    // Process hint
    if (funcBinB) {
      if (!sumBinB) {
	// If this is the first hint, then just save it
	sumBinB = funcBinB ;
      } else {
	
	std::list<Double_t>* newSumBinB = new std::list<Double_t>(sumBinB->size()+funcBinB->size()) ;

	// Merge hints into temporary array
	merge(funcBinB->begin(),funcBinB->end(),sumBinB->begin(),sumBinB->end(),newSumBinB->begin()) ;
	
	// Copy merged array without duplicates to new sumBinBArrau
	delete sumBinB ;
	delete funcBinB ;
	sumBinB = newSumBinB ;
	needClean = kTRUE ;	
      }
    }
  }

  // Remove consecutive duplicates
  if (needClean) {
    std::list<Double_t>::iterator new_end = unique(sumBinB->begin(),sumBinB->end()) ;
    sumBinB->erase(new_end,sumBinB->end()) ;
  }

  return sumBinB ;
}


//_____________________________________________________________________________B
Bool_t RooAddition::isBinnedDistribution(const RooArgSet& obs) const 
{
  // If all components that depend on obs are binned that so is the product
  
  RooFIter iter = _set.fwdIterator() ;
  RooAbsReal* func ;
  while((func=(RooAbsReal*)iter.next())) {
    if (func->dependsOn(obs) && !func->isBinnedDistribution(obs)) {
      return kFALSE ;
    }
  }
  
  return kTRUE  ;  
}




//_____________________________________________________________________________
std::list<Double_t>* RooAddition::plotSamplingHint(RooAbsRealLValue& obs, Double_t xlo, Double_t xhi) const
{
  std::list<Double_t>* sumHint = 0 ;
  Bool_t needClean(kFALSE) ;
  
  RooFIter iter = _set.fwdIterator() ;
  RooAbsReal* func ;
  // Loop over components pdf
  while((func=(RooAbsReal*)iter.next())) {
    
    std::list<Double_t>* funcHint = func->plotSamplingHint(obs,xlo,xhi) ;
    
    // Process hint
    if (funcHint) {
      if (!sumHint) {

	// If this is the first hint, then just save it
	sumHint = funcHint ;

      } else {
	
	std::list<Double_t>* newSumHint = new std::list<Double_t>(sumHint->size()+funcHint->size()) ;
	
	// Merge hints into temporary array
	merge(funcHint->begin(),funcHint->end(),sumHint->begin(),sumHint->end(),newSumHint->begin()) ;

	// Copy merged array without duplicates to new sumHintArrau
	delete sumHint ;
	sumHint = newSumHint ;
	needClean = kTRUE ;	
      }
    }
  }

  // Remove consecutive duplicates
  if (needClean) {
    std::list<Double_t>::iterator new_end = unique(sumHint->begin(),sumHint->end()) ;
    sumHint->erase(new_end,sumHint->end()) ;
  }

  return sumHint ;
}



//_____________________________________________________________________________
RooArgList RooAddition::CacheElem::containedArgs(Action)
{
  // Return list of all RooAbsArgs in cache element
  RooArgList ret(_I) ;
  return ret ;
}

RooAddition::CacheElem::~CacheElem()
{
  // Destructor
}


