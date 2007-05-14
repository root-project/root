/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Name:  $:$Id: RooMultiCatIter.cxx,v 1.19 2007/05/11 09:11:58 verkerke Exp $
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

// -- CLASS DESCRIPTION [AUX] --
// RooMultiCatIter iterators over all state permutations of a list of categories.
// It serves as the state iterator for a RooSuperCategory.

#include "RooFit.h"

#include "RooAbsCategoryLValue.h"
#include "RooAbsCategoryLValue.h"
#include "RooMultiCatIter.h"

ClassImp(RooMultiCatIter)
;


RooMultiCatIter::RooMultiCatIter(const RooArgSet& catList, const char* rangeName) : _catList("catList") 
{
  // Constructor
  if (rangeName) {
    _rangeName = rangeName ;
  }
  initialize(catList) ;
}


RooMultiCatIter::RooMultiCatIter(const RooMultiCatIter& other) : TIterator(other), _catList("catList")
{
  // Copy constructor
  initialize(other._catList) ;
}


void RooMultiCatIter::initialize(const RooArgSet& catList) 
{
  // Build iterator array for given catList

  // Copy RooCategory list into internal argset
  TIterator* catIter = catList.createIterator() ;
  TObject* obj ;
  while ((obj = catIter->Next())) {
    RooAbsCategory *cat= dynamic_cast<RooAbsCategoryLValue*>(obj);
    if(0 == cat) {
      cout << "RooMultiCatIter:: list element " << obj->GetName() 
	   << " is not a RooAbsCategoryLValue, ignored" << endl ;
      continue ;
    }
    _catList.add(*cat) ;
  }
  delete catIter ;
  
  // Allocate storage for component iterators
  _nIter = catList.getSize() ;
  _iterList   = new pTIterator[_nIter] ;
  _catPtrList = new pRooCategory[_nIter] ;
  _curTypeList = new RooCatType[_nIter] ;

  // Construct component iterators
  _curIter = 0 ;
  TIterator* cIter = _catList.createIterator() ;
  RooAbsCategoryLValue* cat ;
  while((cat=(RooAbsCategoryLValue*)cIter->Next())) {
    _catPtrList[_curIter] = cat ;
    _iterList[_curIter++] = cat->typeIterator() ;
  }
  delete cIter ;

  Reset() ;
}


RooMultiCatIter::~RooMultiCatIter() 
{
  // Destructor
  for (_curIter=0 ; _curIter<_nIter ; _curIter++) {
    delete _iterList[_curIter] ;
  }
  delete[] _iterList ;
  delete[] _catPtrList ;
  delete[] _curTypeList ;
}



const TCollection* RooMultiCatIter::GetCollection() const 
{
  // Return set of categories iterated over
  //return &_catList.getCollection() ;
  return 0 ;
}



TObjString* RooMultiCatIter::compositeLabel() 
{
  TString& str = _compositeLabel.String() ;

  str = "{" ;
  Int_t i ;
  for (i=0 ; i<_nIter ; i++) {
    if (i>0) str.Append(";") ;
    str.Append(_curTypeList[i].GetName()) ;
  }
  str.Append("}") ;

  return &_compositeLabel ;
}



TObject* RooMultiCatIter::Next() 
{
  // Iterator increment operator

  // Check for end
  if (_curIter==_nIter) {
    return 0 ;
  }

  RooCatType* next = (RooCatType*) _iterList[_curIter]->Next() ;
  if (next) { 

    // Increment current iterator
    _curTypeList[_curIter] = *next ;
    //_catPtrList[_curIter]->setIndex(next->getVal()) ;

    // If higher order increment was successful, reset master iterator
    if (_curIter>0) _curIter=0 ;

    return compositeLabel() ;    
  } else {

    // Reset current iterator
    _iterList[_curIter]->Reset() ;
    next = (RooCatType*) _iterList[_curIter]->Next() ;
    if (next) _curTypeList[_curIter] = *next ; 
    //if (next) _catPtrList[_curIter]->setIndex(next->getVal()) ;

    // Increment next iterator 
    _curIter++ ;
    return Next() ;
  }
}



void RooMultiCatIter::Reset() 
{
  // Rewind master iterator

  for (_curIter=0 ; _curIter<_nIter ; _curIter++) {
    TIterator* cIter = _iterList[_curIter] ;
    cIter->Reset() ;
    RooCatType* first = (RooCatType*) cIter->Next() ;
    if (first) {
      if (_curIter==0) cIter->Reset() ;
//    _catPtrList[_curIter]->setIndex(first->getVal()) ;
      _curTypeList[_curIter] = *first ;
    }
  }
  _curIter=0 ;
}
