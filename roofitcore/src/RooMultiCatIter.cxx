/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooMultiCatIter.cc,v 1.4 2001/05/11 23:37:41 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION --
// RooMultiCatIter iterators over all state permutations of a list of categories.
// It serves as the state iterator for a RooSuperCategory.

#include "RooFitCore/RooAbsCategoryLValue.hh"
#include "RooFitCore/RooMultiCatIter.hh"

ClassImp(RooMultiCatIter)
;


RooMultiCatIter::RooMultiCatIter(const RooArgSet& catList) : _catList("catList") 
{
  // Constructor
  initialize(catList) ;
}


RooMultiCatIter::RooMultiCatIter(const RooMultiCatIter& other) : _catList("catList")
{
  // Copy constructor
  initialize(other._catList) ;
}


void RooMultiCatIter::initialize(const RooArgSet& catList) 
{
  // Build iterator array for given catList

  // Copy RooCategory list into internal argset
  TIterator* catIter = catList.MakeIterator() ;
  TObject* obj ;
  while (obj = catIter->Next()) {
    if (!obj->IsA()->InheritsFrom(RooAbsCategoryLValue::Class())) {
      cout << "RooMultiCatIter:: list element " << obj->GetName() 
	   << " is not a RooAbsCategoryLValue, ignored" << endl ;
      continue ;
    }
    _catList.Add(obj) ;
  }
  delete catIter ;
  
  // Allocate storage for component iterators
  _nIter = catList.GetSize() ;
  _iterList   = new pTIterator[_nIter] ;
  _catPtrList = new pRooCategory[_nIter] ;

  // Construct component iterators
  _curIter = 0 ;
  TIterator* cIter = _catList.MakeIterator() ;
  RooAbsCategoryLValue* cat ;
  while(cat=(RooAbsCategoryLValue*)cIter->Next()) {
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
}



const TCollection* RooMultiCatIter::GetCollection() const 
{
  // Return set of categories iterated over
  return &_catList ;
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
    _catPtrList[_curIter]->setIndex(next->getVal()) ;

    // If higher order increment was successful, reset master iterator
    if (_curIter>0) _curIter=0 ;

    return &_catList ;    
  } else {

    // Reset current iterator
    _iterList[_curIter]->Reset() ;
    next = (RooCatType*) _iterList[_curIter]->Next() ;
    if (next) _catPtrList[_curIter]->setIndex(next->getVal()) ;

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
      _catPtrList[_curIter]->setIndex(first->getVal()) ;
    }
  }
  _curIter=0 ;
}
