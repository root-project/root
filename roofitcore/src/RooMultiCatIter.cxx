/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooMultiCatIter.cc,v 1.1 2001/04/18 20:38:02 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

#include "RooFitCore/RooCategory.hh"
#include "RooFitCore/RooMultiCatIter.hh"

ClassImp(RooMultiCatIter)
;


RooMultiCatIter::RooMultiCatIter(const RooArgSet& catList) : _catList("catList") 
{
  initialize(catList) ;
}


RooMultiCatIter::RooMultiCatIter(const RooMultiCatIter& other) : _catList("catList")
{
  initialize(other._catList) ;
}


void RooMultiCatIter::initialize(const RooArgSet& catList) 
{
  // Copy RooCategory list into internal argset
  TIterator* catIter = catList.MakeIterator() ;
  TObject* obj ;
  while (obj = catIter->Next()) {
    if (obj->IsA()!=RooCategory::Class()) {
      cout << "RooMultiCatIter:: list element " << obj->GetName() 
	   << " is not a RooCategory, ignored" << endl ;
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
  RooCategory* cat ;
  while(cat=(RooCategory*)cIter->Next()) {
    _catPtrList[_curIter] = cat ;
    _iterList[_curIter++] = cat->typeIterator() ;
  }
  delete cIter ;

  Reset() ;
}


RooMultiCatIter::~RooMultiCatIter() 
{
  for (_curIter=0 ; _curIter<_nIter ; _curIter++) {
    delete _iterList[_curIter] ;
  }
  delete[] _iterList ;
  delete[] _catPtrList ;
}



const TCollection* RooMultiCatIter::GetCollection() const 
{
  return &_catList ;
}



TObject* RooMultiCatIter::Next() 
{
  // Check for end
  if (_curIter==_nIter) {
    return 0 ;
  }

  // Increment current iterator
  RooCatType* next = (RooCatType*) _iterList[_curIter]->Next() ;
  if (next) { 
    _catPtrList[_curIter]->setIndex(next->getVal()) ;
    return &_catList ;
  }  

  // If at end of current iter, rewind and increment next iter
  _iterList[_curIter++]->Reset() ;
  return Next() ;
}



void RooMultiCatIter::Reset() 
{
  for (_curIter=0 ; _curIter<_nIter ; _curIter++) {
    TIterator* cIter = _iterList[_curIter] ;
    cIter->Reset() ;
    RooCatType* first = (RooCatType*) cIter->Next() ;
    if (first) {
      cIter->Reset() ;
      _catPtrList[_curIter]->setIndex(first->getVal()) ;
    }
  }
  _curIter=0 ;
}
