/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooAICRegistry.cc,v 1.2 2001/09/26 18:29:32 verkerke Exp $
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   24-Sep-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION [AUX] --

#include "RooFitCore/RooAICRegistry.hh"
#include "RooFitCore/RooArgSet.hh"

ClassImp(RooAICRegistry)
;

RooAICRegistry::RooAICRegistry(Int_t regSize) :
  _clArr(0), _asArr1(0), _asArr2(0), _regSize(regSize)
{
}


RooAICRegistry::RooAICRegistry(const RooAICRegistry& other) :
  _clArr(0), _asArr1(0), _asArr2(0), _regSize(other._regSize)
{
  // Copy code-list array if other PDF has one
  if (other._clArr) {
    _clArr = new pInt_t[other._regSize] ;    
    _asArr1 = new pRooArgSet[other._regSize] ;
    _asArr2 = new pRooArgSet[other._regSize] ;
    _clSize = new Int_t[other._regSize] ;
    Int_t i,j ;
    for (i=0 ; i<_regSize ; i++) {
      _clArr[i]=0 ;
      _clSize[i]=0 ;
      _asArr1[i]=0 ;
      _asArr2[i]=0 ;
    }
    i=0 ;
    while(other._clArr[i] && i<_regSize) {
      _clSize[i] = other._clSize[i] ;
      _asArr1[i] = other._asArr1[i] ;
      _asArr2[i] = other._asArr2[i] ;
      _clArr[i] = new Int_t[_clSize[i]] ;
      for (j=0 ; j<_clSize[i] ; j++) {
	_clArr[i][j] = other._clArr[i][j] ;
      }
      i++ ;
    }
  }
}



RooAICRegistry::~RooAICRegistry() 
{
  // Delete code list array, if allocated
  if (_clArr) {
    Int_t i(0) ;
    while(_clArr[i] && i<_regSize) {
      delete[] _clArr[i++] ;
      if (_asArr1[i]) delete _asArr1[i] ;
      if (_asArr2[i]) delete _asArr2[i] ;
    }
    delete[] _clArr ;
    delete[] _clSize ;
    delete[] _asArr1 ;
    delete[] _asArr2 ;
  }
}



Int_t RooAICRegistry::store(Int_t* codeList, Int_t size, RooArgSet* set1, RooArgSet* set2)
{
  Int_t i,j ;

  // If code list array has never been used, allocate and initialize here
  if (!_clArr) {
    _clArr = new pInt_t[_regSize] ;
    _clSize = new Int_t[_regSize] ;
    _asArr1 = new pRooArgSet[_regSize] ;
    _asArr2 = new pRooArgSet[_regSize] ;
    for (i=0 ; i<_regSize ; i++) {
      _clArr[i] = 0 ;    
      _clSize[i] = 0 ;
      _asArr1[i] = 0 ;
      _asArr2[i] = 0 ;
    }
  }

  // Loop over code-list array
  for (i=0 ; i<_regSize ; i++) {
    if (_clArr[i]==0) {
      // Empty slot, store code list and return index
      _clArr[i] = new Int_t[size] ;
      _clSize[i] = size ;
      _asArr1[i] = set1 ;
      _asArr2[i] = set2 ;
      for (j=0 ; j<size ; j++) _clArr[i][j] = codeList[j] ;
      return i ;
    } else {
      // Existing slot, compare with current list, if matched return index
      Bool_t match(kTRUE) ;
      for (j=0 ; j<size ; j++) {
	if (_clArr[i][j] != codeList[j]) match=kFALSE ;
      }
      if (match) {
	if (set1) delete set1 ;
	if (set2) delete set2 ;
	return i ;
      }
    }
  }

  cout << "RooAICRegistry::store: ERROR: capacity exceeded" << endl ;
  assert(0) ;
  return 0 ;
}


const Int_t* RooAICRegistry::retrieve(Int_t masterCode) const 
{
  return _clArr[masterCode] ;
}


const Int_t* RooAICRegistry::retrieve(Int_t masterCode, pRooArgSet& set1) const 
{
  set1 = _asArr1[masterCode] ;
  return _clArr[masterCode] ;
}

const Int_t* RooAICRegistry::retrieve(Int_t masterCode, pRooArgSet& set1, pRooArgSet& set2) const 
{
  set1 = _asArr1[masterCode] ;
  set2 = _asArr2[masterCode] ;
  return _clArr[masterCode] ;
}
