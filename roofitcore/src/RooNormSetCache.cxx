/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id$
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   24-Sep-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION [AUX] --
#include "RooFitCore/RooNormSetCache.hh"
#include "RooFitCore/RooArgSet.hh"

ClassImp(RooNormSetCache)
;

RooNormSetCache::RooNormSetCache(Int_t regSize) :
  _asArr1(0), _asArr2(0), _regSize(regSize), _nreg(0)
{
}


RooNormSetCache::RooNormSetCache(const RooNormSetCache& other) :
  _asArr1(0), _asArr2(0), _regSize(other._regSize), _nreg(0)
{
}



RooNormSetCache::~RooNormSetCache() 
{
  delete[] _asArr1 ;
  delete[] _asArr2 ;
}



void RooNormSetCache::clear()
{
  _nreg = 0 ;
}



void RooNormSetCache::add(const RooArgSet* set1, const RooArgSet* set2)
{
  Int_t i,j ;

  // If code list array has never been used, allocate and initialize here
  if (!_asArr1) {
    _asArr1 = new pRooArgSet[_regSize] ;
    _asArr2 = new pRooArgSet[_regSize] ;
  }

  if (!contains(set1,set2) && _nreg<_regSize) {
    // Add to cache
    _asArr1[_nreg] = (RooArgSet*) set1 ;
    _asArr2[_nreg] = (RooArgSet*) set2 ;
    _nreg++ ;
  }

}

