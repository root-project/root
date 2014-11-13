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
// RooDataHistSliceIter iterates over all bins in a RooDataHist that
// occur in a slice defined by the bin coordinates of the input
// sliceSet.
// END_HTML
//

#include "RooFit.h"

#if ROOT_VERSION_CODE <= ROOT_VERSION(5,19,04)
#ifndef nullptr
#define nullptr 0
#endif
#endif

#include "RooDataHist.h"
#include "RooArgSet.h"
#include "RooAbsLValue.h"
#include "RooDataHistSliceIter.h"

using namespace std;

ClassImp(RooDataHistSliceIter)
;



//_____________________________________________________________________________
RooDataHistSliceIter::RooDataHistSliceIter(RooDataHist& hist, RooAbsArg& sliceArg) : _hist(&hist), _sliceArg(&sliceArg)
{
  // Construct an iterator over all bins of RooDataHist 'hist' in the slice defined
  // by the values of the arguments in 'sliceArg'

  // Calculate base index (for 0th bin) for slice    
  RooAbsArg* sliceArgInt = hist.get()->find(sliceArg.GetName()) ;
  dynamic_cast<RooAbsLValue&>(*sliceArgInt).setBin(0) ;

  if (hist._vars.getSize()>1) {
    _baseIndex = hist.calcTreeIndex() ;  
  } else {
    _baseIndex = 0 ;
  }

  _nStep = dynamic_cast<RooAbsLValue&>(*sliceArgInt).numBins() ;

//   cout << "RooDataHistSliceIter" << endl ;
//   hist.Print() ;
//   cout << "hist._iterator = " << hist._iterator << endl ;

  hist._iterator->Reset() ;
  RooAbsArg* arg ;
  Int_t i=0 ;
  while((arg=(RooAbsArg*)hist._iterator->Next())) {
    if (arg==sliceArgInt) break ;
    i++ ;
  }
  _stepSize = hist._idxMult[i] ;  
  _curStep = 0 ;  

}



//_____________________________________________________________________________
RooDataHistSliceIter::RooDataHistSliceIter(const RooDataHistSliceIter& other) : 
  TIterator(other), 
  _hist(other._hist), 
  _sliceArg(other._sliceArg),   
  _baseIndex(other._baseIndex),
  _stepSize(other._stepSize),
  _nStep(other._nStep), 
  _curStep(other._curStep)
{
  // Copy constructor
}



//_____________________________________________________________________________
RooDataHistSliceIter::~RooDataHistSliceIter() 
{
  // Destructor
}



//_____________________________________________________________________________
const TCollection* RooDataHistSliceIter::GetCollection() const 
{
  // Dummy
  return 0 ;
}




//_____________________________________________________________________________
TObject* RooDataHistSliceIter::Next() 
{  
  // Iterator increment operator

  if (_curStep==_nStep) {
    return 0 ;
  }
  
  // Select appropriate entry in RooDataHist 
  _hist->get(_baseIndex + _curStep*_stepSize) ;

  // Increment iterator position 
  _curStep++ ;

  return _sliceArg ;
}



//_____________________________________________________________________________
void RooDataHistSliceIter::Reset() 
{
  // Reset iterator position to beginning
  _curStep=0 ;
}



//_____________________________________________________________________________
TObject *RooDataHistSliceIter::operator*() const
{
  // Iterator dereference operator, not functional for this iterator

   Int_t step = _curStep == 0 ? _curStep : _curStep - 1;
   // Select appropriate entry in RooDataHist 
   _hist->get(_baseIndex + step*_stepSize) ;

   return _sliceArg ;
}


//_____________________________________________________________________________
bool RooDataHistSliceIter::operator!=(const TIterator &aIter) const
{
  // Returns true if position of this iterator differs from position
  // of iterator 'aIter'

   if ((aIter.IsA() == RooDataHistSliceIter::Class())) {
      const RooDataHistSliceIter &iter(dynamic_cast<const RooDataHistSliceIter &>(aIter));
      return (_curStep != iter._curStep);
   }
   
   return false;
}
