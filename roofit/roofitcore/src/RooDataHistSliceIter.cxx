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

// -- CLASS DESCRIPTION [AUX] --
// RooDataHistSliceIter iterators over all state permutations of a list of categories.
// It serves as the state iterator for a RooSuperCategory.

#include "RooFit.h"

#include "RooDataHist.h"
#include "RooArgSet.h"
#include "RooAbsLValue.h"
#include "RooDataHistSliceIter.h"

ClassImp(RooDataHistSliceIter)
;


RooDataHistSliceIter::RooDataHistSliceIter(RooDataHist& hist, RooAbsArg& sliceArg) : _hist(&hist), _sliceArg(&sliceArg)
{
  // Calculate base index (for 0th bin) for slice    
  RooAbsArg* sliceArgInt = hist.get()->find(sliceArg.GetName()) ;
  dynamic_cast<RooAbsLValue&>(*sliceArgInt).setBin(0,hist.bname()) ;
  _baseIndex = hist.calcTreeIndex() ;
  _nStep = dynamic_cast<RooAbsLValue&>(*sliceArgInt).numBins(hist.bname()) ;

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


RooDataHistSliceIter::RooDataHistSliceIter(const RooDataHistSliceIter& other) : 
  TIterator(other), 
  _hist(other._hist), 
  _sliceArg(other._sliceArg), 
  _nStep(other._nStep), 
  _curStep(other._curStep)
{
  // Copy constructor
}



RooDataHistSliceIter::~RooDataHistSliceIter() 
{
  // Destructor
}



const TCollection* RooDataHistSliceIter::GetCollection() const 
{
  // Return set of categories iterated over
  return 0 ;
}




TObject* RooDataHistSliceIter::Next() 
{
  // Iterator increment operator
  if (_curStep==_nStep) return 0 ;
  
  // Select appropriate entry in RooDataHist 
  _hist->get(_baseIndex + _curStep*_stepSize) ;

  // Increment iterator position 
  _curStep++ ;

  return _sliceArg ;
}



void RooDataHistSliceIter::Reset() 
{
  _curStep=0 ;
}


TObject *RooDataHistSliceIter::operator*() const
{
   Int_t step = _curStep == 0 ? _curStep : _curStep - 1;
   // Select appropriate entry in RooDataHist 
   _hist->get(_baseIndex + step*_stepSize) ;

   return _sliceArg ;
}

bool RooDataHistSliceIter::operator!=(const TIterator &aIter) const
{
   if (nullptr == &aIter)
      return false;
   if ((aIter.IsA() == RooDataHistSliceIter::Class())) {
      const RooDataHistSliceIter &iter(dynamic_cast<const RooDataHistSliceIter &>(aIter));
      return (_curStep != iter._curStep);
   }
   
   return false;
}
