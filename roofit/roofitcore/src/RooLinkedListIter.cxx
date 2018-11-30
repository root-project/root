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

/**
\file RooLinkedListIter.cxx
\class RooLinkedListIter
\ingroup Roofitcore

RooLinkedListIter is the TIterator implementation for RooLinkedList
**/

#include "RooLinkedListIter.h"

#include "RooAbsArg.h"

#include <vector>
#include <deque>

template class TIteratorToSTLInterface<std::vector<RooAbsArg*>>;
template class TIteratorToSTLInterface<std::deque<RooAbsArg*>>;


template<class STLContainer>
TObject * TIteratorToSTLInterface<STLContainer>::Next() {
  return static_cast<TObject*>(next());
}


template<class STLContainer>
TObject * TIteratorToSTLInterface<STLContainer>::operator*() const {
  if (atEnd())
    return nullptr;

#ifndef NDEBUG
  assert(fCurrentElem == fSTLContainer[fIndex]);
#endif

  return static_cast<TObject*>(fSTLContainer[fIndex]);
}


#ifndef NDEBUG
template<class STLContainer>
RooAbsArg * TIteratorToSTLInterface<STLContainer>::nextChecked() {

  RooAbsArg * ret = fSTLContainer.at(fIndex);
  assert(fCurrentElem == nullptr || ret == fCurrentElem);
  fCurrentElem = ++fIndex < fSTLContainer.size() ? fSTLContainer[fIndex] : nullptr;

  return ret;
}
#endif




//ClassImp(RooLinkedListIterImpl);
