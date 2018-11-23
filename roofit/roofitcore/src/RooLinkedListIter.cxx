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
#include <iostream>

template<class STLContainer>
RooAbsArg * TIteratorToSTLInterface<STLContainer>::nextChecked() {
    assert(fSTLContainer->begin() <= fSTLIter && fSTLIter < fSTLContainer->end());

    RooAbsArg * ret = *fSTLIter;
//    std::cout << "It at " << fSTLIter - fSTLContainer->begin() << ": ";
//    ret->Print();
#ifndef NDEBUG
    assert(ret == fCurrentElem);
    fCurrentElem = ++fSTLIter != fSTLContainer->end() ? *fSTLIter : nullptr;
#endif
    return ret;
}

template class TIteratorToSTLInterface<std::vector<RooAbsArg*>>;


//ClassImp(RooLinkedListIterImpl);

;
