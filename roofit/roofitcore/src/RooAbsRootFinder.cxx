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
// RooAbsRootFinder is the abstract interface for finding roots of real-valued
// 1-dimensional function that implements the RooAbsFunc interface.
// END_HTML
//
//

#include "RooFit.h"

#include "RooAbsRootFinder.h"
#include "RooAbsRootFinder.h"
#include "RooAbsFunc.h"
#include "RooMsgService.h"
#include "Riostream.h"

ClassImp(RooAbsRootFinder)
;


//_____________________________________________________________________________
RooAbsRootFinder::RooAbsRootFinder(const RooAbsFunc& function) :
  _function(&function), _valid(function.isValid())
{
  // Constructor take function binding as argument
  if(_function->getDimension() != 1) {
    oocoutE((TObject*)0,Eval) << "RooAbsRootFinder:: cannot find roots for function of dimension "
			      << _function->getDimension() << endl;
    _valid= kFALSE;
  }
}
