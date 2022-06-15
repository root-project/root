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
\file RooFracRemainder.cxx
\class RooFracRemainder
\ingroup Roofitcore


RooFracRemainder calculates the remainder fraction of a sum of RooAbsReal
fraction, i.e (1 - sum_i a_i). This class is used by RooSimWSTool to
as specialization of the remainder fraction term of a parameter with
a constrained split
**/


#include "Riostream.h"
#include <math.h>

#include "RooFracRemainder.h"
#include "RooAbsReal.h"
#include "RooAbsPdf.h"
#include "RooErrorHandler.h"
#include "RooArgSet.h"
#include "RooMsgService.h"

using namespace std;

ClassImp(RooFracRemainder);


////////////////////////////////////////////////////////////////////////////////
/// Constructor with given set of input fractions. All arguments in sumSet must be of type RooAbsReal.

RooFracRemainder::RooFracRemainder(const char* name, const char* title, const RooArgSet& sumSet) :
  RooAbsReal(name, title),
  _set1("set1","First set of components",this)
{
  for(RooAbsArg * comp : sumSet) {
    if (!dynamic_cast<RooAbsReal*>(comp)) {
      coutE(InputArguments) << "RooFracRemainder::ctor(" << GetName() << ") ERROR: component " << comp->GetName()
             << " is not of type RooAbsReal" << endl ;
      RooErrorHandler::softAbort() ;
    }
    _set1.add(*comp) ;
  }
}


////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

RooFracRemainder::RooFracRemainder(const RooFracRemainder& other, const char* name) :
  RooAbsReal(other, name),
  _set1("set1",this,other._set1)
{
  // Member _ownedList is intentionally not copy-constructed -- ownership is not transferred
}


////////////////////////////////////////////////////////////////////////////////
/// Calculate value

double RooFracRemainder::evaluate() const
{
  double sum(1);
  const RooArgSet* nset = _set1.nset() ;

  for (auto * comp : static_range_cast<RooAbsReal*>(_set1)) {
    sum -= comp->getVal(nset) ;
  }

  return sum ;
}
