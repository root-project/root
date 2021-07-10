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
\file RooSetPair.cxx
\class RooSetPair
\ingroup Roofitlegacy

RooSetPair is a utility class that stores a pair of RooArgSets
**/

#include "RooFit.h"
#include "TROOT.h"

#define ROOSETPAIR_CXX
#include "RooFitLegacy/RooSetPair.h"

using namespace std;

ClassImp(RooSetPair);

////////////////////////////////////////////////////////////////////////////////
/// RooSetPair destructor.

RooSetPair::~RooSetPair()
{
   // Required since we overload TObject::Hash.
   ROOT::CallRecursiveRemoveIfNeeded(*this);
}
