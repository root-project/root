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
// Class RooRealVarSharedProperties is an implementation of RooSharedProperties
// that stores the properties of a RooRealVar that are shared among clones.
// For RooRealVars these are the definitions of the named ranges.
// END_HTML
//

#include "RooFit.h"
#include "RooRealVarSharedProperties.h"
#include "Riostream.h"


using namespace std;

ClassImp(RooRealVarSharedProperties)
;



////////////////////////////////////////////////////////////////////////////////
/// Default constructor

RooRealVarSharedProperties::RooRealVarSharedProperties() 
{
} 


////////////////////////////////////////////////////////////////////////////////
/// Constructor with unique-id string

RooRealVarSharedProperties::RooRealVarSharedProperties(const char* uuidstr) : RooSharedProperties(uuidstr)
{
} 


////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

RooRealVarSharedProperties::RooRealVarSharedProperties(const RooRealVarSharedProperties& other) :
  RooSharedProperties(other), 
  _altBinning(other._altBinning)
{
}



////////////////////////////////////////////////////////////////////////////////
/// Destructor

RooRealVarSharedProperties::~RooRealVarSharedProperties() 
{
  _altBinning.Delete() ;
} 


