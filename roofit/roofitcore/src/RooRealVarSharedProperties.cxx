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



//_____________________________________________________________________________
RooRealVarSharedProperties::RooRealVarSharedProperties() 
{
  // Default constructor
} 


//_____________________________________________________________________________
RooRealVarSharedProperties::RooRealVarSharedProperties(const char* uuidstr) : RooSharedProperties(uuidstr)
{
  // Constructor with unique-id string
} 


//_____________________________________________________________________________
RooRealVarSharedProperties::RooRealVarSharedProperties(const RooRealVarSharedProperties& other) :
  RooSharedProperties(other), 
  _altBinning(other._altBinning)
{
  // Copy constructor
}



//_____________________________________________________________________________
RooRealVarSharedProperties::~RooRealVarSharedProperties() 
{
  // Destructor

  _altBinning.Delete() ;
} 


