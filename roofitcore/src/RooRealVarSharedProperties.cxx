/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooRealVarSharedProperties.cc,v 1.1 2005/12/01 16:10:20 wverkerke Exp $
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

#include "RooFitCore/RooFit.hh"
#include "RooFitCore/RooRealVarSharedProperties.hh"

ClassImp(RooRealVarSharedProperties)
;


RooRealVarSharedProperties::RooRealVarSharedProperties() 
{
} 

RooRealVarSharedProperties::RooRealVarSharedProperties(const char* uuidstr) : RooSharedProperties(uuidstr)
{
} 


RooRealVarSharedProperties::~RooRealVarSharedProperties() 
{
  _altBinning.Delete() ;
} 


