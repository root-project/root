/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooMPSentinel.cc,v 1.9 2005/06/16 09:31:28 wverkerke Exp $
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
// RooMPSentinel is a singleton class that keeps track of all
// parellel execution processes for goodness-of-fit calculations.
// The primary task of RooMPSentinel is to terminate all server processes
// when the main ROOT process is exiting.


#include "RooFit.h"

#include "RooMPSentinel.h"
#include "RooMPSentinel.h"
#include "RooRealMPFE.h"

ClassImp(RooMPSentinel)
  ;

RooMPSentinel::RooMPSentinel() 
{
}


RooMPSentinel::~RooMPSentinel() 
{
  TIterator *iter = _mpfeSet.createIterator() ;
  RooRealMPFE* mpfe ;
  while((mpfe=(RooRealMPFE*)iter->Next())) {
    mpfe->standby() ;
  }
  delete iter ;
}
 

void RooMPSentinel::add(RooRealMPFE& mpfe) 
{
  _mpfeSet.add(mpfe,kTRUE) ;
}


void RooMPSentinel::remove(RooRealMPFE& mpfe) 
{
  _mpfeSet.remove(mpfe,kTRUE) ;
}
