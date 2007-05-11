/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooAbsProxy.cc,v 1.14 2005/06/16 09:31:24 wverkerke Exp $
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

#include "RooFit.h"

#include "RooAbsProxy.h"
#include "RooAbsProxy.h"
#include "RooArgSet.h"
#include "RooAbsArg.h"

// -- CLASS DESCRIPTION [AUX] --
// RooAbsProxy is the abstact interface for proxy classes.
// Proxy classes hold pointers to other RFC objects 
// and process serverRedirect changes so that the proxied
// pointers are updated accordingly on a clone or copy of
// of the owning class


ClassImp(RooAbsProxy)
;


RooAbsProxy::RooAbsProxy() : _nset(0)
{
  // Constructor with owner and proxied object
}


RooAbsProxy::RooAbsProxy(const char* /*name*/, const RooAbsProxy& other) : 
  _nset(other._nset)
{
  // Copy constructor
}


void RooAbsProxy::changeNormSet(const RooArgSet* newNormSet) 
{
  // Destructor
  _nset = (RooArgSet*) newNormSet ;
}
