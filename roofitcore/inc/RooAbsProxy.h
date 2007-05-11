/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooAbsProxy.rdl,v 1.13 2005/06/16 09:31:24 wverkerke Exp $
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
#ifndef ROO_ABS_PROXY
#define ROO_ABS_PROXY

#include "TObject.h"
#include "RooAbsArg.h"

#ifdef _WIN32
// Turn off 'warning C4355: 'this' : used in base member initializer list'
// 
// This message will pop up for any class that initializes member proxy objects
// Including the pragma here will automatically disable that warning message
// for all such cases
#pragma warning ( disable:4355 )
#endif

class RooAbsProxy {
public:

  // Constructors, assignment etc.
  RooAbsProxy() ;
  RooAbsProxy(const char* name, const RooAbsProxy& other) ;
  virtual ~RooAbsProxy() {} ;

  virtual const char* name() const { return "dummy" ; } ;  
  inline const RooArgSet* nset() const { return _nset ; }

protected:

  RooArgSet* _nset ;

  friend class RooAbsArg ;
  virtual Bool_t changePointer(const RooAbsCollection& newServerSet, Bool_t nameChange=kFALSE) = 0 ;

  friend class RooAbsPdf ;
  virtual void changeNormSet(const RooArgSet* newNormSet) ;

  ClassDef(RooAbsProxy,0) // Abstract proxy interface
} ;

#endif

