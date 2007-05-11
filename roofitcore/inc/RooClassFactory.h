/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooClassFactory.rdl,v 1.1 2005/06/20 15:44:49 wverkerke Exp $
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

#ifndef ROO_CODE_FACTORY
#define ROO_CODE_FACTORY

#include "TNamed.h"
#include "RooArgSet.h"
#include "RooPrintable.h"

class RooClassFactory : public TNamed, public RooPrintable {

public:

  // Constructors, assignment etc
  RooClassFactory() ;
  virtual ~RooClassFactory() ;

  static Bool_t makePdf(const char* name, const char* argNames=0, Bool_t hasAnaInt=kFALSE, Bool_t hasIntGen=kFALSE) ;
  static Bool_t makeFunction(const char* name, const char* argNames=0, Bool_t hasAnaInt=kFALSE) ;
  static Bool_t makeClass(const char* className, const char* name, const char* argNames=0, Bool_t hasAnaInt=kFALSE, Bool_t hasIntGen=kFALSE) ;
  
protected:

  
  RooClassFactory(const RooClassFactory&) ;

  ClassDef(RooClassFactory,0) // RooFit class factory
} ;

#endif
