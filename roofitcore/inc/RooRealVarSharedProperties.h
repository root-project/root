/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooRealVarSharedProperties.rdl,v 1.1 2005/12/01 16:10:20 wverkerke Exp $
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
#ifndef ROO_REAL_VAR_SHARED_PROPERTY
#define ROO_REAL_VAR_SHARED_PROPERTY

#include "TObject.h"
#include "RooFitCore/RooSharedProperties.hh"
#include "RooFitCore/RooLinkedList.hh"

class RooRealVarSharedProperties : public RooSharedProperties {
public:

  RooRealVarSharedProperties() ;
  RooRealVarSharedProperties(const char* uuidstr) ;
  virtual ~RooRealVarSharedProperties() ;

protected:

  friend class RooRealVar ;

  RooLinkedList _altBinning ;  // Optional alternative ranges and binnings

  ClassDef(RooRealVarSharedProperties,1) // Shared properties of a RooRealVar clone set
};


#endif
