/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooSharedProperties.rdl,v 1.1 2005/12/01 16:10:20 wverkerke Exp $
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
#ifndef ROO_ABS_SHARED_PROPERTY
#define ROO_ABS_SHARED_PROPERTY

#include "TObject.h"
#include "TUUID.h"

class RooSharedProperties : public TObject {
public:

  RooSharedProperties() ;
  RooSharedProperties(const char* uuidstr) ;
  virtual ~RooSharedProperties() ;
  Bool_t operator==(const RooSharedProperties& other) ;

  virtual void Print(Option_t* opts=0) const ;

protected:

  TUUID _uuid ; // Unique object ID

  ClassDef(RooSharedProperties,1) // Abstract interface for shared property implementations
};


#endif
