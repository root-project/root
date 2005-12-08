/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id$
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
#ifndef ROO_CATEGORY_SHARED_PROPERTY
#define ROO_CATEGORY_SHARED_PROPERTY

#include "RooFitCore/RooSharedProperties.hh"
#include "RooFitCore/RooLinkedList.hh"

class RooCategorySharedProperties : public RooSharedProperties {
public:

  RooCategorySharedProperties() ;
  virtual ~RooCategorySharedProperties() ;

protected:

  friend class RooCategory ;

  RooLinkedList _altRanges ;  // Optional alternative ranges 

  ClassDef(RooCategorySharedProperties,1) // Shared properties of a RooCategory clone set
};


#endif
