/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooCategorySharedProperties.h,v 1.2 2007/05/11 09:11:30 verkerke Exp $
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

/**
\file RooCategorySharedProperties.h
\class RooCategorySharedProperties
\ingroup Roofitcore

RooCategorySharedProperties is the container for all properties
that are shared between instance of RooCategory objects that
are clones of each other. At present the only property that is
shared in this way is the list of alternate named range definitions
**/

#ifndef ROO_CATEGORY_SHARED_PROPERTY
#define ROO_CATEGORY_SHARED_PROPERTY

#include "RooSharedProperties.h"
#include "RooLinkedList.h"

class RooCategorySharedProperties : public RooSharedProperties {
public:

  /// Constructor.
  RooCategorySharedProperties() {}
  /// Constructor with unique-id string.
  RooCategorySharedProperties(const char* uuidstr) : RooSharedProperties(uuidstr) {}
  /// Destructor.
  ~RooCategorySharedProperties() override {
    _altRanges.Delete() ;
  }

protected:

  friend class RooCategory ;

  RooLinkedList _altRanges ;  ///< Optional alternative ranges

  ClassDefOverride(RooCategorySharedProperties,1) // Shared properties of a RooCategory clone set
};


#endif
