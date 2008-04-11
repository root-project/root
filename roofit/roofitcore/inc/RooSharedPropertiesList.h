/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooSharedPropertiesList.h,v 1.2 2007/05/11 09:11:30 verkerke Exp $
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
#ifndef ROO_SHARED_PROPERTY_LIST
#define ROO_SHARED_PROPERTY_LIST

#include "TObject.h"
#include <assert.h>
#include "RooRefCountList.h"
#include "RooSharedProperties.h"

class RooSharedPropertiesList : public TObject {
public:

  RooSharedPropertiesList() ;
  virtual ~RooSharedPropertiesList() ;

  RooSharedProperties* registerProperties(RooSharedProperties*, Bool_t canDeleteIncoming=kTRUE) ;
  void unregisterProperties(RooSharedProperties*) ;

protected:

  RooRefCountList _propList ;

  ClassDef(RooSharedPropertiesList,0) // Manager for shared properties among clones of certain RooAbsArg-derived types
};


#endif
