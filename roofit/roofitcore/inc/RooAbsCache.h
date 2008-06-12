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
#ifndef ROO_ABS_CACHE
#define ROO_ABS_CACHE

#include "Rtypes.h"
#include "Riosfwd.h"

class RooAbsArg ;
class RooAbsCollection ;
class RooArgSet ;
class RooArgList ;
class RooLinkedList ;

class RooAbsCache {

public:

  RooAbsCache(RooAbsArg* owner=0) ;
  RooAbsCache(const RooAbsCache&, RooAbsArg* owner=0 ) ;
  virtual Bool_t redirectServersHook(const RooAbsCollection& /*newServerList*/, Bool_t /*mustReplaceAll*/, Bool_t /*nameChange*/, Bool_t /*isRecursive*/) ;
  virtual void operModeHook() ;
  virtual void optimizeCacheMode(const RooArgSet&, RooArgSet&, RooLinkedList& ) ;
  virtual void findConstantNodes(const RooArgSet&, RooArgSet& , RooLinkedList&) ;
  virtual void printCompactTreeHook(ostream&, const char *) ;

  virtual ~RooAbsCache() ;
   
protected:

  RooAbsArg* _owner ; // Pointer to owning RooAbsArg

  ClassDef(RooAbsCache,1) // Base class for cache managers 

} ;


#endif 
