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

class RooAbsArg ;
class RooAbsCollection ;
class RooArgSet ;
class RooArgList ;
class RooLinkedList ;

class RooAbsCache {

public:

  RooAbsCache(RooAbsArg* owner=nullptr) ;

  RooAbsCache(const RooAbsCache&, RooAbsArg* owner=nullptr ) ;

  virtual ~RooAbsCache() ;

  void setOwner(RooAbsArg* owner);

  /// Interface for server redirect calls.
  virtual bool redirectServersHook(const RooAbsCollection& /*newServerList*/,
                                     bool /*mustReplaceAll*/,
                                     bool /*nameChange*/,
                                     bool /*isRecursive*/) { return false; }

  /// Interface for operation mode changes.
  virtual void operModeHook() {}

  /// Interface for processing of cache mode optimization calls.
  virtual void optimizeCacheMode(const RooArgSet&, RooArgSet&, RooLinkedList& ) {}

  /// Interface for constant term node finding calls.
  virtual void findConstantNodes(const RooArgSet&, RooArgSet& , RooLinkedList&) {}

  /// Interface for printing of cache guts in tree mode printing.
  virtual void printCompactTreeHook(std::ostream&, const char *) {}

  virtual void wireCache() {}

protected:

  RooAbsArg* _owner ; ///< Pointer to owning RooAbsArg

  ClassDef(RooAbsCache,1) // Base class for cache managers

} ;


#endif
