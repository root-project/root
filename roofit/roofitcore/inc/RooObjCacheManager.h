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
#ifndef ROO_OBJ_CACHE_MANAGER
#define ROO_OBJ_CACHE_MANAGER

#include "Rtypes.h"

#include "RooNormSetCache.h"
#include "RooAbsReal.h"
#include "RooArgSet.h"
#include "RooArgList.h"
#include "RooAbsCache.h"
#include "RooAbsCacheElement.h"
#include "RooCacheManager.h"


class RooObjCacheManager : public RooCacheManager<RooAbsCacheElement> {

public:

  RooObjCacheManager(RooAbsArg* owner=nullptr, Int_t maxSize=2, bool clearCacheOnServerRedirect=true, bool allowOptimize=false) ;
  RooObjCacheManager(const RooObjCacheManager& other, RooAbsArg* owner=nullptr) ;
  ~RooObjCacheManager() override ;

  bool redirectServersHook(const RooAbsCollection& /*newServerList*/, bool /*mustReplaceAll*/, bool /*nameChange*/, bool /*isRecursive*/) override ;
  void operModeHook() override ;
  void optimizeCacheMode(const RooArgSet& /*obs*/, RooArgSet& /*optSet*/, RooLinkedList& /*processedNodes*/) override ;
  void printCompactTreeHook(std::ostream&, const char *) override ;
  void findConstantNodes(const RooArgSet& /*obs*/, RooArgSet& /*cacheList*/, RooLinkedList& /*processedNodes*/) override ;

  void insertObjectHook(RooAbsCacheElement&) override ;

  void sterilize() override ;

  static void doClearObsList(bool flag) { _clearObsList = flag ; }
  static bool clearObsList() { return _clearObsList ; }

  void setClearOnRedirect(bool flag) { _clearOnRedirect = flag ; }

protected:

  bool _clearOnRedirect ;
  bool _allowOptimize ;
  bool _optCacheModeSeen  ;              ///<!

  RooArgSet* _optCacheObservables ;        ///<! current optCacheObservables

  static bool _clearObsList ; ///< Clear obslist on sterilize?

  ClassDefOverride(RooObjCacheManager,3) ///< Cache manager for generic caches that contain RooAbsArg objects
} ;



#endif
