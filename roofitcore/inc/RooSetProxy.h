/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooSetProxy.rdl,v 1.3 2001/06/06 00:06:39 verkerke Exp $
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   04-Jun-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/
#ifndef ROO_SET_PROXY
#define ROO_SET_PROXY

#include "TObject.h"
#include "RooFitCore/RooAbsProxy.hh"
#include "RooFitCore/RooAbsArg.hh"
#include "RooFitCore/RooArgSet.hh"

class RooSetProxy : public RooAbsProxy, public RooArgSet {
public:

  // Constructors, assignment etc.
  RooSetProxy() {} ;
  RooSetProxy(const char* name, const char* desc, RooAbsArg* owner, 
	      Bool_t defValueServer=kTRUE, Bool_t defShapeServer=kFALSE) ;
  RooSetProxy(const char* name, RooAbsArg* owner, const RooSetProxy& other) ;
  virtual ~RooSetProxy() ;

  virtual const char* name() const { return GetName() ; }

  // List content management (modified for server hooks)
  virtual Bool_t add(const RooAbsArg& var, Bool_t silent=kFALSE) ;
  virtual Bool_t add(const RooAbsArg& var, Bool_t valueServer, Bool_t shapeServer, Bool_t silent) ;
  virtual Bool_t replace(const RooAbsArg& var1, const RooAbsArg& var2) ;
  virtual Bool_t remove(const RooAbsArg& var) ;
  virtual void removeAll() ;

  RooSetProxy& operator=(const RooArgSet& other) ;
  
protected:

  RooAbsArg* _owner ;
  Bool_t _defValueServer ;
  Bool_t _defShapeServer ;

  virtual Bool_t changePointer(const RooArgSet& newServerSet) ;

  ClassDef(RooSetProxy,0) // Proxy class for a RooArgSet
};

#endif

