/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooListProxy.rdl,v 1.1 2001/09/17 18:48:15 verkerke Exp $
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   04-Jun-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/
#ifndef ROO_LIST_PROXY
#define ROO_LIST_PROXY

#include "TObject.h"
#include "RooFitCore/RooAbsProxy.hh"
#include "RooFitCore/RooAbsArg.hh"
#include "RooFitCore/RooArgList.hh"

class RooListProxy : public RooArgList, public RooAbsProxy  {
public:

  // Constructors, assignment etc.
  RooListProxy() {} ;
  RooListProxy(const char* name, const char* desc, RooAbsArg* owner, 
	      Bool_t defValueServer=kTRUE, Bool_t defShapeServer=kFALSE) ;
  RooListProxy(const char* name, RooAbsArg* owner, const RooListProxy& other) ;
  virtual ~RooListProxy() ;

  virtual const char* name() const { return GetName() ; }

  // List content management (modified for server hooks)
  virtual Bool_t add(const RooAbsArg& var, Bool_t silent=kFALSE) ;
  virtual Bool_t add(const RooAbsCollection& list, Bool_t silent=kFALSE) { return RooAbsCollection::add(list,silent) ; }
  virtual Bool_t add(const RooAbsArg& var, Bool_t valueServer, Bool_t shapeServer, Bool_t silent) ;
  virtual Bool_t replace(const RooAbsArg& var1, const RooAbsArg& var2) ;
  virtual Bool_t remove(const RooAbsArg& var, Bool_t silent=kFALSE, Bool_t matchByNameOnly=kFALSE) ;
  virtual void removeAll() ;

  RooListProxy& operator=(const RooArgList& other) ;
  
protected:

  RooAbsArg* _owner ;
  Bool_t _defValueServer ;
  Bool_t _defShapeServer ;

  virtual Bool_t changePointer(const RooAbsCollection& newServerSet, Bool_t nameChange=kFALSE) ;

  ClassDef(RooListProxy,1) // Proxy class for a RooArgList
};

#endif

