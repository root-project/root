/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooSetProxy.rdl,v 1.1 2001/05/11 06:30:01 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/
#ifndef ROO_SET_PROXY
#define ROO_SET_PROXY

#include "TObject.h"
#include "RooFitCore/RooAbsProxy.hh"
#include "RooFitCore/RooAbsArg.hh"

class RooSetProxy : public RooAbsProxy {
public:

  // Constructors, assignment etc.
  RooSetProxy() {} ;
  RooSetProxy(const char* name, const char* desc, RooAbsArg* owner, RooArgSet& arg,
	      Bool_t valueServer=kTRUE, Bool_t shapeServer=kFALSE) ;
  RooSetProxy(const char* name, RooAbsArg* owner, const RooSetProxy& other) ;
  virtual ~RooSetProxy() {} ;
  inline const RooArgSet* set() const { return _set ; }

protected:

  RooArgSet* _set ;
  friend class RooAbsArg ;
  friend class RooAbsPdf ;
  virtual Bool_t changePointer(const RooArgSet& newServerSet) ;

  ClassDef(RooSetProxy,0) // Proxy class for a RooArgSet
};

#endif

