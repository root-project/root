/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooArgProxy.rdl,v 1.4 2001/05/11 06:29:59 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/
#ifndef ROO_ARG_PROXY
#define ROO_ARG_PROXY

#include "TObject.h"
#include "RooFitCore/RooAbsProxy.hh"
#include "RooFitCore/RooAbsArg.hh"

class RooArgProxy : public RooAbsProxy {
public:

  // Constructors, assignment etc.
  RooArgProxy() {} ;
  RooArgProxy(const char* name, const char* desc, RooAbsArg* owner, RooAbsArg& arg, 
	      Bool_t valueServer, Bool_t shapeServer) ;
  RooArgProxy(const char* name, RooAbsArg* owner, const RooArgProxy& other) ;
  virtual ~RooArgProxy() {} ;
  inline RooAbsArg* absArg() const { return _arg ; }

protected:

  RooAbsArg* _arg ;
  friend class RooAbsArg ;
  friend class RooAbsPdf ;
  virtual Bool_t changePointer(const RooArgSet& newServerSet) ;

  ClassDef(RooArgProxy,0) // Abstract proxy for RooAbsArg objects
};

#endif

