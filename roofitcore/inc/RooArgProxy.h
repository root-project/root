/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooArgProxy.rdl,v 1.7 2001/06/08 05:51:05 verkerke Exp $
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

#include "TNamed.h"
#include "RooFitCore/RooAbsProxy.hh"
#include "RooFitCore/RooAbsArg.hh"

class RooArgProxy : public RooAbsProxy, public TNamed {
public:

  // Constructors, assignment etc.
  RooArgProxy() {} ;
  RooArgProxy(const char* name, const char* desc, RooAbsArg* owner, RooAbsArg& arg, 
	      Bool_t valueServer, Bool_t shapeServer) ;
  RooArgProxy(const char* name, RooAbsArg* owner, const RooArgProxy& other) ;
  virtual ~RooArgProxy() ;
  inline RooAbsArg* absArg() const { return _arg ; }

  virtual const char* name() const { return GetName() ; }

protected:

  RooAbsArg* _owner ;
  RooAbsArg* _arg ;

  Bool_t _valueServer ;
  Bool_t _shapeServer ;

  friend class RooAbsArg ;

  inline Bool_t isValueServer() const { return _valueServer ; }
  inline Bool_t isShapeServer() const { return _shapeServer ; }
  virtual Bool_t changePointer(const RooArgSet& newServerSet) ;

  virtual void changeDataSet(const RooDataSet* newDataSet) ;

  ClassDef(RooArgProxy,0) // Abstract proxy for RooAbsArg objects
};

#endif

