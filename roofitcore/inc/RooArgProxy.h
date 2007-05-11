/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooArgProxy.rdl,v 1.19 2005/06/16 09:31:26 wverkerke Exp $
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
#ifndef ROO_ARG_PROXY
#define ROO_ARG_PROXY

#include "TNamed.h"
#include "RooAbsProxy.h"
#include "RooAbsArg.h"


class RooArgProxy : public TNamed, public RooAbsProxy  {
public:

  // Constructors, assignment etc.
  RooArgProxy() : _owner(0), _arg(0) {} ;
  RooArgProxy(const char* name, const char* desc, RooAbsArg* owner, RooAbsArg& arg, 
	      Bool_t valueServer, Bool_t shapeServer, Bool_t proxyOwnsArg=kFALSE) ;
  RooArgProxy(const char* name, RooAbsArg* owner, const RooArgProxy& other) ;
  virtual ~RooArgProxy() ;
  inline RooAbsArg* absArg() const { return _arg ; }

  virtual const char* name() const { return GetName() ; }

protected:

  friend class RooSimultaneous ;
  RooAbsArg* _owner ;
  RooAbsArg* _arg ;

  Bool_t _valueServer ;
  Bool_t _shapeServer ;
  Bool_t _isFund ;
  Bool_t _ownArg ;

  friend class RooAbsArg ;

  inline Bool_t isValueServer() const { return _valueServer ; }
  inline Bool_t isShapeServer() const { return _shapeServer ; }
  virtual Bool_t changePointer(const RooAbsCollection& newServerSet, Bool_t nameChange=kFALSE) ;

  virtual void changeDataSet(const RooArgSet* newNormSet) ;

  ClassDef(RooArgProxy,0) // Abstract proxy for RooAbsArg objects
};

#endif

