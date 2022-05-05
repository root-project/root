/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooArgProxy.h,v 1.21 2007/07/12 20:30:28 wouter Exp $
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

  /// Default constructor
  RooArgProxy() : _owner(0), _arg(0), _valueServer(false), _shapeServer(false), _isFund(true), _ownArg(false) {
  } ;
  RooArgProxy(const char* name, const char* desc, RooAbsArg* owner,
         bool valueServer, bool shapeServer, bool proxyOwnsArg=false) ;
  RooArgProxy(const char* name, const char* desc, RooAbsArg* owner, RooAbsArg& arg,
         bool valueServer, bool shapeServer, bool proxyOwnsArg=false) ;
  RooArgProxy(const char* name, RooAbsArg* owner, const RooArgProxy& other) ;
  ~RooArgProxy() override ;

  /// Return pointer to contained argument
  inline RooAbsArg* absArg() const {
    return _arg ;
  }

  /// Return name of proxy
  const char* name() const override {
    return GetName() ;
  }
  void print(std::ostream& os, bool addContents=false) const override ;

protected:

  friend class RooSimultaneous ;
  RooAbsArg* _owner ;       ///< Pointer to owner of proxy
  RooAbsArg* _arg ;         ///< Pointer to content of proxy

  bool _valueServer ;     ///< If true contents is value server of owner
  bool _shapeServer ;     ///< If true contents is shape server of owner
  bool _isFund ;          ///< If true proxy contains an lvalue
  bool _ownArg ;          ///< If true proxy owns contents

  friend class RooAbsArg ;

  /// Returns true of contents is value server of owner
  inline bool isValueServer() const {
    return _valueServer ;
  }
  /// Returns true if contents is shape server of owner
  inline bool isShapeServer() const {
    return _shapeServer ;
  }
  bool changePointer(const RooAbsCollection& newServerSet, bool nameChange=false, bool factoryInitMode=false) override ;

  virtual void changeDataSet(const RooArgSet* newNormSet) ;

  ClassDefOverride(RooArgProxy,1) // Abstract proxy for RooAbsArg objects
};

#endif

