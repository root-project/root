/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooCategoryProxy.rdl,v 1.19 2005/12/08 13:19:54 wverkerke Exp $
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
#ifndef ROO_CATEGORY_PROXY
#define ROO_CATEGORY_PROXY

#include "RooAbsCategory.h"
#include "RooArgProxy.h"
#include "RooAbsCategoryLValue.h"

class RooCategoryProxy : public RooArgProxy {
public:

  // Constructors, assignment etc.
  RooCategoryProxy() {} ;
  RooCategoryProxy(const char* name, const char* desc, RooAbsArg* owner, RooAbsCategory& ref,
		   Bool_t valueServer=kTRUE, Bool_t shapeServer=kFALSE, Bool_t proxyOwnsArg=kFALSE) ;
  RooCategoryProxy(const char* name, RooAbsArg* owner, const RooCategoryProxy& other) ;
  virtual TObject* Clone(const char* newName=0) const { return new RooCategoryProxy(newName,_owner,*this); }
  virtual ~RooCategoryProxy();

  // Accessors
  inline operator Int_t() const { return ((RooAbsCategory*)_arg)->getIndex() ; }
  inline operator const char*() const { return ((RooAbsCategory*)_arg)->getLabel() ; }
  inline const RooAbsCategory& arg() const { return (RooAbsCategory&)*_arg ; }
  inline Bool_t hasRange(const char* rangeName) const { return arg().hasRange(rangeName) ; }

protected:

  inline RooAbsCategoryLValue* lvptr() const {
    // Assert that the held arg is an LValue
    RooAbsCategoryLValue* lvptr = dynamic_cast<RooAbsCategoryLValue*>(_arg) ;
    if (!lvptr) {
      cout << "RooCategoryProxy(" << name() << ")::INTERNAL error, expected " << _arg->GetName() << " to be an lvalue" << endl ;
      assert(0) ;
    }
    return lvptr ;
  }

public:

  // LValue operations 
  RooCategoryProxy& operator=(Int_t index) { lvptr()->setIndex(index) ; return *this ; }
  RooCategoryProxy& operator=(const char* label) { lvptr()->setLabel(label) ; return *this ; }

protected:

  ClassDef(RooCategoryProxy,0) // Proxy for a RooAbsCategory object
};

#endif
