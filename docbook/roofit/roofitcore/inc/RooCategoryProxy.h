/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooCategoryProxy.h,v 1.20 2007/05/11 09:11:30 verkerke Exp $
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
  RooCategoryProxy() {
    // Default constructor
  } ;
  RooCategoryProxy(const char* name, const char* desc, RooAbsArg* owner,
		   Bool_t valueServer=kTRUE, Bool_t shapeServer=kFALSE, Bool_t proxyOwnsArg=kFALSE) ;
  RooCategoryProxy(const char* name, const char* desc, RooAbsArg* owner, RooAbsCategory& ref,
		   Bool_t valueServer=kTRUE, Bool_t shapeServer=kFALSE, Bool_t proxyOwnsArg=kFALSE) ;
  RooCategoryProxy(const char* name, RooAbsArg* owner, const RooCategoryProxy& other) ;
  virtual TObject* Clone(const char* newName=0) const { return new RooCategoryProxy(newName,_owner,*this); }
  virtual ~RooCategoryProxy();

  // Accessors
  inline operator Int_t() const { 
    // Facilitates use of proxy as integer
    return ((RooAbsCategory*)_arg)->getIndex() ; 
  }
  inline const char* label() const { 
    // Facilitates use of proxy as string value
    return ((RooAbsCategory*)_arg)->getLabel() ; 
  }
  inline const RooAbsCategory& arg() const { 
    // Return proxies argument
    return (RooAbsCategory&)*_arg ; 
  }
  inline Bool_t hasRange(const char* rangeName) const { 
    // Returns true if proxied argument has range with given name
    return arg().hasRange(rangeName) ; 
  }

  // Modifier
  virtual Bool_t setArg(RooAbsCategory& newRef) ;
			
protected:

  RooAbsCategoryLValue* lvptr() const ;

public:

  // LValue operations 
  RooCategoryProxy& operator=(Int_t index) { 
    // Assignment operator with index value
    lvptr()->setIndex(index) ; return *this ; 
  }
  RooCategoryProxy& operator=(const char* _label) { 
    // Assignment value with string value
    lvptr()->setLabel(_label) ; return *this ; 
  }

protected:

  ClassDef(RooCategoryProxy,1) // Proxy for a RooAbsCategory object
};

#endif
