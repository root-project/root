/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooCategoryProxy.rdl,v 1.7 2001/07/31 05:54:18 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/
#ifndef ROO_CATEGORY_PROXY
#define ROO_CATEGORY_PROXY

#include "TString.h"
#include "RooFitCore/RooAbsCategory.hh"
#include "RooFitCore/RooArgProxy.hh"

class RooCategoryProxy : public RooArgProxy {
public:

  // Constructors, assignment etc.
  RooCategoryProxy() {} ;
  RooCategoryProxy(const char* name, const char* desc, RooAbsArg* owner, RooAbsCategory& ref,
		   Bool_t valueServer=kTRUE, Bool_t shapeServer=kFALSE) ;
  RooCategoryProxy(const char* name, RooAbsArg* owner, const RooCategoryProxy& other) ;
  virtual TObject* Clone(const char* newName=0) const { return new RooCategoryProxy(*this); }
  virtual ~RooCategoryProxy();

  // Accessors
  inline operator Int_t() const { return ((RooAbsCategory*)_arg)->getIndex() ; }
  inline operator const char*() const { return ((RooAbsCategory*)_arg)->getLabel() ; }
  inline const RooAbsCategory& arg() const { return (RooAbsCategory&)*_arg ; }

protected:

  ClassDef(RooCategoryProxy,0) // Proxy for a RooAbsCategory object
};

#endif
