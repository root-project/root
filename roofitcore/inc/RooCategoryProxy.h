/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooCategoryProxy.rdl,v 1.1 2001/03/27 01:20:19 verkerke Exp $
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

#include "RooFitCore/RooAbsCategory.hh"
#include "RooFitCore/RooArgProxy.hh"

class RooCategoryProxy : public RooArgProxy {
public:

  // Constructors, assignment etc.
  RooCategoryProxy() {} ;
  RooCategoryProxy(const char* name, RooAbsArg* owner, RooAbsCategory& ref) ;
  RooCategoryProxy(const char* name, RooAbsArg* owner, const RooCategoryProxy& other) ;
  virtual ~RooCategoryProxy();

  // Accessors
  inline operator Int_t() const { return arg().getIndex() ; }
  inline operator const char*() const { return arg().getLabel() ; }
  inline const RooAbsCategory& arg() const { return (RooAbsCategory&)*_arg ; }

protected:

  ClassDef(RooCategoryProxy,0) // not persistable 
};

#endif
