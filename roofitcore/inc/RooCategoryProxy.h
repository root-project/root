/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooStreamParser.rdl,v 1.2 2001/03/22 15:31:25 verkerke Exp $
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

class RooCategoryProxy {
public:

  // Constructors, assignment etc.
  RooCategoryProxy(RooAbsCategory& ref) ;
  virtual ~RooCategoryProxy();

  // Accessors
  operator Int_t() { return _ref.getIndex() ; }
  operator const char*() { return _ref.getLabel() ; }
  const RooAbsCategory& arg() { return _ref ; }

protected:

  RooAbsCategory& _ref ;  
  ClassDef(RooCategoryProxy,0) // not persistable 
};

#endif
