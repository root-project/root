/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id$
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
#include "RooFitCore/RooAbsArg.hh"

class RooArgProxy : public TNamed {
public:

  // Constructors, assignment etc.
  RooArgProxy() {} ;
  RooArgProxy(const char* name, RooAbsArg* owner, RooAbsArg& arg) ;
  RooArgProxy(const char* name, RooAbsArg* owner, const RooArgProxy& other) ;
  virtual ~RooArgProxy() {} ;
  inline RooAbsArg* absArg() const { return _arg ; }

protected:

  RooAbsArg* _arg ;
  friend class RooAbsArg ;
  Bool_t changePointer(RooArgSet& newServerSet) ;

  ClassDef(RooArgProxy,0) // not persistable 
};

#endif

