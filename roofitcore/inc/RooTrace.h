/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooTrace.rdl,v 1.6 2001/10/19 22:19:49 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/
#ifndef ROO_TRACE
#define ROO_TRACE

#include <iostream.h>
#include <assert.h>
#include "RooFitCore/RooLinkedList.hh"

class RooTrace {
public:

  inline static void create(const TObject* obj) { if (_active) create2(obj) ; }
  inline static void destroy(const TObject* obj) { if (_active) destroy2(obj) ; }
  
  inline static void active(Bool_t flag) { _active = flag ; }
  inline static void verbose(Bool_t flag) { _verbose = flag ; }
  
  static void dump(ostream& os=cout) ;

protected:

  static void create2(const TObject* obj) ;
  static void destroy2(const TObject* obj) ;

  void addPad(const TObject* ref, Bool_t doPad) ;
  Bool_t removePad(const TObject* ref) ;

  static Bool_t _active ;
  static Bool_t _verbose ;
  
  static RooLinkedList _list ;

  ClassDef(RooTrace,0) // Memory tracer utility for RooFitTools objects
};


#endif
