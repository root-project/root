/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooTrace.h,v 1.16 2007/05/11 09:11:30 verkerke Exp $
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
#ifndef ROO_TRACE
#define ROO_TRACE

#include <assert.h>
#include "Riosfwd.h"
#include "RooLinkedList.h"

class RooTrace {
public:

  virtual ~RooTrace() {} ;

  static void create(const TObject* obj) ;
  static void destroy(const TObject* obj) ;
  
  static void active(Bool_t flag) ;
  static void verbose(Bool_t flag) ;

  static void dump() ;
  static void dump(ostream& os, Bool_t sinceMarked=kFALSE) ;
  static void mark() ;

protected:

  static void create2(const TObject* obj) ;
  static void destroy2(const TObject* obj) ;

  void addPad(const TObject* ref, Bool_t doPad) ;
  Bool_t removePad(const TObject* ref) ;

  static Bool_t _active ;
  static Bool_t _verbose ;
  
  static RooLinkedList _list ;
  static RooLinkedList _markList ;

  ClassDef(RooTrace,0) // Memory tracer utility for RooFit objects
};


#endif
