/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooTrace.rdl,v 1.12 2004/11/29 12:22:24 wverkerke Exp $
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

#include <iostream>
#include <assert.h>
#include "RooFitCore/RooLinkedList.hh"

class RooTrace {
public:

  inline static void create(const TObject* obj) { if (_active) create2(obj) ; }
  inline static void destroy(const TObject* obj) { if (_active) destroy2(obj) ; }
  
  inline static void active(Bool_t flag) { _active = flag ; }
  inline static void verbose(Bool_t flag) { _verbose = flag ; }
  
  static void dump(std::ostream& os=std::cout, Bool_t sinceMarked=kFALSE) ;
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

  ClassDef(RooTrace,0) // Memory tracer utility for RooFitTools objects
};


#endif
