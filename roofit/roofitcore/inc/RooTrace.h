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

#include "RooLinkedList.h"
#include <map>
#include <string>

#define TRACE_CREATE
#define TRACE_DESTROY

class RooTrace {
public:
  RooTrace() ;
  virtual ~RooTrace() {} ;

  static void create(const TObject* obj) ;
  static void destroy(const TObject* obj) ;


  static void createSpecial(const char* name, int size) ;
  static void destroySpecial(const char* name) ;


  static void active(bool flag) ;
  static void verbose(bool flag) ;

  static void dump() ;
  static void dump(std::ostream& os, bool sinceMarked=false) ;
  static void mark() ;

  static void callgrind_zero() ;
  static void callgrind_dump() ;


  static RooTrace& instance() ;

  static void printObjectCounts() ;


protected:

  static RooTrace* _instance ;

  void dump3(std::ostream&, bool sinceMarked) ;

  void mark3() ;
  void printObjectCounts3() ;

  void create2(const TObject* obj) ;
  void destroy2(const TObject* obj) ;

  void create3(const TObject* obj) ;
  void destroy3(const TObject* obj) ;

  void createSpecial3(const char* name, int size) ;
  void destroySpecial3(const char* name) ;

  void addPad(const TObject* ref, bool doPad) ;
  bool removePad(const TObject* ref) ;

  bool _active ;
  bool _verbose ;
  RooLinkedList _list ;
  RooLinkedList _markList ;
  std::map<TClass*,int> _objectCount ;
  std::map<std::string,int> _specialCount ;
  std::map<std::string,int> _specialSize ;

  ClassDef(RooTrace,0) // Memory tracer utility for RooFit objects
};


#endif
