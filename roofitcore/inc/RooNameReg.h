/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooNameReg.rdl,v 1.2 2005/02/25 14:22:59 wverkerke Exp $
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
#ifndef ROO_NAME_REG
#define ROO_NAME_REG

#include "TNamed.h"
#include "RooHashTable.h"

class RooNameReg : public TNamed {
public:

  static RooNameReg& instance() ;
  virtual ~RooNameReg();
  const TNamed* constPtr(const char* stringPtr) ;
  const char* constStr(const TNamed* namePtr) ; 
  static const TNamed* ptr(const char* stringPtr) { return instance().constPtr(stringPtr) ; }
  static const char* str(const TNamed* ptr) { return instance().constStr(ptr) ; }

protected:

  static RooNameReg* _instance ;

  RooNameReg() : TNamed("RooNameReg","RooFit Name Registry"), _htable(31) {} 
  RooNameReg(const RooNameReg& other) ;

  RooHashTable _htable ; // Repository of registered names

  ClassDef(RooNameReg,1) // String name registry
};

#endif


