/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooRefCountList.h,v 1.7 2007/05/11 09:11:30 verkerke Exp $
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
#ifndef ROO_REF_COUNT_LIST
#define ROO_REF_COUNT_LIST

#include "RooLinkedList.h"

class RooRefCountList : public RooLinkedList {
public:
  RooRefCountList() ;
  ~RooRefCountList() override {} ;

  void Add(TObject* arg) override { Add(arg,1) ; }
  void Add(TObject* obj, Int_t count) override ;
  Bool_t Remove(TObject* obj) override ;
  virtual Bool_t RemoveAll(TObject* obj) ;
  Int_t refCount(TObject* obj) const;

protected:
  ClassDefOverride(RooRefCountList,1) // RooLinkedList with reference counting
};

#endif
