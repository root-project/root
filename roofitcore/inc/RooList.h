/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooList.rdl,v 1.3 2001/05/14 22:54:21 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   20-Apr-2001 DK Created initial version
 *
 * Copyright (C) 2001 Stanford University
 *****************************************************************************/
#ifndef ROO_LIST
#define ROO_LIST

#include "TList.h"

class RooList : public TList {
public:
  inline RooList() : TList() { }
  virtual ~RooList() {} ;
  TObjOptLink *findLink(const char *name, const char *caller= 0) const;
  Bool_t moveBefore(const char *before, const char *target, const char *caller= 0);
  Bool_t moveAfter(const char *after, const char *target, const char *caller= 0);
protected:  
  ClassDef(RooList,1) // TList with extra support for Option_t associations
};

#endif
