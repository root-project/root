/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooPlot.rdl,v 1.4 2001/04/14 00:43:20 davidk Exp $
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
protected:  
  ClassDef(RooList,1) // A TList with extra support for Option_t associations
};

#endif
