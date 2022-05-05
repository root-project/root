/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooTable.h,v 1.15 2007/05/11 09:11:30 verkerke Exp $
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
#ifndef ROO_TABLE
#define ROO_TABLE

#include "TNamed.h"
#include "RooAbsCategory.h"
#include "RooPrintable.h"

class RooTable : public TNamed, public RooPrintable {
public:

  // Constructors, cloning and assignment
  RooTable() {} ;
  ~RooTable() override ;
  RooTable(const char *name, const char *title);
  RooTable(const RooTable& other) ;

  virtual void fill(RooAbsCategory& cat, Double_t weight=1.0) = 0 ;

  virtual bool isIdentical(const RooTable& other, bool verbose) = 0 ;

protected:

  ClassDefOverride(RooTable,1) // Abstract interface for tables
};

#endif
