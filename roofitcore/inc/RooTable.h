/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooTable.rdl,v 1.14 2005/12/08 13:19:57 wverkerke Exp $
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

#include "Riostream.h"
#include <assert.h>
#include "TNamed.h"
#include "RooAbsCategory.h"
#include "RooPrintable.h"

class RooTable : public TNamed, public RooPrintable {
public:

  // Constructors, cloning and assignment
  RooTable() {} ;
  virtual ~RooTable() ;
  RooTable(const char *name, const char *title);
  RooTable(const RooTable& other) ;

  virtual void fill(RooAbsCategory& cat, Double_t weight=1.0) = 0 ;

  // Printing interface (human readable) WVE change to RooPrintable interface
  virtual void printToStream(ostream& stream, PrintOption opt=Standard, TString indent="") const ;

  inline virtual void Print(Option_t *options= 0) const {
    printToStream(defaultStream(),parseOptions(options));
  }

protected:

  ClassDef(RooTable,1) // Abstract interface for tables
};

#endif
