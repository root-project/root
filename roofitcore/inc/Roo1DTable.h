/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: Roo1DTable.rdl,v 1.18 2005/12/08 13:19:54 wverkerke Exp $
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
#ifndef ROO_1D_TABLE
#define ROO_1D_TABLE

#include "Riostream.h"
#include <assert.h>
#include "TObjArray.h"
#include "RooTable.h"

class Roo1DTable : public RooTable {
public:

  // Constructors, cloning and assignment
  Roo1DTable() {} ;
  virtual ~Roo1DTable();
  Roo1DTable(const char *name, const char *title, const RooAbsCategory &cat);
  Roo1DTable(const Roo1DTable& other) ;

  virtual void fill(RooAbsCategory& cat, Double_t weight=1.0) ;
  Double_t get(const char* label, Bool_t silent=kFALSE) const ;
  Double_t getFrac(const char* label, Bool_t silent=kFALSE) const ;
  Double_t getOverflow() const ;

  // Printing interface (human readable)
  virtual void printToStream(ostream& stream, PrintOption opt=Standard, TString indent="") const ;

protected:

  
  TObjArray _types ;
  Double_t* _count ; //! do not persist
  Double_t  _total ;
  Double_t  _nOverflow ;

  ClassDef(Roo1DTable,1) // 1-dimensional table
};

#endif
