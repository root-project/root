/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooTable.rdl,v 1.6 2001/09/12 01:25:44 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/
#ifndef ROO_TABLE
#define ROO_TABLE

#include <iostream.h>
#include <assert.h>
#include "TNamed.h"
#include "THashList.h"
#include "RooFitCore/RooAbsCategory.hh"
#include "RooFitCore/RooPrintable.hh"

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
