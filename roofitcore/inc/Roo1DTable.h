/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: Roo1DTable.rdl,v 1.6 2001/05/03 02:15:53 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/
#ifndef ROO_1D_TABLE
#define ROO_1D_TABLE

#include <iostream.h>
#include <assert.h>
#include "TNamed.h"
#include "TObjArray.h"
#include "RooFitCore/RooTable.hh"

class Roo1DTable : public RooTable {
public:

  // Constructors, cloning and assignment
  Roo1DTable() {} ;
  virtual ~Roo1DTable();
  Roo1DTable(const char *name, const char *title, const RooAbsCategory &cat);
  Roo1DTable(const Roo1DTable& other) ;

  virtual void fill(RooAbsCategory& cat) ;

  // Printing interface (human readable)
  virtual void printToStream(ostream& stream, PrintOption opt=Standard, TString indent="") const ;

protected:

  
  TObjArray _types ;
  Int_t* _count ; //! do not persist
  Int_t _nOverflow ;

  virtual Roo1DTable& operator=(const Roo1DTable& other) { return *this ; } ; 

  ClassDef(Roo1DTable,1) // 1-dimensional table
};

#endif
