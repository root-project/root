/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooTable.rdl,v 1.1 2001/03/17 03:47:39 verkerke Exp $
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

class RooTable : public TNamed {
public:

  // Constructors, cloning and assignment
  RooTable() {} ;
  virtual ~RooTable() ;
  RooTable(const char *name, const char *title);
  RooTable(const RooTable& other) ;

  virtual void fill(RooAbsCategory& cat) = 0 ;

  // Printing interface (human readable)
  enum PrintOption { Standard=0 } ;
  virtual void printToStream(ostream& stream, PrintOption opt=Standard) ;
  void print(PrintOption opt=Standard) { printToStream(cout,opt) ; }

protected:
  virtual RooTable& operator=(const RooTable& other) {} ; 

  ClassDef(RooTable,1) // a real-valued variable and its value
};

#endif
