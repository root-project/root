/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id$
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/
#ifndef ROO_CATEGORY
#define ROO_CATEGORY

#include <iostream.h>
#include "RooFitCore/RooAbsCategory.hh"

class RooCategory : public RooAbsCategory {
public:
  // Constructor, assignment etc.
  RooCategory() {} ;
  RooCategory(const char *name, const char *title);
  RooCategory(const RooCategory& other) ;
  virtual ~RooCategory();
  virtual RooAbsArg& operator=(RooAbsArg& other) ; 

  // Value modifiers
  Bool_t setIndex(Int_t index) ;
  Bool_t setLabel(char* label) ;
  
  // Value accessors (overridden from base class)
  virtual Int_t getIndex() { return _value.getVal(); }
  virtual const char* getLabel() { return _value.GetName() ; } 

  // I/O streaming interface (machine readable)
  virtual Bool_t readFromStream(istream& is, Bool_t compact, Bool_t verbose=kFALSE) ;
  virtual void writeToStream(ostream& os, Bool_t compact) ;

  // Printing interface (human readable)
  virtual void printToStream(ostream& stream, PrintOption opt=Standard) ;

protected:

  virtual void attachToTree(TTree& t, Int_t bufSize=32000) ;

  ClassDef(RooCategory,1) // a real-valued variable and its value
};

#endif
