/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooCategory.rdl,v 1.8 2001/04/18 20:38:02 verkerke Exp $
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
  RooCategory(const RooCategory& other, const char* name=0) ;
  virtual ~RooCategory();
  virtual TObject* clone() const { return new RooCategory(*this); }
  virtual RooCategory& operator=(const RooCategory& other) ; 

  // Value modifiers
  Bool_t setIndex(Int_t index, Bool_t printError=kTRUE) ;
  Bool_t setLabel(const char* label, Bool_t printError=kTRUE) ;
  RooCategory& operator=(int index) ; 
  RooCategory& operator=(const char* label) ; 
  
  // I/O streaming interface (machine readable)
  virtual Bool_t readFromStream(istream& is, Bool_t compact, Bool_t verbose=kFALSE) ;
  virtual void writeToStream(ostream& os, Bool_t compact) const ;

protected:

  Int_t _dummy ;

  virtual RooAbsArg& operator=(const RooAbsArg& other) ; 

  virtual void attachToTree(TTree& t, Int_t bufSize=32000) ;
  virtual void postTreeLoadHook() ;

  ClassDef(RooCategory,1) // a real-valued variable and its value
};

#endif
