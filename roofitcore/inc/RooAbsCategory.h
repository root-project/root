/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooAbsCategory.rdl,v 1.3 2001/03/19 15:57:29 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/
#ifndef ROO_ABS_CATEGORY
#define ROO_ABS_CATEGORY

#include <iostream.h>
#include "TNamed.h"
#include "TObjArray.h"
#include "RooFitCore/RooAbsArg.hh"
#include "RooFitCore/RooCatType.hh"

class TTree ;
class RooArgSet ;
class RooDataSet ;
class Roo1DTable ;

class RooAbsCategory : public RooAbsArg {
public:
  // Constructors, assignment etc.
  RooAbsCategory() {} ;
  RooAbsCategory(const char *name, const char *title);
  RooAbsCategory(const RooAbsCategory& other) ;
  virtual ~RooAbsCategory();
  virtual RooAbsArg& operator=(RooAbsArg& other) ; 

  // Value accessors
  virtual Int_t getIndex() ;
  virtual const char* getLabel() ; 
  Bool_t operator==(Int_t index) ;
  Bool_t operator==(const char* label) ;
  
  // Type definition management
  Bool_t defineType(Int_t index, char* label) ;
  Bool_t isValidIndex(Int_t index) ;
  Bool_t isValidLabel(char* label) ;
  const RooCatType* lookupType(Int_t index, Bool_t printError=kFALSE) const ;
  const RooCatType* lookupType(const char* label, Bool_t printError=kFALSE) const ;
  TIterator* typeIterator() ;

  Roo1DTable *createTable(const char *label) ;

  // I/O streaming interface
  virtual Bool_t readFromStream(istream& is, Bool_t compact, Bool_t verbose=kFALSE) ;
  virtual void writeToStream(ostream& os, Bool_t compact) ;
  virtual void printToStream(ostream& stream, PrintOption opt=Standard) ;

  void printValue() { _value.print() ; }

protected:

  // Ordinal index representation is strictly for internal use
  Int_t getOrdinalIndex() ;
  Bool_t setOrdinalIndex(Int_t newIndex) ;

  RooCatType     _value ; // Current value
  TObjArray  _types ; // Array of allowed values

  // Function evaluation and error tracing
  RooCatType traceEval() ;
  virtual Bool_t traceEvalHook(RooCatType value) {}
  virtual RooCatType evaluate() { return RooCatType("",0) ; }

  virtual Bool_t isValid() ;
  virtual Bool_t isValid(RooCatType value) ;

  friend class RooMappedCategory ;

  ClassDef(RooAbsCategory,1) // a real-valued variable and its value
};

#endif
