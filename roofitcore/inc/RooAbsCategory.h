/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooAbsCategory.rdl,v 1.7 2001/03/29 22:37:39 verkerke Exp $
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
  RooAbsCategory(const char* name, const RooAbsCategory& other) ;
  virtual ~RooAbsCategory();
  
  // Value accessors
  virtual Int_t getIndex() const ;
  virtual const char* getLabel() const ; 
  Bool_t operator==(Int_t index) const ;
  Bool_t operator==(const char* label) const ;
  
  // Type definition management
  Bool_t defineType(Int_t index, const char* label) ;
  Bool_t isValidIndex(Int_t index) const ;
  Bool_t isValidLabel(const char* label) const ;  
  const RooCatType* lookupType(Int_t index, Bool_t printError=kFALSE) const ;
  const RooCatType* lookupType(const char* label, Bool_t printError=kFALSE) const ;
  TIterator* typeIterator() const ;

  Roo1DTable *createTable(const char *label) const ;

  // I/O streaming interface
  virtual Bool_t readFromStream(istream& is, Bool_t compact, Bool_t verbose=kFALSE) ;
  virtual void writeToStream(ostream& os, Bool_t compact) const ;
  virtual void printToStream(ostream& stream, PrintOption opt=Standard) const ;

protected:

  RooAbsCategory& operator=(const RooAbsCategory& other) ; 
  virtual RooAbsArg& operator=(const RooAbsArg& other) ; 
  void initCopy(const RooAbsCategory& other) ;

  // Ordinal index representation is strictly for internal use
  Int_t getOrdinalIndex() const ;
  Bool_t setOrdinalIndex(Int_t newIndex) ;

  mutable RooCatType _value ; // Current value
  TObjArray  _types ; // Array of allowed values

  // Function evaluation and error tracing
  RooCatType traceEval() const ;
  virtual Bool_t traceEvalHook(RooCatType value) const {}
  virtual RooCatType evaluate() const { return RooCatType("",0) ; }

  virtual Bool_t isValid() const ;
  virtual Bool_t isValid(RooCatType value) const ;

  friend class RooMappedCategory ;

  ClassDef(RooAbsCategory,1) // a real-valued variable and its value
};

#endif
