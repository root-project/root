/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitTools
 *    File: $Id: RooMappedCategory.rdl,v 1.2 2001/03/17 00:32:55 verkerke Exp $
 * Authors:
 *   WV, Wouter Verkerke, UCSB, verkerke@slac.stanford.edu
 * History:
 *   01-Mar-2001 WV Create initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/
#ifndef ROO_MAPPED_CATEGORY
#define ROO_MAPPED_CATEGORY

#include "TObjArray.h"
#include "RooFitCore/RooAbsCategory.hh"
#include "RooFitCore/RooCatType.hh"

class RooMappedCategory : public RooAbsCategory {
public:
  // Constructors etc.
  inline RooMappedCategory() { }
  RooMappedCategory(const char *name, const char *title, RooAbsCategory& inputCat);
  virtual ~RooMappedCategory();

  // Mapping definition functions
  Bool_t setDefault(int def)  ;	
  Bool_t setDefault(const char* def_key)  ;	
  Bool_t mapValue(int   in,     int   out) ; 
  Bool_t mapValue(const char* in_key, int   out) ; 
  Bool_t mapValue(int   in,     const char* out_key) ; 
  Bool_t mapValue(const char* in_key, const char* out_key) ; 

  Bool_t mapRange(const char* inlo_key, const char* inhi_key, int   out) ;
  Bool_t mapRange(int   inlo,     int   inhi,     const char* out) ;
  Bool_t mapRange(const char* inlo_key, const char* inhi_key, const char* out_key) ;
  Bool_t mapRange(int   inlo,     int   inhi,     int   out) ;

  // Printing interface (human readable)
  virtual void printToStream(ostream& stream, PrintOption opt=Standard) ;

  // I/O streaming interface (machine readable)
  virtual Bool_t readFromStream(istream& is, Bool_t compact, Bool_t verbose=kFALSE) ;
  virtual void writeToStream(ostream& os, Bool_t compact) ;

protected:
  
  TObjArray _inlo ;
  TObjArray _inhi ;
  TObjArray _out ;
  RooCatType _defout ;

  inline RooAbsCategory* inputCat() const { return (RooAbsCategory*)_serverList.First() ; }
  Bool_t addMap(const RooCatType* inlo, const RooCatType* inhi, const RooCatType* out) ;

  virtual RooCatType evaluate() ; 

  ClassDef(RooMappedCategory,1) // a integer-valued category variable
};

#endif
