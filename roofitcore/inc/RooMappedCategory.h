/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitTools
 *    File: $Id: RooMappedCategory.rdl,v 1.1 2001/03/17 00:09:29 verkerke Exp $
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
  Bool_t setDefault(char* def_key)  ;	
  Bool_t mapValue(int   in,     int   out) ; 
  Bool_t mapValue(char* in_key, int   out) ; 
  Bool_t mapValue(int   in,     char* out_key) ; 
  Bool_t mapValue(char* in_key, char* out_key) ; 

  Bool_t mapRange(char* inlo_key, char* inhi_key, int   out) ;
  Bool_t mapRange(int   inlo,     int   inhi,     char* out) ;
  Bool_t mapRange(char* inlo_key, char* inhi_key, char* out_key) ;
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
