/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitTools
 *    File: $Id$
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
#include "RooFitCore/RooAbsIndex.hh"
#include "RooFitCore/RooCat.hh"

class RooMappedCategory : public RooAbsIndex {
public:
  // Constructors etc.
  inline RooMappedCategory() { }
  RooMappedCategory(const char *name, const char *title, RooAbsIndex& inputCat);
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
  RooCat _defout ;

  inline RooAbsIndex* inputCat() const { return (RooAbsIndex*)_serverList.First() ; }
  Bool_t addMap(const RooCat* inlo, const RooCat* inhi, const RooCat* out) ;

  virtual RooCat evaluate() ; 

  ClassDef(RooMappedCategory,1) // a integer-valued category variable
};

#endif
