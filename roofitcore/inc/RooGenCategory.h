/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitTools
 *    File: $Id: RooGenCategory.rdl,v 1.1 2001/05/11 06:30:00 verkerke Exp $
 * Authors:
 *   WV, Wouter Verkerke, UCSB, verkerke@slac.stanford.edu
 * History:
 *   01-Mar-2001 WV Create initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/
#ifndef ROO_GEN_CATEGORY
#define ROO_GEN_CATEGORY

#include "TObjArray.h"
#include "TMethodCall.h"
#include "RooFitCore/RooAbsCategory.hh"
#include "RooFitCore/RooSuperCategory.hh"
#include "RooFitCore/RooCatType.hh"

class RooGenCategory : public RooAbsCategory {
public:
  // Constructors etc.
  inline RooGenCategory() { }
  RooGenCategory(const char *name, const char *title, void* userFunc, RooArgSet& catList);
  RooGenCategory(const RooGenCategory& other, const char *name=0) ;
  virtual TObject* clone() const { return new RooGenCategory(*this); }
  virtual ~RooGenCategory();

  // Printing interface (human readable)
  virtual void printToStream(ostream& os, PrintOption opt=Standard, TString indent= "") const ;

  // I/O streaming interface (machine readable)
  virtual Bool_t readFromStream(istream& is, Bool_t compact, Bool_t verbose=kFALSE) ;
  virtual void writeToStream(ostream& os, Bool_t compact) const ;

protected:

  void initialize() ;
  TString evalUserFunc(RooArgSet *vars) ;
  void updateIndexList() ;
  
  RooSuperCategory _superCat ; //  Super category of input categories
  Int_t *_map ;                //! Super-index to generic-index map

  TString      _userFuncName ; // 
  TMethodCall* _userFunc;      // User function hook
  Long_t _userArgs[1];         // 
                                 
  virtual RooCatType evaluate() const ; 
  ClassDef(RooGenCategory,1) // Index variable derived from other indeces, via an external global function
};

#endif
