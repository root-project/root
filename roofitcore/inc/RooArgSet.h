/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooArgSet.rdl,v 1.2 2001/03/16 07:59:11 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/
#ifndef ROO_ARG_SET
#define ROO_ARG_SET

#include "THashList.h"
#include "TString.h"
#include "RooFitCore/RooAbsArg.hh"

class RooAbsArg ;

class RooArgSet : public THashList {
public:

  // Constructors, assignment etc.
  RooArgSet();
  RooArgSet(const char *name);
  RooArgSet(const char *name, RooAbsArg& var1);
  RooArgSet(const char *name, RooAbsArg& var1, RooAbsArg& var2);
  RooArgSet(const char *name, RooAbsArg& var1, RooAbsArg& var2,
	     RooAbsArg& var3);
  RooArgSet(const char *name, RooAbsArg& var1, RooAbsArg& var2,
	     RooAbsArg& var3, RooAbsArg& var4);
  RooArgSet(const char *name, RooAbsArg& var1, RooAbsArg& var2,
	     RooAbsArg& var3, RooAbsArg& var4, RooAbsArg& var5);
  RooArgSet(const char *name, RooAbsArg& var1, RooAbsArg& var2,
	     RooAbsArg& var3, RooAbsArg& var4, RooAbsArg& var5,
	     RooAbsArg& var6);
  RooArgSet(const char *name, RooAbsArg& var1, RooAbsArg& var2,
	     RooAbsArg& var3, RooAbsArg& var4, RooAbsArg& var5,
	     RooAbsArg& var6, RooAbsArg& var7);
  RooArgSet(const char *name, RooAbsArg& var1, RooAbsArg& var2,
	     RooAbsArg& var3, RooAbsArg& var4, RooAbsArg& var5,
	     RooAbsArg& var6, RooAbsArg& var7, RooAbsArg& var8);
  virtual ~RooArgSet();
  // Create a copy of an existing list. New variables cannot be added
  // to a copied list. The variables in the copied list are independent
  // of the original variables.
  RooArgSet(const char *name, const RooArgSet& other);
  RooArgSet& operator=(const RooArgSet& other);
  // Deep copy operator (copies all extern dependents into list)
  RooArgSet *snapshot() ;

  // List content management
  Bool_t add(RooAbsArg& var) ;
  Bool_t add(RooArgSet& list) { return kTRUE ; }
  Bool_t remove(RooAbsArg& var) ;
  Bool_t replace(RooAbsArg& var1, RooAbsArg& var2) { return kTRUE ; }

  // List search methods
  RooAbsArg *find(const char *name) const ;
  Bool_t contains(RooAbsArg& var) const { return kTRUE ; }

  // I/O streaming interface (machine readable)
  virtual Bool_t readFromStream(istream& is, Bool_t compact, Bool_t verbose=kFALSE) ;
  virtual void writeToStream(ostream& os, Bool_t compact) ;
  
  // Printing interface (human readable)
  enum PrintOption {Standard=0} ;
  virtual void printToStream(ostream& os, RooAbsArg::PrintOption opt=RooAbsArg::Standard) ;
  void print(RooAbsArg::PrintOption opt=RooAbsArg::Standard) ;

protected:

  Bool_t _isCopy; // Copied lists own contents
  TString _name;  // THashList doesn't inherit from TNamed

  // Support for snapshot method 
  void addServerClonesToList(RooAbsArg& var) ;

  ClassDef(RooArgSet,1) // a list of real-valued variables
};

#endif






