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
#ifndef ROO_ARG_SET
#define ROO_ARG_SET

#include "THashList.h"
#include "TString.h"
#include "RooFitCore/RooAbsArg.hh"

class RooAbsArg ;

class RooArgSet : public THashList {
public:
  RooArgSet();
  virtual ~RooArgSet();

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

  // Create a copy of an existing list. New variables cannot be added
  // to a copied list. The variables in the copied list are independent
  // of the original variables.
  RooArgSet(const char *name, const RooArgSet& other);

  Bool_t add(RooAbsArg& var) ;
  Bool_t add(RooArgSet& list) { return kTRUE ; }
  Bool_t remove(RooAbsArg& var) ;
  RooAbsArg *find(const char *name) const ;
  Bool_t contains(RooAbsArg& var) const { return kTRUE ; }
  Bool_t replace(RooAbsArg& var1, RooAbsArg& var2) { return kTRUE ; }
  RooArgSet *snapshot() ;

  // Return the number of non-constant (ie, variable) members in this list
  Int_t GetNVar() const { return 0 ; }

  // Initialize this list's values from variables with the same name in
  // another list. Also copy const-ness of variables.
  RooArgSet& operator=(const RooArgSet& other);

  virtual void Print(Option_t* = 0) ;

protected:
  Bool_t _isCopy;
  TString _name;

  void addServerClonesToList(RooAbsArg& var) ;

  ClassDef(RooArgSet,1) // a list of real-valued variables
};

#endif






