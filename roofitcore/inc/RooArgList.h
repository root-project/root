/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooArgList.rdl,v 1.5 2001/10/19 06:56:52 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/
#ifndef ROO_ARG_LIST
#define ROO_ARG_LIST

#include "THashList.h"
#include "TString.h"
#include "TClass.h"
#include "RooFitCore/RooAbsCollection.hh"
#include "RooFitCore/RooErrorHandler.hh"

class RooArgList : public RooAbsCollection {
public:

  // Constructors, assignment etc.
  RooArgList();
  RooArgList(const RooArgSet& set) ;
  explicit RooArgList(const TCollection& tcoll, const char* name="") ;
  explicit RooArgList(const char *name);
  RooArgList(const RooAbsArg& var1, 
	     const char *name="");
  RooArgList(const RooAbsArg& var1, const RooAbsArg& var2, 
	     const char *name="");
  RooArgList(const RooAbsArg& var1, const RooAbsArg& var2,
	     const RooAbsArg& var3, 
	     const char *name="");
  RooArgList(const RooAbsArg& var1, const RooAbsArg& var2,
	     const RooAbsArg& var3, const RooAbsArg& var4, 
	     const char *name="");
  RooArgList(const RooAbsArg& var1, const RooAbsArg& var2,
	     const RooAbsArg& var3, const RooAbsArg& var4, 
	     const RooAbsArg& var5, 
	     const char *name="");
  RooArgList(const RooAbsArg& var1, const RooAbsArg& var2,
	     const RooAbsArg& var3, const RooAbsArg& var4, 
	     const RooAbsArg& var5, const RooAbsArg& var6, 
	     const char *name="");
  RooArgList(const RooAbsArg& var1, const RooAbsArg& var2,
	     const RooAbsArg& var3, const RooAbsArg& var4, 
	     const RooAbsArg& var5, const RooAbsArg& var6, 
	     const RooAbsArg& var7, 
	     const char *name="");
  RooArgList(const RooAbsArg& var1, const RooAbsArg& var2,
	     const RooAbsArg& var3, const RooAbsArg& var4, 
	     const RooAbsArg& var5, const RooAbsArg& var6, 
	     const RooAbsArg& var7, const RooAbsArg& var8, 
	     const char *name="");
  RooArgList(const RooAbsArg& var1, const RooAbsArg& var2,
	     const RooAbsArg& var3, const RooAbsArg& var4, 
	     const RooAbsArg& var5, const RooAbsArg& var6, 
	     const RooAbsArg& var7, const RooAbsArg& var8, 
	     const RooAbsArg& var9, const char *name="");

  virtual ~RooArgList();
  // Create a copy of an existing list. New variables cannot be added
  // to a copied list. The variables in the copied list are independent
  // of the original variables.
  RooArgList(const RooArgList& other, const char *name="");
  virtual TObject* clone(const char* newname) const { return new RooArgList(*this,newname); }
  virtual TObject* create(const char* newname) const { return new RooArgList(newname); }
  RooArgList& operator=(const RooArgList& other) { RooAbsCollection::operator=(other) ; return *this ; }

  inline void sort(Bool_t reverse=kFALSE) { _list.Sort(!reverse) ; }
  inline Int_t index(const RooAbsArg* arg) const { return _list.IndexOf(arg) ; }
  inline RooAbsArg* at(Int_t idx) const { return (RooAbsArg*) _list.At(idx) ; }

  // I/O streaming interface (machine readable)
  virtual Bool_t readFromStream(istream& is, Bool_t compact, Bool_t verbose=kFALSE) ;
  virtual void writeToStream(ostream& os, Bool_t compact) ;  

  RooAbsArg& operator[](Int_t idx) const ; 

protected:

  ClassDef(RooArgList,1) // List of RooAbsArg objects
};

#endif
