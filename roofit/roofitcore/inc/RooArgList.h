/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooArgList.h,v 1.14 2007/05/11 09:11:30 verkerke Exp $
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/
#ifndef ROO_ARG_LIST
#define ROO_ARG_LIST

#include "RooAbsCollection.h"

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

  inline void sort(Bool_t reverse=kFALSE) { 
    // Sort list in requested order
    _list.Sort(!reverse) ; 
  }
  inline Int_t index(const RooAbsArg* arg) const { 
    // Returns index of given arg, or -1 if arg is not in list
    return _list.IndexOf(arg) ; 
  }
  inline Int_t index(const char* name) const { 
    // Returns index of given arg, or -1 if arg is not in list
    return _list.IndexOf(name) ; 
  }
  inline RooAbsArg* at(Int_t idx) const { 
    // Return object at given index, or 0 if index is out of range
    return (RooAbsArg*) _list.At(idx) ; 
  }

  // I/O streaming interface (machine readable)
  virtual Bool_t readFromStream(std::istream& is, Bool_t compact, Bool_t verbose=kFALSE) ;
  virtual void writeToStream(std::ostream& os, Bool_t compact) ;  

  RooAbsArg& operator[](Int_t idx) const ; 

protected:

  ClassDef(RooArgList,1) // Ordered list of RooAbsArg objects
};

#endif
