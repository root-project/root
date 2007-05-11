/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooAbsCollection.rdl,v 1.24 2005/12/08 13:19:54 wverkerke Exp $
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
#ifndef ROO_ABS_COLLECTION
#define ROO_ABS_COLLECTION

#include "TString.h"
#include "RooAbsArg.h"
#include "RooPrintable.h"
#include "RooLinkedList.h"
#include "RooCmdArg.h"

class RooAbsCollection : public TObject, public RooPrintable {
public:

  // Constructors, assignment etc.
  RooAbsCollection();
  RooAbsCollection(const char *name);
  virtual TObject* clone(const char* newname) const = 0 ;
  virtual TObject* create(const char* newname) const = 0 ;
  virtual TObject* Clone(const char* newname=0) const { 
    return clone(newname?newname:GetName()) ; 
  }
  virtual ~RooAbsCollection();

  // Create a copy of an existing list. New variables cannot be added
  // to a copied list. The variables in the copied list are independent
  // of the original variables.
  RooAbsCollection(const RooAbsCollection& other, const char *name="");
  RooAbsCollection& operator=(const RooAbsCollection& other);

  // Copy list and contents (and optionally 'deep' servers)
  RooAbsCollection *snapshot(Bool_t deepCopy=kTRUE) const ;

  // Hash table control
  void setHashTableSize(Int_t i) { _list.setHashTableSize(i) ; }
  Int_t getHashTableSize() const { return _list.getHashTableSize() ; }

  // List content management
  virtual Bool_t add(const RooAbsArg& var, Bool_t silent=kFALSE) ;
  virtual Bool_t addOwned(RooAbsArg& var, Bool_t silent=kFALSE);
  virtual RooAbsArg *addClone(const RooAbsArg& var, Bool_t silent=kFALSE) ;
  virtual Bool_t replace(const RooAbsArg& var1, const RooAbsArg& var2) ;
  virtual Bool_t remove(const RooAbsArg& var, Bool_t silent=kFALSE, Bool_t matchByNameOnly=kFALSE) ;
  virtual void removeAll() ;

  virtual Bool_t add(const RooAbsCollection& list, Bool_t silent=kFALSE) ;
  virtual Bool_t addOwned(const RooAbsCollection& list, Bool_t silent=kFALSE);
  virtual void   addClone(const RooAbsCollection& list, Bool_t silent=kFALSE);
  Bool_t replace(const RooAbsCollection &other);
  Bool_t remove(const RooAbsCollection& list, Bool_t silent=kFALSE, Bool_t matchByNameOnly=kFALSE) ;

  // Group operations on AbsArgs
  void setAttribAll(const Text_t* name, Bool_t value=kTRUE) ;

  // List search methods
  RooAbsArg *find(const char *name) const ;
  Bool_t contains(const RooAbsArg& var) const { return (0 == find(var.GetName())) ? kFALSE:kTRUE; }
  RooAbsCollection* selectByAttrib(const char* name, Bool_t value) const ;
  RooAbsCollection* selectCommon(const RooAbsCollection& refColl) const ;
  RooAbsCollection* selectByName(const char* nameList, Bool_t verbose=kFALSE) const ;
  Bool_t equals(const RooAbsCollection& otherColl) const ; 
  Bool_t overlaps(const RooAbsCollection& otherColl) const ;

  // export subset of THashList interface
  inline TIterator* createIterator(Bool_t dir = kIterForward) const { return _list.MakeIterator(dir); }
  inline Int_t getSize() const { return _list.GetSize(); }
  // first() returns the first element that the iterator would return, which is not necessarily
  // the first element added to this set!
  inline RooAbsArg *first() const { return (RooAbsArg*)_list.First(); }

  // Printing interface (human readable)
  virtual void printToStream(ostream& os, PrintOption opt= Standard, TString indent= "") const;
  inline virtual void Print(Option_t *options= 0) const {
    printToStream(defaultStream(),parseOptions(options));
  }

  // Latex printing methods
  void printLatex(const RooCmdArg& arg1=RooCmdArg(), const RooCmdArg& arg2=RooCmdArg(),	
		  const RooCmdArg& arg3=RooCmdArg(), const RooCmdArg& arg4=RooCmdArg(),	
		  const RooCmdArg& arg5=RooCmdArg(), const RooCmdArg& arg6=RooCmdArg(),	
		  const RooCmdArg& arg7=RooCmdArg(), const RooCmdArg& arg8=RooCmdArg()) const ;
  void printLatex(ostream& ofs, Int_t ncol, const char* option="NEYU", Int_t sigDigit=1, 
                  const RooLinkedList& siblingLists=RooLinkedList(), const RooCmdArg* formatCmd=0) const ;

  void setName(const char *name) { _name= name; }
  const char* GetName() const { return _name.Data() ; }
  Bool_t isOwning() const { return _ownCont ; }

  Bool_t allInRange(const char* rangeSpec) const ;

  void dump() const ;

protected:

  friend class RooMultiCatIter ;
  //inline const TCollection &getCollection() const { return _list; }

  RooLinkedList _list ; // Actual object store
  //THashList _list ; // Actual object store

  Bool_t _ownCont;  // Flag to identify a list that owns its contents.
  TString _name;    // Our name.

  void safeDeleteList() ;

  // Support for snapshot method 
  Bool_t addServerClonesToList(const RooAbsArg& var) ;

private:

  ClassDef(RooAbsCollection,1) // Collection of RooAbsArg objects
};

#endif
