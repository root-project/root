/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooAbsCollection.rdl,v 1.6 2001/10/11 01:28:49 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/
#ifndef ROO_ABS_COLLECTION
#define ROO_ABS_COLLECTION

#include "THashList.h"
#include "TString.h"
#include "TClass.h"
#include "RooFitCore/RooAbsArg.hh"
#include "RooFitCore/RooPrintable.hh"

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

  // List content management
  virtual Bool_t add(const RooAbsArg& var, Bool_t silent=kFALSE) ;
  virtual Bool_t addOwned(RooAbsArg& var, Bool_t silent=kFALSE);
  virtual RooAbsArg *addClone(const RooAbsArg& var, Bool_t silent=kFALSE) ;
  virtual Bool_t replace(const RooAbsArg& var1, const RooAbsArg& var2) ;
  virtual Bool_t remove(const RooAbsArg& var, Bool_t silent=kFALSE, Bool_t matchByNameOnly=kFALSE) ;
  virtual void removeAll() ;

  virtual Bool_t add(const RooAbsCollection& list, Bool_t silent=kFALSE) ;
  virtual Bool_t addOwned(const RooAbsCollection& list, Bool_t silent=kFALSE);
  Bool_t replace(const RooAbsCollection &other);
  Bool_t remove(const RooAbsCollection& list, Bool_t silent=kFALSE, Bool_t matchByNameOnly=kFALSE) ;

  // Group operations on AbsArgs
  void setAttribAll(const Text_t* name, Bool_t value=kTRUE) ;

  // List search methods
  RooAbsArg *find(const char *name) const ;
  Bool_t contains(const RooAbsArg& var) const { return (0 == find(var.GetName())) ? kFALSE:kTRUE; }
  RooAbsCollection* selectByAttrib(const char* name, Bool_t value) const ;
  RooAbsCollection* selectCommon(const RooAbsCollection& refColl) const ;

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

  void setName(const char *name) { _name= name; }
  const char* GetName() const { return _name.Data() ; }
  Bool_t isOwning() const { return _ownCont ; }

protected:

  friend class RooMultiCatIter ;
  inline const TCollection &getCollection() const { return _list; }
     
  THashList _list ; // Actual object store

  Bool_t _ownCont;  // Flag to identify a list that owns its contents.
  TString _name;    // Our name.

  void safeDeleteList() ;

  // Support for snapshot method 
  void addServerClonesToList(const RooAbsArg& var) ;

private:

  ClassDef(RooAbsCollection,1) // Collection of RooAbsArg objects
};

#endif
