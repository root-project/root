/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooHashTable.rdl,v 1.1 2001/11/19 07:23:56 verkerke Exp $
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   16-Oct-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/
#ifndef ROO_HASH_TABLE
#define ROO_HASH_TABLE

#include "TObject.h"

class RooAbsArg ;
class RooLinkedList ;

class RooHashTable : public TObject {
public:
  // Constructor
  RooHashTable(Int_t initSize = 17) ;
  RooHashTable(const RooHashTable& other) ;

  // Destructor
  virtual ~RooHashTable() ;

  void add(RooAbsArg* arg) ;
  Bool_t remove(RooAbsArg* arg) ;
  RooAbsArg* find(const char* name) const ;
  Bool_t replace(const RooAbsArg* oldArg, const RooAbsArg* newArg) ;
  Int_t size() const { return _size ; }

protected:  

  Int_t _usedSlots ;
  Int_t _entries ;
  Int_t _size ;
  RooLinkedList** _arr ; //! do not persist

  ClassDef(RooHashTable,1) // Hash table
};




#endif
