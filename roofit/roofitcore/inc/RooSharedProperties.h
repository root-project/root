/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooSharedProperties.h,v 1.3 2007/05/11 09:11:30 verkerke Exp $
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
#ifndef ROO_ABS_SHARED_PROPERTY
#define ROO_ABS_SHARED_PROPERTY

#include "TObject.h"
#include "TUUID.h"

class RooSharedProperties : public TObject {
public:

  RooSharedProperties() ;
  RooSharedProperties(const char* uuidstr) ;
  ~RooSharedProperties() override ;
  bool operator==(const RooSharedProperties& other) const ;

  // Copying and moving is disabled for RooSharedProperties and derived classes
  // because it is not meaningful. Instead, one should copy and move around
  // shared pointers to RooSharedProperties instances.
  RooSharedProperties(const RooSharedProperties&) = delete;
  RooSharedProperties& operator=(const RooSharedProperties&) = delete;
  RooSharedProperties(RooSharedProperties &&) = delete;
  RooSharedProperties& operator=(RooSharedProperties &&) = delete;

  void Print(Option_t* opts=nullptr) const override ;

  void increaseRefCount() { _refCount++ ; }
  void decreaseRefCount() { if (_refCount>0) _refCount-- ; }
  Int_t refCount() const { return _refCount ; }

  void setInSharedList() { _inSharedList = true ; }
  bool inSharedList() const { return _inSharedList ; }

   // Wrapper class to make the TUUID comparable for use as key type in std::map.
   class UUID {
   public:
     UUID(TUUID const& tuuid) : _uuid{tuuid} {}
     bool operator<(UUID const& other) const { return _uuid.Compare(other._uuid) < 0; }
     bool operator>(UUID const& other) const { return _uuid.Compare(other._uuid) > 0; }
     bool operator==(UUID const& other) const { return _uuid.Compare(other._uuid) == 0; }
   private:
     TUUID _uuid;
   };

   UUID uuid() const { return _uuid; }

protected:

  TUUID _uuid ; ///< Unique object ID
  Int_t _refCount ; ///<! Use count
  Int_t _inSharedList ; ///<! Is in shared list

  ClassDefOverride(RooSharedProperties,1) // Abstract interface for shared property implementations
};


#endif
