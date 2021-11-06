/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooCmdArg.h,v 1.10 2007/05/11 09:11:30 verkerke Exp $
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

#ifndef ROO_CMD_ARG
#define ROO_CMD_ARG

#include "TNamed.h"
#include "RooLinkedList.h"
#include <string>

class RooArgSet ;

class RooCmdArg final : public TNamed {
public:

  RooCmdArg();
  RooCmdArg(const char* name, 
	    Int_t i1=0, Int_t i2=0, 
	    Double_t d1=0, Double_t d2=0, 
	    const char* s1=0, const char* s2=0, 
	    const TObject* o1=0, const TObject* o2=0, const RooCmdArg* ca=0, const char* s3=0,
	    const RooArgSet* c1=0, const RooArgSet* c2=0) ;
  RooCmdArg(const RooCmdArg& other) ;
  RooCmdArg& operator=(const RooCmdArg& other) ;
  void addArg(const RooCmdArg& arg) ;
  void setProcessRecArgs(Bool_t flag, Bool_t prefix=kTRUE) { 
    // If true flag this object as containing recursive arguments
    _procSubArgs = flag ; 
    _prefixSubArgs = prefix ;
  }

  /// Return list of sub-arguments in this RooCmdArg
  RooLinkedList const& subArgs() const { return _argList ; }

  /// Return list of sub-arguments in this RooCmdArg
  RooLinkedList& subArgs() { return _argList ; }

  virtual TObject* Clone(const char* newName=0) const {
    RooCmdArg* newarg = new RooCmdArg(*this) ;
    if (newName) { newarg->SetName(newName) ; }
    return newarg ;
  }

  virtual ~RooCmdArg();

  static const RooCmdArg& none() ;

  const char* opcode() const { 
    // Return operator code
    return strlen(GetName()) ? GetName() : 0 ; 
  }

  void setInt(Int_t idx,Int_t value) {
    _i[idx] = value ;
  }
  void setDouble(Int_t idx,Double_t value) {
    _d[idx] = value ;
  }
  void setString(Int_t idx,const char* value) {
    _s[idx] = value ;
  }
  void setObject(Int_t idx,TObject* value) {
    _o[idx] = value ;
  }
  void setSet(Int_t idx,const RooArgSet& set) ;

  Int_t getInt(Int_t idx) const { 
    // Return integer stored in slot idx
    return _i[idx] ; 
  }
  Double_t getDouble(Int_t idx) const { 
    // Return double stored in slot idx
    return _d[idx] ; 
  }
  const char* getString(Int_t idx) const { 
    // Return string stored in slot idx
      return (_s[idx].size()>0) ? _s[idx].c_str() : 0 ; 
  }
  const TObject* getObject(Int_t idx) const { 
  // Return TObject stored in slot idx
    return _o[idx] ; 
  }

  const RooArgSet* getSet(Int_t idx) const ;

  void Print(const char* = "") const;

  template<class T>
  static T const& take(T && obj) {
    getNextSharedData().emplace_back(new T{std::move(obj)});
    return static_cast<T const&>(*getNextSharedData().back());
  }

  bool procSubArgs() const { return _procSubArgs; }
  bool prefixSubArgs() const { return _prefixSubArgs; }

private:

  static const RooCmdArg _none  ; // Static instance of null object

  // Payload
  Double_t _d[2] ;       // Payload doubles
  Int_t _i[2] ;          // Payload integers
  std::string _s[3] ;    // Payload strings
  TObject* _o[2] ;       // Payload objects
  Bool_t _procSubArgs ;  // If true argument requires recursive processing
  RooArgSet* _c ;        // Payload RooArgSets 
  RooLinkedList _argList ; // Payload sub-arguments
  Bool_t _prefixSubArgs ; // Prefix subarguments with container name?

  using DataCollection = std::vector<std::unique_ptr<TObject>>;
  std::shared_ptr<DataCollection> _sharedData; //!

  // the next RooCmdArg created will take ownership of this data
  static DataCollection _nextSharedData;
  static DataCollection &getNextSharedData();
  
  ClassDef(RooCmdArg,2) // Generic named argument container
};

#endif
