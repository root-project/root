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
  /// Constructor from payload parameters. Note that the first payload
  /// parameter has no default value, because otherwise the implicit creation
  /// of a RooCmdArg from `const char*` would be possible. This would cause
  /// ambiguity problems in RooFit code. It is not a problem that the first
  /// parameter always has to be given, because creating a RooCmdArg with only
  /// a name and no payload doesn't make sense anyway.
  RooCmdArg(const char* name,
            Int_t i1, Int_t i2=0,
            double d1=0.0, double d2=0.0,
            const char* s1=nullptr, const char* s2=nullptr,
            const TObject* o1=nullptr, const TObject* o2=nullptr, const RooCmdArg* ca=nullptr, const char* s3=nullptr,
            const RooArgSet* c1=nullptr, const RooArgSet* c2=nullptr) ;
  RooCmdArg(const RooCmdArg& other) ;
  RooCmdArg& operator=(const RooCmdArg& other) ;
  void addArg(const RooCmdArg& arg) ;
  void setProcessRecArgs(bool flag, bool prefix=true) {
    // If true flag this object as containing recursive arguments
    _procSubArgs = flag ;
    _prefixSubArgs = prefix ;
  }

  /// Return list of sub-arguments in this RooCmdArg
  RooLinkedList const& subArgs() const { return _argList ; }

  /// Return list of sub-arguments in this RooCmdArg
  RooLinkedList& subArgs() { return _argList ; }

  TObject* Clone(const char* newName=nullptr) const override {
    RooCmdArg* newarg = new RooCmdArg(*this) ;
    if (newName) { newarg->SetName(newName) ; }
    return newarg ;
  }

  ~RooCmdArg() override;

  static const RooCmdArg& none() ;

  const char* opcode() const {
    // Return operator code
    return strlen(GetName()) ? GetName() : nullptr ;
  }

  void setInt(Int_t idx,Int_t value) {
    _i[idx] = value ;
  }
  void setDouble(Int_t idx,double value) {
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
  /// Return double stored in slot idx
  double getDouble(Int_t idx) const {
    return _d[idx] ;
  }
  /// Return string stored in slot idx
  const char* getString(Int_t idx) const {
      return (_s[idx].size()>0) ? _s[idx].c_str() : nullptr ;
  }
  /// Return TObject stored in slot idx
  const TObject* getObject(Int_t idx) const {
    return _o[idx] ;
  }

  const RooArgSet* getSet(Int_t idx) const ;

  void Print(const char* = "") const override;

  template<class T>
  static T const& take(T && obj) {
    getNextSharedData().emplace_back(new T{std::move(obj)});
    return static_cast<T const&>(*getNextSharedData().back());
  }

  bool procSubArgs() const { return _procSubArgs; }
  bool prefixSubArgs() const { return _prefixSubArgs; }

private:

  static const RooCmdArg _none  ; ///< Static instance of null object

  // Payload
  double _d[2] ;         ///< Payload doubles
  Int_t _i[2] ;            ///< Payload integers
  std::string _s[3] ;      ///< Payload strings
  TObject* _o[2] ;         ///< Payload objects
  bool _procSubArgs ;    ///< If true argument requires recursive processing
  RooArgSet* _c ;          ///< Payload RooArgSets
  RooLinkedList _argList ; ///< Payload sub-arguments
  bool _prefixSubArgs ;  ///< Prefix sub-arguments with container name?

  using DataCollection = std::vector<std::unique_ptr<TObject>>;
  std::shared_ptr<DataCollection> _sharedData; ///<!

  // the next RooCmdArg created will take ownership of this data
  static DataCollection _nextSharedData;
  static DataCollection &getNextSharedData();

  ClassDefOverride(RooCmdArg,2) // Generic named argument container
};

#endif
