/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id$                                                             *
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2002, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

#ifndef ROO_CMD_CONFIG
#define ROO_CMD_CONFIG

#include "TObject.h"
#include "TString.h"
#include "TList.h"
#include "RooFitCore/RooCmdArg.hh"
#include "RooFitCore/RooArgSet.hh"


class RooCmdConfig : public TObject {
public:

  RooCmdConfig(const char* methodName);
  RooCmdConfig(const RooCmdConfig& other) ;
  ~RooCmdConfig();

  void setVerbose(Bool_t flag) { _verbose = flag ; }

  void allowUndefined(Bool_t flag=kTRUE) { _allowUndefined = flag ; }
  void defineDependency(const char* refArgName, const char* neededArgName) ;
  void defineMutex(const char* argName1, const char* argName2) ;
  void defineRequiredArgs(const char* argName1, const char* argName2=0,
			  const char* argName3=0, const char* argName4=0,
			  const char* argName5=0, const char* argName6=0,
			  const char* argName7=0, const char* argName8=0) ;

  Bool_t defineInt(const char* name, const char* argName, Int_t intNum, Int_t defValue=0) ;
  Bool_t defineDouble(const char* name, const char* argName, Int_t doubleNum, Double_t defValue=0.) ;
  Bool_t defineString(const char* name, const char* argName, Int_t stringNum, const char* defValue="") ;
  Bool_t defineObject(const char* name, const char* argName, Int_t setNum, const TObject* obj=0) ;

  Bool_t process(const RooCmdArg& arg) ;
  Bool_t process(TList& argList) ;

  Int_t getInt(const char* name, Int_t defaultValue=0) ;
  Double_t getDouble(const char* name, Double_t defaultValue=0) ;
  const char* getString(const char* name, const char* defaultValue="") ;
  const TObject* getObject(const char* name, const TObject* obj=0) ;

  Bool_t ok(Bool_t verbose) const ;

  const char* missingArgs() const ;

  void stripCmdList(TList& cmdList, const char* cmdsToPurge) ;

  void print() ;

protected:

  TString _name ;
  
  Bool_t _verbose ;
  Bool_t _error ;
  Bool_t _allowUndefined ;

  TList _iList ; // Integer list
  TList _dList ; // Double list
  TList _sList ; // String list
  TList _oList ; // ArgSet list

  TList _rList ; // Required cmd list
  TList _fList ; // Forbidden cmd list
  TList _mList ; // Mutex cmd list 
  TList _yList ; // Dependancy cmd list
  TList _pList ; // Processed cmd list 

  TIterator* _iIter ;
  TIterator* _dIter ;
  TIterator* _sIter ;
  TIterator* _oIter ;
  TIterator* _rIter ;
  TIterator* _fIter ;
  TIterator* _mIter ;
  TIterator* _yIter ;
  TIterator* _pIter ;

  ClassDef(RooCmdConfig,0) // Method configuration holder
};

#endif


