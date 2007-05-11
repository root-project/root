/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooCmdConfig.rdl,v 1.11 2005/02/25 14:22:54 wverkerke Exp $
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

#ifndef ROO_CMD_CONFIG
#define ROO_CMD_CONFIG

#include "TObject.h"
#include "TString.h"
#include "TList.h"
#include "RooCmdArg.h"
#include "RooArgSet.h"


class RooCmdConfig : public TObject {
public:

  RooCmdConfig(const char* methodName);
  RooCmdConfig(const RooCmdConfig& other) ;
  ~RooCmdConfig();

  void setVerbose(Bool_t flag) { _verbose = flag ; }

  void allowUndefined(Bool_t flag=kTRUE) { _allowUndefined = flag ; }
  void defineDependency(const char* refArgName, const char* neededArgName) ;
  void defineMutex(const char* argName1, const char* argName2) ;
  void defineMutex(const char* argName1, const char* argName2, const char* argName3) ;
  void defineMutex(const char* argName1, const char* argName2, const char* argName3, const char* argName4) ;
  void defineRequiredArgs(const char* argName1, const char* argName2=0,
			  const char* argName3=0, const char* argName4=0,
			  const char* argName5=0, const char* argName6=0,
			  const char* argName7=0, const char* argName8=0) ;

  Bool_t defineInt(const char* name, const char* argName, Int_t intNum, Int_t defValue=0) ;
  Bool_t defineDouble(const char* name, const char* argName, Int_t doubleNum, Double_t defValue=0.) ;
  Bool_t defineString(const char* name, const char* argName, Int_t stringNum, const char* defValue="",Bool_t appendMode=kFALSE) ;
  Bool_t defineObject(const char* name, const char* argName, Int_t setNum, const TObject* obj=0, Bool_t isArray=kFALSE) ;

  Bool_t process(const RooCmdArg& arg) ;
  Bool_t process(const RooCmdArg& arg1, const RooCmdArg& arg2, 
                 const RooCmdArg& arg3=RooCmdArg::none, const RooCmdArg& arg4=RooCmdArg::none, 
                 const RooCmdArg& arg5=RooCmdArg::none, const RooCmdArg& arg6=RooCmdArg::none, 
                 const RooCmdArg& arg7=RooCmdArg::none, const RooCmdArg& arg8=RooCmdArg::none) ;
  Bool_t process(const RooLinkedList& argList) ;

  Int_t getInt(const char* name, Int_t defaultValue=0) ;
  Double_t getDouble(const char* name, Double_t defaultValue=0) ;
  const char* getString(const char* name, const char* defaultValue="",Bool_t convEmptyToNull=kFALSE) ;
  TObject* getObject(const char* name, TObject* obj=0) ;
  const RooLinkedList& getObjectList(const char* name) ;

  Bool_t ok(Bool_t verbose) const ;

  const char* missingArgs() const ;

  void stripCmdList(RooLinkedList& cmdList, const char* cmdsToPurge) ;
  Bool_t hasProcessed(const char* cmdName) const ;

  void print() ;


  static Int_t decodeIntOnTheFly(const char* callerID, const char* cmdArgName, Int_t intIdx, Int_t defVal, const RooCmdArg& arg1, 
				 const RooCmdArg& arg2=RooCmdArg(), const RooCmdArg& arg3=RooCmdArg(), const RooCmdArg& arg4=RooCmdArg(),
				 const RooCmdArg& arg5=RooCmdArg(), const RooCmdArg& arg6=RooCmdArg(), const RooCmdArg& arg7=RooCmdArg(),
				 const RooCmdArg& arg8=RooCmdArg(), const RooCmdArg& arg9=RooCmdArg()) ;

  static const char* decodeStringOnTheFly(const char* callerID, const char* cmdArgName, Int_t intIdx, const char* defVal, const RooCmdArg& arg1, 
					 const RooCmdArg& arg2=RooCmdArg(), const RooCmdArg& arg3=RooCmdArg(), const RooCmdArg& arg4=RooCmdArg(),
					 const RooCmdArg& arg5=RooCmdArg(), const RooCmdArg& arg6=RooCmdArg(), const RooCmdArg& arg7=RooCmdArg(),
					 const RooCmdArg& arg8=RooCmdArg(), const RooCmdArg& arg9=RooCmdArg()) ;

  static TObject* decodeObjOnTheFly(const char* callerID, const char* cmdArgName, Int_t objIdx, TObject* defVal, const RooCmdArg& arg1, 
				     const RooCmdArg& arg2=RooCmdArg(), const RooCmdArg& arg3=RooCmdArg(), const RooCmdArg& arg4=RooCmdArg(),
				     const RooCmdArg& arg5=RooCmdArg(), const RooCmdArg& arg6=RooCmdArg(), const RooCmdArg& arg7=RooCmdArg(),
				     const RooCmdArg& arg8=RooCmdArg(), const RooCmdArg& arg9=RooCmdArg()) ;

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
  TList _yList ; // Dependency cmd list
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


