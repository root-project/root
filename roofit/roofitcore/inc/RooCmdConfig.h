/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooCmdConfig.h,v 1.12 2007/05/11 09:11:30 verkerke Exp $
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

#include <RooCmdArg.h>
#include <RooStringView.h>

#include <TList.h>
#include <TObjString.h>
#include <TObject.h>
#include <TString.h>

#include <string>

class RooArgSet;

class RooCmdConfig : public TObject {
public:

  RooCmdConfig(RooStringView methodName);
  RooCmdConfig(const RooCmdConfig& other) ;

  /// If flag is true verbose messaging is activated
  void setVerbose(bool flag) {
    _verbose = flag ;
  }
  /// If flag is true the processing of unrecognized RooCmdArgs
  /// is not considered an error
  void allowUndefined(bool flag=true) {
    _allowUndefined = flag ;
  }
  void defineDependency(const char* refArgName, const char* neededArgName) ;

  template<class... Args_t>
  void defineRequiredArgs(const char* first, Args_t && ... args);

  template<class... Args_t>
  void defineMutex(const char* head, Args_t && ... tail);
  void defineMutex(const char*) {} // to end the recursion of defineMutex()

  bool defineInt(const char* name, const char* argName, int intNum, int defValue=0) ;
  bool defineDouble(const char* name, const char* argName, int doubleNum, double defValue=0.0) ;
  bool defineString(const char* name, const char* argName, int stringNum, const char* defValue="",bool appendMode=false) ;
  bool defineObject(const char* name, const char* argName, int setNum, const TObject* obj=nullptr, bool isArray=false) ;
  bool defineSet(const char* name, const char* argName, int setNum, const RooArgSet* set=nullptr) ;

  bool process(const RooCmdArg& arg) ;
  template<class... Args_t>
  bool process(const RooCmdArg& arg, Args_t && ...args);
  bool process(const RooLinkedList& argList) ;
  template<typename It_t>
  bool process(It_t begin, It_t end);

  int getInt(const char* name, int defaultValue=0) const;
  double getDouble(const char* name, double defaultValue=0.0) const;
  const char* getString(const char* name, const char* defaultValue="",bool convEmptyToNull=false) const;
  TObject* getObject(const char* name, TObject* obj=nullptr) const;
  RooArgSet* getSet(const char* name, RooArgSet* set=nullptr) const;
  const RooLinkedList& getObjectList(const char* name) const;

  bool ok(bool verbose) const ;

  std::string missingArgs() const ;

  RooLinkedList filterCmdList(RooLinkedList& cmdInList, const char* cmdNameList, bool removeFromInList=true) const;
  static void stripCmdList(RooLinkedList& cmdList, const char* cmdsToPurge);
  bool hasProcessed(const char* cmdName) const ;

  void print() const;


  template<class ...Args_t>
  static int decodeIntOnTheFly(
          const char* callerID, const char* cmdArgName, int intIdx, int defVal, Args_t && ...args);

  template<class ...Args_t>
  static std::string decodeStringOnTheFly(
          const char* callerID, const char* cmdArgName, int intIdx, const char* defVal, Args_t && ...args);

  template<class ...Args_t>
  static TObject* decodeObjOnTheFly(
          const char* callerID, const char* cmdArgName, int objIdx, TObject* defVal, Args_t && ...args);

  template<class ...Args_t>
  static RooArgSet* decodeSetOnTheFly(
          const char* callerID, const char* cmdArgName, int objIdx, RooArgSet* defVal, Args_t && ...args);

  static double decodeDoubleOnTheFly(const char* callerID, const char* cmdArgName, int idx, double defVal,
      std::initializer_list<std::reference_wrapper<const RooCmdArg>> args);

protected:

  template<class T>
  struct Var {
    std::string name;
    std::string argName;
    T val;
    bool appendMode;
    int num;
  };

  std::string _name;

  bool _verbose = false;
  bool _error = false;
  bool _allowUndefined = false;

  std::vector<Var<int>> _iList ; ///< Integer list
  std::vector<Var<double>> _dList ; ///< Double list
  std::vector<Var<std::string>> _sList ; ///< String list
  std::vector<Var<RooLinkedList>> _oList ; ///< Object list
  std::vector<Var<RooArgSet*>> _cList ; ///< RooArgSet list

  TList _rList ; ///< Required cmd list
  TList _fList ; ///< Forbidden cmd list
  TList _mList ; ///< Mutex cmd list
  TList _yList ; ///< Dependency cmd list
  TList _pList ; ///< Processed cmd list

  ClassDefOverride(RooCmdConfig,0) // Configurable parse of RooCmdArg objects
};


////////////////////////////////////////////////////////////////////////////////
/// Add condition that any of listed arguments must be processed
/// for parsing to be declared successful
template<class... Args_t>
void RooCmdConfig::defineRequiredArgs(const char* first, Args_t && ... args) {
  for(auto const& arg : {first, args...}) {
      if (arg) _rList.Add(new TObjString(arg));
  }
}


//////////////////////////////////////////////////////////////////////////////////
/// Define arguments where any pair is mutually exclusive
template<class... Args_t>
void RooCmdConfig::defineMutex(const char* head, Args_t && ... tail) {
  for(auto const& item : {tail...}) {
    _mList.Add(new TNamed(head,item));
    _mList.Add(new TNamed(item,head));
  }
  defineMutex(tail...);
}


////////////////////////////////////////////////////////////////////////////////
/// Process given RooCmdArgs
template<class... Args_t>
bool RooCmdConfig::process(const RooCmdArg& arg, Args_t && ...args) {
  bool result = false;
  for(auto r : {process(arg), process(std::forward<Args_t>(args))...}) result |= r;
  return result;
}


////////////////////////////////////////////////////////////////////////////////
/// Process several RooCmdArg using iterators.
template<typename It_t>
bool RooCmdConfig::process(It_t begin, It_t end) {
  bool result = false;
  for (auto it = begin; it != end; ++it) {
    result |= process(*it);
  }
  return result;
}


////////////////////////////////////////////////////////////////////////////////
/// Static decoder function allows to retrieve integer property from set of RooCmdArgs
/// For use in base member initializers in constructors

template<class ...Args_t>
int RooCmdConfig::decodeIntOnTheFly(
        const char* callerID, const char* cmdArgName, int intIdx, int defVal, Args_t && ...args)
{
  RooCmdConfig pc(callerID) ;
  pc.allowUndefined() ;
  pc.defineInt("theInt",cmdArgName,intIdx,defVal) ;
  pc.process(std::forward<Args_t>(args)...);
  return pc.getInt("theInt") ;
}


////////////////////////////////////////////////////////////////////////////////
/// Static decoder function allows to retrieve string property from set of RooCmdArgs
/// For use in base member initializers in constructors

template<class ...Args_t>
std::string RooCmdConfig::decodeStringOnTheFly(
        const char* callerID, const char* cmdArgName, int strIdx, const char* defVal, Args_t && ...args)
{
  RooCmdConfig pc(callerID) ;
  pc.allowUndefined() ;
  pc.defineString("theString",cmdArgName,strIdx,defVal) ;
  pc.process(std::forward<Args_t>(args)...);
  const char* ret =  pc.getString("theString",nullptr,true) ;

  return ret ? ret : "";
}


////////////////////////////////////////////////////////////////////////////////
/// Static decoder function allows to retrieve object property from set of RooCmdArgs
/// For use in base member initializers in constructors

template<class ...Args_t>
TObject* RooCmdConfig::decodeObjOnTheFly(
        const char* callerID, const char* cmdArgName, int objIdx, TObject* defVal, Args_t && ...args)
{
  RooCmdConfig pc(callerID) ;
  pc.allowUndefined() ;
  pc.defineObject("theObj",cmdArgName,objIdx,defVal) ;
  pc.process(std::forward<Args_t>(args)...);
  return pc.getObject("theObj") ;
}


template<class ...Args_t>
RooArgSet* RooCmdConfig::decodeSetOnTheFly(
        const char* callerID, const char* cmdArgName, int objIdx, RooArgSet* defVal, Args_t && ...args)
{
  RooCmdConfig pc(callerID) ;
  pc.allowUndefined() ;
  pc.defineSet("theObj",cmdArgName,objIdx,defVal) ;
  pc.process(std::forward<Args_t>(args)...);
  return pc.getSet("theObj") ;
}


#endif
