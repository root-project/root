/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooAbsArg.h,v 1.93 2007/07/16 21:04:28 wouter Exp $
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
#ifndef ROO_ABS_ARG
#define ROO_ABS_ARG

#include <assert.h>
#include "TNamed.h"
#include "THashList.h"
#include "TRefArray.h"
#include "RooPrintable.h"
#include "RooSTLRefCountList.h"
#include "RooAbsCache.h"
#include "RooLinkedListIter.h"
#include "RooNameReg.h"
#include <map>
#include <set>
#include <deque>
#include <stack>

#include <iostream>


#include "TClass.h"

class TTree ;
class RooArgSet ;
class RooAbsCollection ;
class RooTreeData ;
class RooTreeDataStore ;
class RooVectorDataStore ;
class RooAbsData ;
class RooAbsDataStore ;
class RooAbsProxy ;
class RooArgProxy ;
class RooSetProxy ;
class RooListProxy ;
class RooExpensiveObjectCache ;
class RooWorkspace ;
class RooRealProxy ;

class RooRefArray : public TObjArray {
 public:
  RooRefArray() : TObjArray() {
  } ;
  RooRefArray(const RooRefArray& other) : TObjArray(other) {
  }
  virtual ~RooRefArray() {} ;
 protected:
  ClassDef(RooRefArray,1) // Helper class for proxy lists
} ;




class RooAbsArg : public TNamed, public RooPrintable {
public:
  using RefCountList_t = RooSTLRefCountList<RooAbsArg>;
  using RefCountListLegacyIterator_t = TIteratorToSTLInterface<RefCountList_t::Container_t>;

  // Constructors, cloning and assignment
  RooAbsArg() ;
  virtual ~RooAbsArg();
  RooAbsArg(const char *name, const char *title);
  RooAbsArg(const RooAbsArg& other, const char* name=0) ;
  virtual TObject* clone(const char* newname=0) const = 0 ;
  virtual TObject* Clone(const char* newname=0) const {
    return clone(newname) ;
  }
  virtual RooAbsArg* cloneTree(const char* newname=0) const ;

  // Accessors to client-server relation information

  /// Does value or shape of this arg depend on any other arg?
  virtual Bool_t isDerived() const {
    return kTRUE ;
    //std::cout << IsA()->GetName() << "::isDerived(" << GetName() << ") = " << (_serverList.GetSize()>0 || _proxyList.GetSize()>0) << std::endl ;
    //return (_serverList.GetSize()>0 || _proxyList.GetSize()>0)?kTRUE:kFALSE;
  }
  Bool_t isCloneOf(const RooAbsArg& other) const ;
  Bool_t dependsOnValue(const RooAbsCollection& serverList, const RooAbsArg* ignoreArg=0) const {
    // Does this arg depend on the value of any of of the values in serverList?
    return dependsOn(serverList,ignoreArg,kTRUE) ;
  }
  Bool_t dependsOnValue(const RooAbsArg& server, const RooAbsArg* ignoreArg=0) const {
    // Does this arg depend on the value of server?
    return dependsOn(server,ignoreArg,kTRUE) ;
  }
  Bool_t dependsOn(const RooAbsCollection& serverList, const RooAbsArg* ignoreArg=0, Bool_t valueOnly=kFALSE) const ;
  Bool_t dependsOn(const RooAbsArg& server, const RooAbsArg* ignoreArg=0, Bool_t valueOnly=kFALSE) const ;
  Bool_t overlaps(const RooAbsArg& testArg, Bool_t valueOnly=kFALSE) const ;
  Bool_t hasClients() const { return !_clientList.empty(); }

  ////////////////////////////////////////////////////////////////////////////
  // Legacy iterators
  inline TIterator* clientIterator() const
  R__SUGGEST_ALTERNATIVE("Use clients() and begin(), end() or range-based loops.") {
    // Return iterator over all client RooAbsArgs
    return makeLegacyIterator(_clientList);
  }
  inline TIterator* valueClientIterator() const
  R__SUGGEST_ALTERNATIVE("Use valueClients() and begin(), end() or range-based loops.") {
    // Return iterator over all shape client RooAbsArgs
    return makeLegacyIterator(_clientListValue);
  }
  inline TIterator* shapeClientIterator() const
  R__SUGGEST_ALTERNATIVE("Use shapeClients() and begin(), end() or range-based loops.") {
    // Return iterator over all shape client RooAbsArgs
    return makeLegacyIterator(_clientListShape);
  }
  inline TIterator* serverIterator() const
  R__SUGGEST_ALTERNATIVE("Use servers() and begin(), end() or range-based loops.") {
    // Return iterator over all server RooAbsArgs
    return makeLegacyIterator(_serverList);
  }

  inline RooFIter valueClientMIterator() const
  R__SUGGEST_ALTERNATIVE("Use valueClients() and begin(), end() or range-based loops.") {
    return RooFIter(std::unique_ptr<RefCountListLegacyIterator_t>(makeLegacyIterator(_clientListValue)));
  }
  inline RooFIter shapeClientMIterator() const
  R__SUGGEST_ALTERNATIVE("Use shapeClients() and begin(), end() or range-based loops.") {
    return RooFIter(std::unique_ptr<RefCountListLegacyIterator_t>(makeLegacyIterator(_clientListShape)));
  }
  inline RooFIter serverMIterator() const
  R__SUGGEST_ALTERNATIVE("Use shapeClients() and begin(), end() or range-based loops.") {
    return RooFIter(std::unique_ptr<RefCountListLegacyIterator_t>(makeLegacyIterator(_serverList)));
  }

  ////////////////////////////////////////////////////////////////////////////

  /// List of all clients of this object.
  const RefCountList_t& clients() const {
    return _clientList;
  }
  /// List of all value clients of this object. Value clients receive value updates.
  const RefCountList_t& valueClients() const {
    return _clientListValue;
  }
  /// List of all shape clients of this object. Shape clients receive property information such as
  /// changes of a value range.
  const RefCountList_t& shapeClients() const {
    return _clientListShape;
  }

  const RefCountList_t& servers() const {
    return _serverList;
  }

  inline RooAbsArg* findServer(const char *name) const {
    // Return server of this arg with given name. Returns null if not found
    const auto serverIt = _serverList.findByName(name);
    return serverIt != _serverList.end() ? *serverIt : nullptr;
  }
  inline RooAbsArg* findServer(const RooAbsArg& arg) const {
    // Return server of this arg with name of given input arg. Returns null if not found
    const auto serverIt = _serverList.findByNamePointer(&arg);
    return serverIt != _serverList.end() ? *serverIt : nullptr;
  }
  inline RooAbsArg* findServer(Int_t index) const {
    // Return i-th server from server list
    return _serverList.containedObjects()[index];
  }
  inline Bool_t isValueServer(const RooAbsArg& arg) const {
    // If true, arg is a value server of self
    return _clientListValue.containsByNamePtr(&arg);
  }
  inline Bool_t isValueServer(const char* name) const {
    // If true, we have a server with given name
    return _clientListValue.containsSameName(name);
  }
  inline Bool_t isShapeServer(const RooAbsArg& arg) const {
    // If true arg is a shape server of self
    return _clientListShape.containsByNamePtr(&arg);
  }
  inline Bool_t isShapeServer(const char* name) const {
    // If true, we have a shape server with given name
    return _clientListShape.containsSameName(name);
  }
  void leafNodeServerList(RooAbsCollection* list, const RooAbsArg* arg=0, Bool_t recurseNonDerived=kFALSE) const ;
  void branchNodeServerList(RooAbsCollection* list, const RooAbsArg* arg=0, Bool_t recurseNonDerived=kFALSE) const ;
  void treeNodeServerList(RooAbsCollection* list, const RooAbsArg* arg=0,
			  Bool_t doBranch=kTRUE, Bool_t doLeaf=kTRUE,
			  Bool_t valueOnly=kFALSE, Bool_t recurseNonDerived=kFALSE) const ;


  /// Is this object a fundamental type that can be added to a dataset?
  /// Fundamental-type subclasses override this method to return kTRUE.
  /// Note that this test is subtlely different from the dynamic isDerived()
  /// test, e.g. a constant is not derived but is also not fundamental.
  inline virtual Bool_t isFundamental() const {
    return kFALSE;
  }

  /// Create a fundamental-type object that stores our type of value. The
  /// created object will have a valid value, but not necessarily the same
  /// as our value. The caller is responsible for deleting the returned object.
  virtual RooAbsArg *createFundamental(const char* newname=0) const = 0;

  /// Is this argument an l-value, i.e., can it appear on the left-hand side
  /// of an assignment expression? LValues are also special since they can
  /// potentially be analytically integrated and generated.
  inline virtual Bool_t isLValue() const {
    return kFALSE;
  }

  void addParameters(RooArgSet& params, const RooArgSet* nset=0, Bool_t stripDisconnected=kTRUE)  const ;

  // Parameter & observable interpretation of servers
  friend class RooProdPdf ;
  friend class RooAddPdf ;
  friend class RooAddPdfOrig ;
  RooArgSet* getVariables(Bool_t stripDisconnected=kTRUE) const ;
  RooArgSet* getParameters(const RooAbsData* data, Bool_t stripDisconnected=kTRUE) const ;
  /// Return the parameters of this p.d.f when used in conjuction with dataset 'data'
  RooArgSet* getParameters(const RooAbsData& data, Bool_t stripDisconnected=kTRUE) const {
    return getParameters(&data,stripDisconnected) ;
  }
  /// Return the parameters of the p.d.f given the provided set of observables
  RooArgSet* getParameters(const RooArgSet& observables, Bool_t stripDisconnected=kTRUE) const {
    return getParameters(&observables,stripDisconnected);
  }
  virtual RooArgSet* getParameters(const RooArgSet* depList, Bool_t stripDisconnected=kTRUE) const ;
  /// Return the observables of this pdf given a set of observables
  RooArgSet* getObservables(const RooArgSet& set, Bool_t valueOnly=kTRUE) const {
    return getObservables(&set,valueOnly) ;
  }
  RooArgSet* getObservables(const RooAbsData* data) const ;
  /// Return the observables of this pdf given the observables defined by `data`.
  RooArgSet* getObservables(const RooAbsData& data) const {
    return getObservables(&data) ;
  }
  RooArgSet* getObservables(const RooArgSet* depList, Bool_t valueOnly=kTRUE) const ;
  Bool_t observableOverlaps(const RooAbsData* dset, const RooAbsArg& testArg) const ;
  Bool_t observableOverlaps(const RooArgSet* depList, const RooAbsArg& testArg) const ;
  virtual Bool_t checkObservables(const RooArgSet* nset) const ;
  Bool_t recursiveCheckObservables(const RooArgSet* nset) const ;
  RooArgSet* getComponents() const ;

  // --- Obsolete functions for backward compatibility
  inline RooArgSet* getDependents(const RooArgSet& set) const { return getObservables(set) ; }
  inline RooArgSet* getDependents(const RooAbsData* set) const { return getObservables(set) ; }
  inline RooArgSet* getDependents(const RooArgSet* depList) const { return getObservables(depList) ; }
  inline Bool_t dependentOverlaps(const RooAbsData* dset, const RooAbsArg& testArg) const { return observableOverlaps(dset,testArg) ; }
  inline Bool_t dependentOverlaps(const RooArgSet* depList, const RooAbsArg& testArg) const { return observableOverlaps(depList, testArg) ; }
  inline Bool_t checkDependents(const RooArgSet* nset) const { return checkObservables(nset) ; }
  inline Bool_t recursiveCheckDependents(const RooArgSet* nset) const { return recursiveCheckObservables(nset) ; }
  // --- End obsolete functions for backward compatibility

  void attachDataSet(const RooAbsData &set);
  void attachDataStore(const RooAbsDataStore &set);

  // I/O streaming interface (machine readable)
  virtual Bool_t readFromStream(std::istream& is, Bool_t compact, Bool_t verbose=kFALSE) = 0 ;
  virtual void writeToStream(std::ostream& os, Bool_t compact) const = 0 ;

  inline virtual void Print(Option_t *options= 0) const {
    // Printing interface (human readable)
    printStream(defaultPrintStream(),defaultPrintContents(options),defaultPrintStyle(options));
  }

  virtual void printName(std::ostream& os) const ;
  virtual void printTitle(std::ostream& os) const ;
  virtual void printClassName(std::ostream& os) const ;
  virtual void printAddress(std::ostream& os) const ;
  virtual void printArgs(std::ostream& os) const ;
  virtual void printMetaArgs(std::ostream& /*os*/) const {} ;
  virtual void printMultiline(std::ostream& os, Int_t contents, Bool_t verbose=kFALSE, TString indent="") const;
  virtual void printTree(std::ostream& os, TString indent="") const ;

  virtual Int_t defaultPrintContents(Option_t* opt) const ;

  // Accessors to attributes
  void setAttribute(const Text_t* name, Bool_t value=kTRUE) ;
  Bool_t getAttribute(const Text_t* name) const ;
  inline const std::set<std::string>& attributes() const {
    // Returns set of names of boolean attributes defined
    return _boolAttrib ;
  }

  void setStringAttribute(const Text_t* key, const Text_t* value) ;
  const Text_t* getStringAttribute(const Text_t* key) const ;
  inline const std::map<std::string,std::string>& stringAttributes() const {
    // Returns std::map<string,string> with all string attributes defined
    return _stringAttrib ;
  }

  // Accessors to transient attributes
  void setTransientAttribute(const Text_t* name, Bool_t value=kTRUE) ;
  Bool_t getTransientAttribute(const Text_t* name) const ;
  inline const std::set<std::string>& transientAttributes() const {
    // Return set of transient boolean attributes
    return _boolAttribTransient ;
  }

  inline Bool_t isConstant() const {
    // Returns true if 'Constant' attribute is set
    return _isConstant ; //getAttribute("Constant") ;
  }
  RooLinkedList getCloningAncestors() const ;

  // Sorting
  Int_t Compare(const TObject* other) const ;
  virtual Bool_t IsSortable() const {
    // Object is sortable in ROOT container class
    return kTRUE ;
  }

  //Debug hooks
  static void verboseDirty(Bool_t flag) ;
  void printDirty(Bool_t depth=kTRUE) const ;

  static void setDirtyInhibit(Bool_t flag) ;

  virtual Bool_t operator==(const RooAbsArg& other) = 0 ;
  virtual Bool_t isIdentical(const RooAbsArg& other, Bool_t assumeSameType=kFALSE) = 0 ;

  // Range management
  virtual Bool_t inRange(const char*) const {
    // Is value in range (dummy interface always returns true)
    return kTRUE ;
  }
  virtual Bool_t hasRange(const char*) const {
    // Has this argument a defined range (dummy interface always returns flase)
    return kFALSE ;
  }


  enum ConstOpCode { Activate=0, DeActivate=1, ConfigChange=2, ValueChange=3 } ;


  friend class RooMinuit ;

  // Cache mode optimization (tracks changes & do lazy evaluation vs evaluate always)
  virtual void optimizeCacheMode(const RooArgSet& observables) ;
  virtual void optimizeCacheMode(const RooArgSet& observables, RooArgSet& optNodes, RooLinkedList& processedNodes) ;


  // Find constant terms in expression
  Bool_t findConstantNodes(const RooArgSet& observables, RooArgSet& cacheList) ;
  Bool_t findConstantNodes(const RooArgSet& observables, RooArgSet& cacheList, RooLinkedList& processedNodes) ;


  // constant term optimization
  virtual void constOptimizeTestStatistic(ConstOpCode opcode, Bool_t doAlsoTrackingOpt=kTRUE) ;
  enum CacheMode { Always=0, NotAdvised=1, Never=2 } ;
  virtual CacheMode canNodeBeCached() const { return Always ; }
  virtual void setCacheAndTrackHints(RooArgSet& /*trackNodes*/ ) {} ;

  void graphVizTree(const char* fileName, const char* delimiter="\n", bool useTitle=false, bool useLatex=false) ;
  void graphVizTree(std::ostream& os, const char* delimiter="\n", bool useTitle=false, bool useLatex=false) ;

  void printComponentTree(const char* indent="",const char* namePat=0, Int_t nLevel=999) ;
  void printCompactTree(const char* indent="",const char* fileName=0, const char* namePat=0, RooAbsArg* client=0) ;
  void printCompactTree(std::ostream& os, const char* indent="", const char* namePat=0, RooAbsArg* client=0) ;
  virtual void printCompactTreeHook(std::ostream& os, const char *ind="") ;

  // Dirty state accessor
  inline Bool_t isShapeDirty() const {
    // Return true is shape has been invalidated by server value change
    return isDerived()?_shapeDirty:kFALSE ;
  }

  inline Bool_t isValueDirty() const {
    // Returns true of value has been invalidated by server value change
    if (inhibitDirty()) return kTRUE ;
    switch(_operMode) {
    case AClean:
      return kFALSE ;
    case ADirty:
      return kTRUE ;
    case Auto:
      if (_valueDirty) return isDerived() ;
      return kFALSE ;
    }
    return kTRUE ; // we should never get here
  }

  inline Bool_t isValueDirtyAndClear() const {
    // Returns true of value has been invalidated by server value change
    if (inhibitDirty()) return kTRUE ;
    switch(_operMode) {
    case AClean:
      return kFALSE ;
    case ADirty:
      return kTRUE ;
    case Auto:
      if (_valueDirty) {
	_valueDirty = kFALSE ;
	return isDerived();
      }
      return kFALSE ;
    }
    return kTRUE ; // But we should never get here
  }


  inline Bool_t isValueOrShapeDirtyAndClear() const {
    // Returns true of value has been invalidated by server value change

    if (inhibitDirty()) return kTRUE ;
    switch(_operMode) {
    case AClean:
      return kFALSE ;
    case ADirty:
      return kTRUE ;
    case Auto:
      if (_valueDirty || _shapeDirty) {
	_shapeDirty = kFALSE ;
	_valueDirty = kFALSE ;
	return isDerived();
      }
      _shapeDirty = kFALSE ;
      _valueDirty = kFALSE ;
      return kFALSE ;
    }
    return kTRUE ; // But we should never get here
  }

  // Cache management
  void registerCache(RooAbsCache& cache) ;
  void unRegisterCache(RooAbsCache& cache) ;
  Int_t numCaches() const ;
  RooAbsCache* getCache(Int_t index) const ;

  enum OperMode { Auto=0, AClean=1, ADirty=2 } ;
  inline OperMode operMode() const { return _operMode  ; }
  void setOperMode(OperMode mode, Bool_t recurseADirty=kTRUE) ;

  static UInt_t crc32(const char* data);
  static UInt_t crc32(const char* data, ULong_t sz, UInt_t crc = 0);

  static const UInt_t fnv1a32start = 2166136261u;
  static UInt_t fnv1a32(const char* data);
  static UInt_t fnv1a32(const char* data, ULong_t sz, UInt_t hash = fnv1a32start);

  static const ULong64_t fnv1a64start = (ULong64_t(3421674724u) << 32) | ULong64_t(2216829733u);
  static ULong64_t fnv1a64(const char* data);
  static ULong64_t fnv1a64(const char* data, ULong_t sz, ULong64_t hash = fnv1a64start);

  Bool_t addOwnedComponents(const RooArgSet& comps) ;
  const RooArgSet* ownedComponents() const { return _ownedComponents ; }

  void setProhibitServerRedirect(Bool_t flag) { _prohibitServerRedirect = flag ; }

  void setWorkspace(RooWorkspace &ws) { _myws = &ws; }

  protected:

  void graphVizAddConnections(std::set<std::pair<RooAbsArg*,RooAbsArg*> >&) ;

  friend class RooExtendPdf ;
  friend class RooRealIntegral ;
  friend class RooAbsReal ;
  friend class RooProjectedPdf ;
  //friend class RooSimCloneTool ;

  virtual void operModeHook() {} ;

  virtual void optimizeDirtyHook(const RooArgSet* /*obs*/) {} ;

  virtual Bool_t isValid() const ;

  virtual void getParametersHook(const RooArgSet* /*nset*/, RooArgSet* /*list*/, Bool_t /*stripDisconnected*/) const {} ;
  virtual void getObservablesHook(const RooArgSet* /*nset*/, RooArgSet* /*list*/) const {} ;

  // Dirty state modifiers
 public:
  inline void setValueDirty() const {   if (_operMode==Auto && !inhibitDirty()) setValueDirty(0) ; }
  inline void setShapeDirty() const { setShapeDirty(0) ; }

  inline void clearValueAndShapeDirty() const {
    _valueDirty=kFALSE ;
    _shapeDirty=kFALSE ;
  }

  inline void clearValueDirty() const {
    _valueDirty=kFALSE ;
  }
  inline void clearShapeDirty() const {
    _shapeDirty=kFALSE ;
  }

  const char* aggregateCacheUniqueSuffix() const ;
  virtual const char* cacheUniqueSuffix() const { return 0 ; }

  void wireAllCaches() ;

  inline const TNamed* namePtr() const {
    return _namePtr ;
  }

  void SetName(const char* name) ;
  void SetNameTitle(const char *name, const char *title) ;

 protected:

  // Client-Server relatation and Proxy management
  friend class RooArgSet ;
  friend class RooAbsCollection ;
  friend class RooCustomizer ;
  friend class RooWorkspace ;
  RefCountList_t _serverList       ; // list of server objects
  RefCountList_t _clientList; // list of client objects
  RefCountList_t _clientListShape; // subset of clients that requested shape dirty flag propagation
  RefCountList_t _clientListValue; // subset of clients that requested value dirty flag propagation

  RooRefArray _proxyList        ; // list of proxies
  std::deque<RooAbsCache*> _cacheList ; // list of caches

  // Server redirection interface
 public:
  Bool_t redirectServers(const RooAbsCollection& newServerList, Bool_t mustReplaceAll=kFALSE, Bool_t nameChange=kFALSE, Bool_t isRecursionStep=kFALSE) ;
  Bool_t recursiveRedirectServers(const RooAbsCollection& newServerList, Bool_t mustReplaceAll=kFALSE, Bool_t nameChange=kFALSE, Bool_t recurseInNewSet=kTRUE) ;
  virtual Bool_t redirectServersHook(const RooAbsCollection& /*newServerList*/, Bool_t /*mustReplaceAll*/, Bool_t /*nameChange*/, Bool_t /*isRecursive*/) { return kFALSE ; } ;
  virtual void serverNameChangeHook(const RooAbsArg* /*oldServer*/, const RooAbsArg* /*newServer*/) { } ;

  void addServer(RooAbsArg& server, Bool_t valueProp=kTRUE, Bool_t shapeProp=kFALSE, std::size_t refCount = 1);
  void addServerList(RooAbsCollection& serverList, Bool_t valueProp=kTRUE, Bool_t shapeProp=kFALSE) ;
  void replaceServer(RooAbsArg& oldServer, RooAbsArg& newServer, Bool_t valueProp, Bool_t shapeProp) ;
  void changeServer(RooAbsArg& server, Bool_t valueProp, Bool_t shapeProp) ;
  void removeServer(RooAbsArg& server, Bool_t force=kFALSE) ;
  RooAbsArg *findNewServer(const RooAbsCollection &newSet, Bool_t nameChange) const;

  RooExpensiveObjectCache& expensiveObjectCache() const ;
  virtual void setExpensiveObjectCache(RooExpensiveObjectCache &cache) { _eocache = &cache; }

  virtual Bool_t importWorkspaceHook(RooWorkspace &ws)
  {
     _myws = &ws;
     return kFALSE;
  };

  protected:
  // Proxy management
  friend class RooAddModel ;
  friend class RooArgProxy ;
  friend class RooSetProxy ;
  friend class RooListProxy ;
  friend class RooObjectFactory ;
  friend class RooHistPdf ;
  friend class RooHistFunc ;
  friend class RooHistFunc2 ;
  void registerProxy(RooArgProxy& proxy) ;
  void registerProxy(RooSetProxy& proxy) ;
  void registerProxy(RooListProxy& proxy) ;
  void unRegisterProxy(RooArgProxy& proxy) ;
  void unRegisterProxy(RooSetProxy& proxy) ;
  void unRegisterProxy(RooListProxy& proxy) ;
  RooAbsProxy* getProxy(Int_t index) const ;
  void setProxyNormSet(const RooArgSet* nset) ;
  Int_t numProxies() const ;

  // Attribute list
  std::set<std::string> _boolAttrib ; // Boolean attributes
  std::map<std::string,std::string> _stringAttrib ; // String attributes
  std::set<std::string> _boolAttribTransient ; //! Transient boolean attributes (not copied in ctor)

  void printAttribList(std::ostream& os) const;

  // Hooks for RooTreeData interface
  friend class RooCompositeDataStore ;
  friend class RooTreeDataStore ;
  friend class RooVectorDataStore ;
  friend class RooTreeData ;
  friend class RooDataSet ;
  friend class RooRealMPFE ;
  virtual void syncCache(const RooArgSet* nset=0) = 0 ;
  virtual void copyCache(const RooAbsArg* source, Bool_t valueOnly=kFALSE, Bool_t setValDirty=kTRUE) = 0 ;

  virtual void attachToTree(TTree& t, Int_t bufSize=32000) = 0 ;
  virtual void attachToVStore(RooVectorDataStore& vstore) = 0 ;
  void attachToStore(RooAbsDataStore& store) ;

  virtual void setTreeBranchStatus(TTree& t, Bool_t active) = 0 ;
  virtual void fillTreeBranch(TTree& t) = 0 ;
  TString cleanBranchName() const ;

  // Global
  friend std::ostream& operator<<(std::ostream& os, const RooAbsArg &arg);
  friend std::istream& operator>>(std::istream& is, RooAbsArg &arg) ;

  // Debug stuff
  static Bool_t _verboseDirty ; // Static flag controlling verbose messaging for dirty state changes
  static Bool_t _inhibitDirty ; // Static flag controlling global inhibit of dirty state propagation
  Bool_t _deleteWatch ; //! Delete watch flag

  Bool_t inhibitDirty() const ;

 public:
  void setLocalNoDirtyInhibit(Bool_t flag) const { _localNoInhibitDirty = flag ; }
  Bool_t localNoDirtyInhibit() const { return _localNoInhibitDirty ; }
 protected:

  // Value and Shape dirty state bits
  void setValueDirty(const RooAbsArg* source) const ;
  void setShapeDirty(const RooAbsArg* source) const ;
  mutable Bool_t _valueDirty ;  // Flag set if value needs recalculating because input values modified
  mutable Bool_t _shapeDirty ;  // Flag set if value needs recalculating because input shapes modified

  friend class RooRealProxy ;
  mutable OperMode _operMode ; // Dirty state propagation mode
  mutable Bool_t _fast ; // Allow fast access mode in getVal() and proxies

  // Owned components
  RooArgSet* _ownedComponents ; //! Set of owned component

  mutable Bool_t _prohibitServerRedirect ; //! Prohibit server redirects -- Debugging tool

  mutable RooExpensiveObjectCache* _eocache ; // Pointer to global cache manager for any expensive components created by this object

  mutable TNamed* _namePtr ; //! Do not persist. Pointer to global instance of string that matches object named
  Bool_t _isConstant ; //! Cached isConstant status

  mutable Bool_t _localNoInhibitDirty ; //! Prevent 'AlwaysDirty' mode for this node

/*   RooArgSet _leafNodeCache ; //! Cached leaf nodes */
/*   RooArgSet _branchNodeCache //! Cached branch nodes     */

  mutable RooWorkspace *_myws; //! In which workspace do I live, if any

  public:
  virtual void ioStreamerPass2() ;
  static void ioStreamerPass2Finalize() ;
  static std::map<RooAbsArg*,TRefArray*> _ioEvoList ; // temporary holding list for proxies needed in schema evolution
  static std::stack<RooAbsArg*> _ioReadStack ; // reading stack

  private:
  RefCountListLegacyIterator_t * makeLegacyIterator(const RefCountList_t& list) const;

  ClassDef(RooAbsArg,7) // Abstract variable
};

std::ostream& operator<<(std::ostream& os, const RooAbsArg &arg);
std::istream& operator>>(std::istream& is, RooAbsArg &arg) ;

/// Print a RDataFrame at the prompt
namespace cling {
std::string printValue(RooAbsArg *raa);
} // namespace cling


#endif
