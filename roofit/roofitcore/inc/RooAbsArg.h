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
#include "RooRefCountList.h"
#include "RooAbsCache.h"
#include "RooLinkedListIter.h"
#include "RooNameReg.h"
#include <map>
#include <set>
#include <deque>

#include <iostream>
using namespace std ;
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
/* class TGraphStruct ; */

class RooAbsArg : public TNamed, public RooPrintable {
public:

  // Constructors, cloning and assignment
  RooAbsArg() ;
  virtual ~RooAbsArg();
  RooAbsArg(const char *name, const char *title);
  RooAbsArg(const RooAbsArg& other, const char* name=0) ;
  virtual TObject* clone(const char* newname) const = 0 ;
  virtual TObject* Clone(const char* newname=0) const { 
    return clone(newname?newname:GetName()) ; 
  }
  virtual RooAbsArg* cloneTree(const char* newname=0) const ;

  // Accessors to client-server relation information 
  virtual Bool_t isDerived() const { 
    // Does value or shape of this arg depend on any other arg?
    return kTRUE ;
    //cout << IsA()->GetName() << "::isDerived(" << GetName() << ") = " << (_serverList.GetSize()>0 || _proxyList.GetSize()>0) << endl ;
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
  Bool_t hasClients() const { return _clientList.GetSize()>0 ? kTRUE : kFALSE ; }
  inline TIterator* clientIterator() const { 
    // Return iterator over all client RooAbsArgs
    return _clientList.MakeIterator() ; 
  }
  inline TIterator* valueClientIterator() const { 
    // Return iterator over all value client RooAbsArgs
    return _clientListValue.MakeIterator() ; 
  }
  inline TIterator* shapeClientIterator() const { 
    // Return iterator over all shape client RooAbsArgs
    return _clientListShape.MakeIterator() ; 
  }
  inline TIterator* serverIterator() const { 
    // Return iterator over all server RooAbsArgs
    return _serverList.MakeIterator() ; 
  }

  inline RooFIter valueClientMIterator() const { return _clientListValue.fwdIterator() ; }
  inline RooFIter shapeClientMIterator() const { return _clientListShape.fwdIterator() ; }
  inline RooFIter serverMIterator() const { return _serverList.fwdIterator() ; }


  inline RooAbsArg* findServer(const char *name) const { 
    // Return server of this arg with given name. Returns null if not found
    return (RooAbsArg*)_serverList.FindObject(name); 
  }
  inline RooAbsArg* findServer(const RooAbsArg& arg) const { 
    // Return server of this arg with name of given input arg. Returns null if not found
    return (RooAbsArg*)_serverList.FindObject(&arg); 
  }
  inline RooAbsArg* findServer(Int_t index) const { 
    // Return i-th server from server list
    return (RooAbsArg*)_serverList.At(index); 
  }
  inline Bool_t isValueServer(const RooAbsArg& arg) const { 
    // If true, arg is a value server of self
    return _clientListValue.FindObject(&arg)?kTRUE:kFALSE ; 
  }
  inline Bool_t isValueServer(const char* name) const { 
    // If true, we have a server with given name
    return _clientListValue.FindObject(name)?kTRUE:kFALSE ; 
  }
  inline Bool_t isShapeServer(const RooAbsArg& arg) const { 
    // If true arg is a shape server of self
    return _clientListShape.FindObject(&arg)?kTRUE:kFALSE ; 
  }
  inline Bool_t isShapeServer(const char* name) const { 
    // If true, we have a shape server with given name
    return _clientListShape.FindObject(name)?kTRUE:kFALSE ; 
  }
  void leafNodeServerList(RooAbsCollection* list, const RooAbsArg* arg=0, Bool_t recurseNonDerived=kFALSE) const ;
  void branchNodeServerList(RooAbsCollection* list, const RooAbsArg* arg=0, Bool_t recurseNonDerived=kFALSE) const ;
  void treeNodeServerList(RooAbsCollection* list, const RooAbsArg* arg=0, 
			  Bool_t doBranch=kTRUE, Bool_t doLeaf=kTRUE, 
			  Bool_t valueOnly=kFALSE, Bool_t recurseNonDerived=kFALSE) const ;
  

  inline virtual Bool_t isFundamental() const { 
    // Is this object a fundamental type that can be added to a dataset?
    // Fundamental-type subclasses override this method to return kTRUE.
    // Note that this test is subtlely different from the dynamic isDerived()
    // test, e.g. a constant is not derived but is also not fundamental.
    return kFALSE; 
  }

  // Create a fundamental-type object that stores our type of value. The
  // created object will have a valid value, but not necessarily the same
  // as our value. The caller is responsible for deleting the returned object.
  virtual RooAbsArg *createFundamental(const char* newname=0) const = 0;

  inline virtual Bool_t isLValue() const { 
    // Is this argument an l-value, ie, can it appear on the left-hand side
    // of an assignment expression? LValues are also special since they can
    // potentially be analytically integrated and generated.
    return kFALSE; 
  }

  void addParameters(RooArgSet& params, const RooArgSet* nset=0, Bool_t stripDisconnected=kTRUE)  const ; 

  // Parameter & observable interpretation of servers
  friend class RooProdPdf ;
  friend class RooAddPdf ;
  friend class RooAddPdfOrig ;
  RooArgSet* getVariables(Bool_t stripDisconnected=kTRUE) const ;
  RooArgSet* getParameters(const RooAbsData* data, Bool_t stripDisconnected=kTRUE) const ;
  RooArgSet* getParameters(const RooAbsData& data, Bool_t stripDisconnected=kTRUE) const { 
    // Return the parameters of this p.d.f when used in conjuction with dataset 'data'
    return getParameters(&data,stripDisconnected) ; 
  }
  RooArgSet* getParameters(const RooArgSet& set, Bool_t stripDisconnected=kTRUE) const { 
    // Return the parameters of the p.d.f given the provided set of observables
    return getParameters(&set,stripDisconnected) ; 
  }
  virtual RooArgSet* getParameters(const RooArgSet* depList, Bool_t stripDisconnected=kTRUE) const ;
  RooArgSet* getObservables(const RooArgSet& set, Bool_t valueOnly=kTRUE) const { 
    // Return the observables of _this_ pdf given a set of observables
    return getObservables(&set,valueOnly) ; 
  }
  RooArgSet* getObservables(const RooAbsData* data) const ;
  RooArgSet* getObservables(const RooAbsData& data) const { 
    // Return the observables of _this_ pdf given the observables defined by 'data'
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
  virtual Bool_t readFromStream(istream& is, Bool_t compact, Bool_t verbose=kFALSE) = 0 ;
  virtual void writeToStream(ostream& os, Bool_t compact) const = 0 ;

  inline virtual void Print(Option_t *options= 0) const {
    // Printing interface (human readable)
    printStream(defaultPrintStream(),defaultPrintContents(options),defaultPrintStyle(options));
  }

  virtual void printName(ostream& os) const ;
  virtual void printTitle(ostream& os) const ;
  virtual void printClassName(ostream& os) const ;
  virtual void printAddress(ostream& os) const ;
  virtual void printArgs(ostream& os) const ;
  virtual void printMetaArgs(ostream& /*os*/) const {} ;
  virtual void printMultiline(ostream& os, Int_t contents, Bool_t verbose=kFALSE, TString indent="") const;
  virtual void printTree(ostream& os, TString indent="") const ;

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
    // Returns map<string,string> with all string attributes defined
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
    return getAttribute("Constant") ; 
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
  
  void graphVizTree(const char* fileName, const char* delimiter="\n", bool useTitle=false, bool useLatex=false) ;
  void graphVizTree(ostream& os, const char* delimiter="\n", bool useTitle=false, bool useLatex=false) ;

/*   TGraphStruct* graph(Bool_t useFactoryTag=kFALSE, Double_t textSize=0.03) ; */

  void printComponentTree(const char* indent="",const char* namePat=0, Int_t nLevel=999) ;
  void printCompactTree(const char* indent="",const char* fileName=0, const char* namePat=0, RooAbsArg* client=0) ;
  void printCompactTree(ostream& os, const char* indent="", const char* namePat=0, RooAbsArg* client=0) ;
  virtual void printCompactTreeHook(ostream& os, const char *ind="") ;

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

  static UInt_t crc32(const char* data) ;
  
  Bool_t addOwnedComponents(const RooArgSet& comps) ;
  const RooArgSet* ownedComponents() const { return _ownedComponents ; }
  
  void setProhibitServerRedirect(Bool_t flag) { _prohibitServerRedirect = flag ; }

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
    if (!_namePtr) {
      _namePtr = (TNamed*) RooNameReg::instance().constPtr(GetName()) ;
    }
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
  RooRefCountList _serverList       ; // list of server objects
  RooRefCountList _clientList       ; // list of client objects
  RooRefCountList _clientListShape  ; // subset of clients that requested shape dirty flag propagation
  RooRefCountList _clientListValue  ; // subset of clients that requested value dirty flag propagation
  TRefArray _proxyList        ; // list of proxies
  std::deque<RooAbsCache*> _cacheList ; // list of caches
  TIterator* _clientShapeIter ; //! Iterator over _clientListShape 
  TIterator* _clientValueIter ; //! Iterator over _clientListValue 

  // Server redirection interface
 public:
  Bool_t redirectServers(const RooAbsCollection& newServerList, Bool_t mustReplaceAll=kFALSE, Bool_t nameChange=kFALSE, Bool_t isRecursionStep=kFALSE) ;
  Bool_t recursiveRedirectServers(const RooAbsCollection& newServerList, Bool_t mustReplaceAll=kFALSE, Bool_t nameChange=kFALSE, Bool_t recurseInNewSet=kTRUE) ;
  virtual Bool_t redirectServersHook(const RooAbsCollection& /*newServerList*/, Bool_t /*mustReplaceAll*/, Bool_t /*nameChange*/, Bool_t /*isRecursive*/) { return kFALSE ; } ;
  virtual void serverNameChangeHook(const RooAbsArg* /*oldServer*/, const RooAbsArg* /*newServer*/) { } ;

  void addServer(RooAbsArg& server, Bool_t valueProp=kTRUE, Bool_t shapeProp=kFALSE) ;
  void addServerList(RooAbsCollection& serverList, Bool_t valueProp=kTRUE, Bool_t shapeProp=kFALSE) ;
  void replaceServer(RooAbsArg& oldServer, RooAbsArg& newServer, Bool_t valueProp, Bool_t shapeProp) ;
  void changeServer(RooAbsArg& server, Bool_t valueProp, Bool_t shapeProp) ;
  void removeServer(RooAbsArg& server, Bool_t force=kFALSE) ;
  RooAbsArg *findNewServer(const RooAbsCollection &newSet, Bool_t nameChange) const;

  RooExpensiveObjectCache& expensiveObjectCache() const ;
  void setExpensiveObjectCache(RooExpensiveObjectCache& cache) { _eocache = &cache ; }  

  virtual Bool_t importWorkspaceHook(RooWorkspace&) { return kFALSE ; } ;

 protected:

  // Proxy management
  friend class RooAddModel ;
  friend class RooArgProxy ;
  friend class RooSetProxy ;
  friend class RooListProxy ;
  friend class RooObjectFactory ;
  friend class RooHistPdf ;
  friend class RooHistFunc ;
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

  void printAttribList(ostream& os) const;

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
  friend ostream& operator<<(ostream& os, const RooAbsArg &arg);  
  friend istream& operator>>(istream& is, RooAbsArg &arg) ;

  // Debug stuff
  static Bool_t _verboseDirty ; // Static flag controlling verbose messaging for dirty state changes
  static Bool_t _inhibitDirty ; // Static flag controlling global inhibit of dirty state propagation
  Bool_t _deleteWatch ; //! Delete watch flag 

  static Bool_t inhibitDirty() ;
  
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
  
  ClassDef(RooAbsArg,5) // Abstract variable
};

ostream& operator<<(ostream& os, const RooAbsArg &arg);  
istream& operator>>(istream& is, RooAbsArg &arg) ;

#endif
