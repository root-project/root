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

#include <RooAbsCache.h>
#include <RooFit/Config.h>
#include <RooFit/Detail/NormalizationHelpers.h>
#include <RooLinkedListIter.h>
#include <RooNameReg.h>
#include <RooPrintable.h>
#include <RooSTLRefCountList.h>
#include <RooStringView.h>

#include <TNamed.h>
#include <TObjArray.h>
#include <TRefArray.h>

#include <deque>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <stack>
#include <string>
#include <unordered_map>


class TTree ;
class RooArgSet ;
class RooAbsCollection ;
class RooVectorDataStore ;
class RooAbsData ;
class RooAbsDataStore ;
class RooAbsProxy ;
class RooArgProxy ;
template<class RooCollection_t>
class RooCollectionProxy;
using RooSetProxy = RooCollectionProxy<RooArgSet>;
using RooListProxy = RooCollectionProxy<RooArgList>;
class RooExpensiveObjectCache ;
class RooWorkspace ;
namespace RooFit {
namespace Experimental {
class CodegenContext;
}
}

class RooRefArray : public TObjArray {
 public:
    RooRefArray() = default;
    RooRefArray(const RooRefArray &other) : TObjArray(other) {}
    RooRefArray &operator=(const RooRefArray &other) = default;
 protected:
  ClassDefOverride(RooRefArray,1) // Helper class for proxy lists
} ;

class RooAbsArg;
/// Print at the prompt
namespace cling {
std::string printValue(RooAbsArg*);
}

class RooAbsArg : public TNamed, public RooPrintable {
public:
  using RefCountList_t = RooSTLRefCountList<RooAbsArg>;
  using RefCountListLegacyIterator_t = TIteratorToSTLInterface<RefCountList_t::Container_t>;

  // Constructors, cloning and assignment
  RooAbsArg() ;
  ~RooAbsArg() override;
  RooAbsArg(const char *name, const char *title);
  RooAbsArg(const RooAbsArg& other, const char* name=nullptr) ;
  RooAbsArg& operator=(const RooAbsArg& other) = delete;
  virtual TObject* clone(const char* newname=nullptr) const = 0 ;
  TObject* Clone(const char* newname = nullptr) const override {
    return clone(newname && newname[0] != '\0' ? newname : nullptr);
  }
  virtual RooAbsArg* cloneTree(const char* newname=nullptr) const ;

  // Accessors to client-server relation information

  /// Does value or shape of this arg depend on any other arg?
  virtual bool isDerived() const {
    return true ;
  }

  /// Check whether this object depends on values from an element in the `serverList`.
  ///
  /// @param serverList Test if one of the elements in this list serves values to `this`.
  /// @param ignoreArg Ignore values served by this object.
  /// @return True if values are served.
  bool dependsOnValue(const RooAbsCollection& serverList, const RooAbsArg* ignoreArg=nullptr) const {
    return dependsOn(serverList,ignoreArg,true) ;
  }
  /// Check whether this object depends on values served from the object passed as `server`.
  ///
  /// @param server Test if `server` serves values to `this`.
  /// @param ignoreArg Ignore values served by this object.
  /// @return True if values are served.
  bool dependsOnValue(const RooAbsArg& server, const RooAbsArg* ignoreArg=nullptr) const {
    return dependsOn(server,ignoreArg,true) ;
  }
  bool dependsOn(const RooAbsCollection& serverList, const RooAbsArg* ignoreArg=nullptr, bool valueOnly=false) const ;
  /// Test whether we depend on (ie, are served by) the specified object.
  /// Note that RooAbsArg objects are considered equivalent if they have
  /// the same name.
  inline bool dependsOn(const RooAbsArg& server, const RooAbsArg* ignoreArg=nullptr, bool valueOnly=false) const {
    return dependsOn(server.namePtr(), ignoreArg, valueOnly);
  }
  bool dependsOn(TNamed const* namePtr, const RooAbsArg* ignoreArg=nullptr, bool valueOnly=false) const ;
  bool overlaps(const RooAbsArg& testArg, bool valueOnly=false) const ;
  bool hasClients() const { return !_clientList.empty(); }

  ////////////////////////////////////////////////////////////////////////////
  /// \anchor clientServerInterface
  /// \name Client-Server Interface
  /// These functions allow RooFit to figure out who is serving values to whom.
  /// @{

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

  /// List of all servers of this object.
  const RefCountList_t& servers() const {
    return _serverList;
  }
  /// Return server of `this` with name `name`. Returns nullptr if not found.
  inline RooAbsArg* findServer(const char *name) const {
    const auto serverIt = _serverList.findByName(name);
    return serverIt != _serverList.end() ? *serverIt : nullptr;
  }
  /// Return server of `this` that has the same name as `arg`. Returns `nullptr` if not found.
  inline RooAbsArg* findServer(const RooAbsArg& arg) const {
    return _serverList.findByNamePointer(&arg);
  }
  /// Return i-th server from server list.
  inline RooAbsArg* findServer(Int_t index) const {
    return _serverList.containedObjects()[index];
  }
  /// Check if `this` is serving values to `arg`.
  inline bool isValueServer(const RooAbsArg& arg) const {
    return _clientListValue.containsByNamePtr(&arg);
  }
  /// Check if `this` is serving values to an object with name `name`.
  inline bool isValueServer(const char* name) const {
    return _clientListValue.containsSameName(name);
  }
  /// Check if `this` is serving shape to `arg`.
  inline bool isShapeServer(const RooAbsArg& arg) const {
    return _clientListShape.containsByNamePtr(&arg);
  }
  /// Check if `this` is serving shape to an object with name `name`.
  inline bool isShapeServer(const char* name) const {
    return _clientListShape.containsSameName(name);
  }
  void leafNodeServerList(RooAbsCollection* list, const RooAbsArg* arg=nullptr, bool recurseNonDerived=false) const ;
  void branchNodeServerList(RooAbsCollection* list, const RooAbsArg* arg=nullptr, bool recurseNonDerived=false) const ;
  void treeNodeServerList(RooAbsCollection* list, const RooAbsArg* arg=nullptr,
           bool doBranch=true, bool doLeaf=true,
           bool valueOnly=false, bool recurseNonDerived=false) const ;


  /// Is this object a fundamental type that can be added to a dataset?
  /// Fundamental-type subclasses override this method to return true.
  /// Note that this test is subtlely different from the dynamic isDerived()
  /// test, e.g. a constant is not derived but is also not fundamental.
  inline virtual bool isFundamental() const {
    return false;
  }

  /// Create a fundamental-type object that stores our type of value. The
  /// created object will have a valid value, but not necessarily the same
  /// as our value. The caller is responsible for deleting the returned object.
  virtual RooFit::OwningPtr<RooAbsArg> createFundamental(const char* newname=nullptr) const = 0;

  /// Is this argument an l-value, i.e., can it appear on the left-hand side
  /// of an assignment expression? LValues are also special since they can
  /// potentially be analytically integrated and generated.
  inline virtual bool isLValue() const {
    return false;
  }


  // Server redirection interface
  bool redirectServers(const RooAbsCollection& newServerList, bool mustReplaceAll=false, bool nameChange=false, bool isRecursionStep=false) ;
  bool redirectServers(std::unordered_map<RooAbsArg*, RooAbsArg*> const& replacements);
  bool recursiveRedirectServers(const RooAbsCollection& newServerList, bool mustReplaceAll=false, bool nameChange=false, bool recurseInNewSet=true) ;

  virtual bool redirectServersHook(const RooAbsCollection & newServerList, bool mustReplaceAll,
                                   bool nameChange, bool isRecursiveStep);

  virtual void serverNameChangeHook(const RooAbsArg* /*oldServer*/, const RooAbsArg* /*newServer*/) { } ;

  void addServer(RooAbsArg& server, bool valueProp=true, bool shapeProp=false, std::size_t refCount = 1);
  void addServerList(RooAbsCollection& serverList, bool valueProp=true, bool shapeProp=false) ;
  void
  R__SUGGEST_ALTERNATIVE("This interface is unsafe! Use RooAbsArg::redirectServers()")
  replaceServer(RooAbsArg& oldServer, RooAbsArg& newServer, bool valueProp, bool shapeProp) ;
  void changeServer(RooAbsArg& server, bool valueProp, bool shapeProp) ;
  void removeServer(RooAbsArg& server, bool force=false) ;
  RooAbsArg *findNewServer(const RooAbsCollection &newSet, bool nameChange) const;


  /// @}
  ///////////////////////////////////////////////////////////////////////////////


  // Parameter & observable interpretation of servers
  RooFit::OwningPtr<RooArgSet> getVariables(bool stripDisconnected=true) const ;
  RooFit::OwningPtr<RooArgSet> getParameters(const RooAbsData* data, bool stripDisconnected=true) const;
  RooFit::OwningPtr<RooArgSet> getParameters(const RooAbsData& data, bool stripDisconnected=true) const;
  RooFit::OwningPtr<RooArgSet> getParameters(const RooArgSet& observables, bool stripDisconnected=true) const;
  RooFit::OwningPtr<RooArgSet> getParameters(const RooArgSet* observables, bool stripDisconnected=true) const;
  virtual bool getParameters(const RooArgSet* observables, RooArgSet& outputSet, bool stripDisconnected=true) const;
  RooFit::OwningPtr<RooArgSet> getObservables(const RooArgSet& set, bool valueOnly=true) const;
  RooFit::OwningPtr<RooArgSet> getObservables(const RooAbsData* data) const;
  RooFit::OwningPtr<RooArgSet> getObservables(const RooAbsData& data) const;
  RooFit::OwningPtr<RooArgSet> getObservables(const RooArgSet* depList, bool valueOnly=true) const;
  bool getObservables(const RooAbsCollection* depList, RooArgSet& outputSet, bool valueOnly=true) const;
  bool observableOverlaps(const RooAbsData* dset, const RooAbsArg& testArg) const ;
  bool observableOverlaps(const RooArgSet* depList, const RooAbsArg& testArg) const ;
  virtual bool checkObservables(const RooArgSet* nset) const ;
  bool recursiveCheckObservables(const RooArgSet* nset) const ;
  RooFit::OwningPtr<RooArgSet> getComponents() const ;



  void attachArgs(const RooAbsCollection &set);
  void attachDataSet(const RooAbsData &set);
  void attachDataStore(const RooAbsDataStore &set);

  // I/O streaming interface (machine readable)
  virtual bool readFromStream(std::istream& is, bool compact, bool verbose=false) = 0 ;
  virtual void writeToStream(std::ostream& os, bool compact) const = 0 ;

  /// Print the object to the defaultPrintStream().
  /// \param[in] options **V** print verbose. **T** print a tree structure with all children.
  void Print(Option_t *options= nullptr) const override {
    // Printing interface (human readable)
    printStream(defaultPrintStream(),defaultPrintContents(options),defaultPrintStyle(options));
  }

  void printName(std::ostream& os) const override ;
  void printTitle(std::ostream& os) const override ;
  void printClassName(std::ostream& os) const override ;
  void printAddress(std::ostream& os) const override ;
  void printArgs(std::ostream& os) const override ;
  virtual void printMetaArgs(std::ostream& /*os*/) const {} ;
  void printMultiline(std::ostream& os, Int_t contents, bool verbose=false, TString indent="") const override;
  void printTree(std::ostream& os, TString indent="") const override ;

  Int_t defaultPrintContents(Option_t* opt) const override ;

  // Accessors to attributes
  void setAttribute(const Text_t* name, bool value=true) ;
  bool getAttribute(const Text_t* name) const ;
  inline const std::set<std::string>& attributes() const {
    // Returns set of names of boolean attributes defined
    return _boolAttrib ;
  }

  void setStringAttribute(const Text_t* key, const Text_t* value) ;
  void removeStringAttribute(const Text_t* key) ;
  const Text_t* getStringAttribute(const Text_t* key) const ;
  inline const std::map<std::string,std::string>& stringAttributes() const {
    // Returns std::map<string,string> with all string attributes defined
    return _stringAttrib ;
  }

  // Accessors to transient attributes
  void setTransientAttribute(const Text_t* name, bool value=true) ;
  bool getTransientAttribute(const Text_t* name) const ;
  inline const std::set<std::string>& transientAttributes() const {
    // Return set of transient boolean attributes
    return _boolAttribTransient ;
  }

  /// Check if the "Constant" attribute is set.
  inline bool isConstant() const {
    return _isConstant ; //getAttribute("Constant") ;
  }

  // Sorting
  Int_t Compare(const TObject* other) const override ;
  bool IsSortable() const override {
    // Object is sortable in ROOT container class
    return true ;
  }

  virtual bool operator==(const RooAbsArg& other) const = 0 ;
  virtual bool isIdentical(const RooAbsArg& other, bool assumeSameType=false) const = 0 ;

  // Range management
  virtual bool inRange(const char*) const {
    // Is value in range (dummy interface always returns true)
    return true ;
  }
  virtual bool hasRange(const char*) const {
    // Has this argument a defined range (dummy interface always returns false)
    return false ;
  }


  enum ConstOpCode { Activate=0, DeActivate=1, ConfigChange=2, ValueChange=3 } ;
  enum CacheMode { Always=0, NotAdvised=1, Never=2 } ;
  enum OperMode { Auto=0, AClean=1, ADirty=2 } ;

  ////////////////////////////////////////////////////////////////////////////
  /// \anchor optimisationInterface
  /// \name Optimisation interface
  /// These functions allow RooFit to optimise a computation graph, to keep track
  /// of cached values, and to invalidate caches.
  /// @{

  // Cache mode optimization (tracks changes & do lazy evaluation vs evaluate always)
  virtual void optimizeCacheMode(const RooArgSet& observables) ;
  virtual void optimizeCacheMode(const RooArgSet& observables, RooArgSet& optNodes, RooLinkedList& processedNodes) ;


  // Find constant terms in expression
  bool findConstantNodes(const RooArgSet& observables, RooArgSet& cacheList) ;
  bool findConstantNodes(const RooArgSet& observables, RooArgSet& cacheList, RooLinkedList& processedNodes) ;


  // constant term optimization
  virtual void constOptimizeTestStatistic(ConstOpCode opcode, bool doAlsoTrackingOpt=true) ;

  virtual CacheMode canNodeBeCached() const { return Always ; }
  virtual void setCacheAndTrackHints(RooArgSet& /*trackNodes*/ ) {} ;

  // Dirty state accessor
  inline bool isShapeDirty() const {
    // Return true is shape has been invalidated by server value change
    return isDerived()?_shapeDirty:false ;
  }

  inline bool isValueDirty() const {
    // Returns true of value has been invalidated by server value change
    if (inhibitDirty()) return true ;
    switch(_operMode) {
    case AClean:
      return false ;
    case ADirty:
      return true ;
    case Auto:
      if (_valueDirty) return isDerived() ;
      return false ;
    }
    return true ; // we should never get here
  }

  inline bool isValueDirtyAndClear() const {
    // Returns true of value has been invalidated by server value change
    if (inhibitDirty()) return true ;
    switch(_operMode) {
    case AClean:
      return false ;
    case ADirty:
      return true ;
    case Auto:
      if (_valueDirty) {
   _valueDirty = false ;
   return isDerived();
      }
      return false ;
    }
    return true ; // But we should never get here
  }


  inline bool isValueOrShapeDirtyAndClear() const {
    // Returns true of value has been invalidated by server value change

    if (inhibitDirty()) return true ;
    switch(_operMode) {
    case AClean:
      return false ;
    case ADirty:
      return true ;
    case Auto:
      if (_valueDirty || _shapeDirty) {
   _shapeDirty = false ;
   _valueDirty = false ;
   return isDerived();
      }
      _shapeDirty = false ;
      _valueDirty = false ;
      return false ;
    }
    return true ; // But we should never get here
  }

  // Cache management
  void registerCache(RooAbsCache& cache) ;
  void unRegisterCache(RooAbsCache& cache) ;
  Int_t numCaches() const ;
  RooAbsCache* getCache(Int_t index) const ;

  /// Query the operation mode of this node.
  inline OperMode operMode() const { return _operMode  ; }
  /// Set the operation mode of this node.
  void setOperMode(OperMode mode, bool recurseADirty=true) ;

  // Dirty state modifiers
  /// Mark the element dirty. This forces a re-evaluation when a value is requested.
  void setValueDirty() {
    if (_operMode == Auto && !inhibitDirty())
      setValueDirty(nullptr);
  }
  /// Notify that a shape-like property (*e.g.* binning) has changed.
  void setShapeDirty() { setShapeDirty(nullptr); }

  const char* aggregateCacheUniqueSuffix() const ;
  virtual const char* cacheUniqueSuffix() const { return nullptr ; }

  void wireAllCaches() ;

  RooExpensiveObjectCache& expensiveObjectCache() const ;
  virtual void setExpensiveObjectCache(RooExpensiveObjectCache &cache) { _eocache = &cache; }

  /// Overwrite the current value stored in this object, making it look like this object computed that value.
  // \param[in] value Value to store.
  // \param[in] notifyClients Notify users of this object that they need to
  /// recompute their values.
  virtual void setCachedValue(double /*value*/, bool /*notifyClients*/ = true) {};

  /// @}
  ////////////////////////////////////////////////////////////////////////////

  //Debug hooks
  static void verboseDirty(bool flag) ;
  void printDirty(bool depth=true) const ;
  static void setDirtyInhibit(bool flag) ;

  void graphVizTree(const char* fileName, const char* delimiter="\n", bool useTitle=false, bool useLatex=false) ;
  void graphVizTree(std::ostream& os, const char* delimiter="\n", bool useTitle=false, bool useLatex=false) ;

  void printComponentTree(const char* indent="",const char* namePat=nullptr, Int_t nLevel=999) ;
  void printCompactTree(const char* indent="",const char* fileName=nullptr, const char* namePat=nullptr, RooAbsArg* client=nullptr) ;
  void printCompactTree(std::ostream& os, const char* indent="", const char* namePat=nullptr, RooAbsArg* client=nullptr) ;
  virtual void printCompactTreeHook(std::ostream& os, const char *ind="") ;

  // We want to support three cases here:
  //   * passing a RooArgSet
  //   * passing a RooArgList
  //   * passing an initializer list
  // Before, there was only an overload taking a RooArg set, which caused an
  // implicit creation of a RooArgSet when a RooArgList was passed. This needs
  // to be avoided, because if the passed RooArgList is owning the arguments,
  // this information will be lost with the copy. The solution is to have one
  // overload that takes a general RooAbsCollection, and one overload for
  // RooArgList that is invoked in the case of passing an initializer list.
  bool addOwnedComponents(const RooAbsCollection& comps) ;
  bool addOwnedComponents(RooAbsCollection&& comps) ;
  bool addOwnedComponents(RooArgList&& comps) ;

  // Transfer the ownership of one or more other RooAbsArgs to this RooAbsArg
  // via a `std::unique_ptr`.
  template<typename... Args_t>
  bool addOwnedComponents(std::unique_ptr<Args_t>... comps) {
    return addOwnedComponents({*comps.release() ...});
  }
  const RooArgSet* ownedComponents() const { return _ownedComponents ; }

  void setProhibitServerRedirect(bool flag) { _prohibitServerRedirect = flag ; }

  void setWorkspace(RooWorkspace &ws) { _myws = &ws; }
  inline RooWorkspace* workspace() const { return _myws; }

  RooAbsProxy* getProxy(Int_t index) const ;
  Int_t numProxies() const ;

  /// De-duplicated pointer to this object's name.
  /// This can be used for fast name comparisons.
  /// like `if (namePtr() == other.namePtr())`.
  /// \note TNamed::GetName() will return a pointer that's
  /// different for each object, but namePtr() always points
  /// to a unique instance.
  inline const TNamed* namePtr() const {
    return _namePtr ;
  }

  void SetName(const char* name) override ;
  void SetNameTitle(const char *name, const char *title) override ;

  virtual bool importWorkspaceHook(RooWorkspace &ws)
  {
     _myws = &ws;
     return false;
  };

  virtual bool canComputeBatchWithCuda() const { return false; }
  virtual bool isReducerNode() const { return false; }

  virtual void applyWeightSquared(bool flag);

  virtual std::unique_ptr<RooAbsArg> compileForNormSet(RooArgSet const &normSet, RooFit::Detail::CompileContext & ctx) const;

  virtual bool isCategory() const { return false; }

protected:
   void graphVizAddConnections(std::set<std::pair<RooAbsArg*,RooAbsArg*> >&) ;

   virtual void operModeHook() {} ;

   virtual void optimizeDirtyHook(const RooArgSet* /*obs*/) {} ;

   virtual bool isValid() const ;

   virtual void getParametersHook(const RooArgSet* /*nset*/, RooArgSet* /*list*/, bool /*stripDisconnected*/) const {} ;
   virtual void getObservablesHook(const RooArgSet* /*nset*/, RooArgSet* /*list*/) const {} ;

   void clearValueAndShapeDirty() const {
     _valueDirty=false ;
     _shapeDirty=false ;
   }

   void clearValueDirty() const {
     _valueDirty=false ;
   }
   void clearShapeDirty() const {
     _shapeDirty=false ;
   }

   /// Force element to re-evaluate itself when a value is requested.
   void setValueDirty(const RooAbsArg* source);
   /// Notify that a shape-like property (*e.g.* binning) has changed.
   void setShapeDirty(const RooAbsArg* source);

   virtual void ioStreamerPass2() ;
   static void ioStreamerPass2Finalize() ;


private:
  void addParameters(RooAbsCollection& params, const RooArgSet* nset = nullptr, bool stripDisconnected = true) const;
  std::size_t getParametersSizeEstimate(const RooArgSet* nset = nullptr) const;

  RefCountListLegacyIterator_t * makeLegacyIterator(const RefCountList_t& list) const;


 protected:
  friend class RooAbsReal;

  // Client-Server relation and Proxy management
  friend class RooAbsCollection ;
  friend class RooWorkspace ;
  friend class RooRealIntegral ;
  RefCountList_t _serverList       ; // list of server objects
  RefCountList_t _clientList; // list of client objects
  RefCountList_t _clientListShape; // subset of clients that requested shape dirty flag propagation
  RefCountList_t _clientListValue; // subset of clients that requested value dirty flag propagation

  RooRefArray _proxyList        ; // list of proxies

  std::vector<RooAbsCache*> _cacheList ; //! list of caches


  // Proxy management
  friend class RooArgProxy ;
  template<class RooCollection_t>
  friend class RooCollectionProxy;
  friend class RooHistPdf ;
  friend class RooHistFunc ;
  void registerProxy(RooArgProxy& proxy) ;
  void registerProxy(RooSetProxy& proxy) ;
  void registerProxy(RooListProxy& proxy) ;
  void unRegisterProxy(RooArgProxy& proxy) ;
  void unRegisterProxy(RooSetProxy& proxy) ;
  void unRegisterProxy(RooListProxy& proxy) ;
  void setProxyNormSet(const RooArgSet* nset) ;

  // Attribute list
  std::set<std::string> _boolAttrib ; // Boolean attributes
  std::map<std::string,std::string> _stringAttrib ; // String attributes
  std::set<std::string> _boolAttribTransient ; //! Transient boolean attributes (not copied in ctor)

  void printAttribList(std::ostream& os) const;

  // Hooks for RooTreeData interface
  friend class RooCompositeDataStore ;
  friend class RooTreeDataStore ;
  friend class RooVectorDataStore ;
  friend class RooDataSet ;
  friend class RooRealMPFE ;
  virtual void syncCache(const RooArgSet* nset=nullptr) = 0 ;
  virtual void copyCache(const RooAbsArg* source, bool valueOnly=false, bool setValDirty=true) = 0 ;

  virtual void attachToTree(TTree& t, Int_t bufSize=32000) = 0 ;
  virtual void attachToVStore(RooVectorDataStore& vstore) = 0 ;
  /// Attach this argument to the data store such that it reads data from there.
  void attachToStore(RooAbsDataStore& store) ;

  virtual void setTreeBranchStatus(TTree& t, bool active) = 0 ;
  virtual void fillTreeBranch(TTree& t) = 0 ;
  TString cleanBranchName() const ;

  // Global
  friend std::ostream& operator<<(std::ostream& os, const RooAbsArg &arg);
  friend std::istream& operator>>(std::istream& is, RooAbsArg &arg) ;
  friend void RooRefArray::Streamer(TBuffer&);

  struct ProxyListCache {
    std::vector<RooAbsProxy*> cache;
    bool isDirty = true;
  };
  ProxyListCache _proxyListCache; //! cache of the list of proxies. Avoids type casting.

  // Debug stuff
  static bool _verboseDirty ; // Static flag controlling verbose messaging for dirty state changes
  static bool _inhibitDirty ; // Static flag controlling global inhibit of dirty state propagation
  bool _deleteWatch = false; //! Delete watch flag

  bool inhibitDirty() const ;

 public:
  void setLocalNoDirtyInhibit(bool flag) const { _localNoInhibitDirty = flag ; }
  bool localNoDirtyInhibit() const { return _localNoInhibitDirty ; }

  /// Returns the token for retrieving results in the BatchMode. For internal use only.
  std::size_t dataToken() const { return _dataToken; }
  bool hasDataToken() const { return _dataToken != std::numeric_limits<std::size_t>::max(); }
  void setDataToken(std::size_t index);
  void resetDataToken() { _dataToken = std::numeric_limits<std::size_t>::max(); }
 protected:


  mutable bool _valueDirty = true;  // Flag set if value needs recalculating because input values modified
  mutable bool _shapeDirty = true;  // Flag set if value needs recalculating because input shapes modified

  mutable OperMode _operMode = Auto; // Dirty state propagation mode
  mutable bool _fast = false; // Allow fast access mode in getVal() and proxies

  // Owned components
  RooArgSet* _ownedComponents = nullptr; //! Set of owned component

  mutable bool _prohibitServerRedirect = false; //! Prohibit server redirects -- Debugging tool

  mutable RooExpensiveObjectCache* _eocache{nullptr}; //! Pointer to global cache manager for any expensive components created by this object

  mutable const TNamed * _namePtr = nullptr; //! De-duplicated name pointer. This will be equal for all objects with the same name.
  bool _isConstant = false; //! Cached isConstant status

  mutable bool _localNoInhibitDirty = false; //! Prevent 'AlwaysDirty' mode for this node

/*   RooArgSet _leafNodeCache ; //! Cached leaf nodes */
/*   RooArgSet _branchNodeCache //! Cached branch nodes     */

  mutable RooWorkspace *_myws = nullptr; //! In which workspace do I live, if any

  std::size_t _dataToken = std::numeric_limits<std::size_t>::max(); //! Set by the RooFitDriver for this arg to retrieve its result in the run context

  /// \cond ROOFIT_INTERNAL
  // Legacy streamers need the following statics:
  friend class RooFitResult;

 public:
  // Used internally for schema evolution.
  static void addToIoEvoList(RooAbsArg *newObj, TRefArray const &onfileProxyList);
  /// \endcond

 private:
  void substituteServer(RooAbsArg *oldServer, RooAbsArg *newServer);
  bool callRedirectServersHook(RooAbsCollection const& newSet, bool mustReplaceAll, bool nameChange, bool isRecursionStep);

  ClassDefOverride(RooAbsArg,9) // Abstract variable
};

std::ostream& operator<<(std::ostream& os, const RooAbsArg &arg);
std::istream& operator>>(std::istream& is, RooAbsArg &arg);


#endif
