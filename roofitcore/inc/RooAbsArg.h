/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooAbsArg.rdl,v 1.90 2005/12/08 13:19:54 wverkerke Exp $
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

#include "Riostream.h"
#include <assert.h>
#include "TNamed.h"
#include "THashList.h"
#include "RooPrintable.h"
#include "RooRefCountList.h"

class TTree ;
class RooArgSet ;
class RooAbsCollection ;
class RooTreeData ;
class RooAbsData ;
class RooAbsProxy ;
class RooArgProxy ;
class RooSetProxy ;
class RooListProxy ;

class RooAbsArg : public TNamed, public RooPrintable {
public:

  // Constructors, cloning and assignment
  RooAbsArg() ;
  virtual ~RooAbsArg();
  RooAbsArg(const char *name, const char *title);
  RooAbsArg(const RooAbsArg& other, const char* name=0) ;
  virtual TObject* clone(const char* newname) const = 0 ;
  virtual TObject* Clone(const char* newname=0) const { 
//     cout << "RooAbsArg::Clone(" << GetName() << "," << this << ") newname=" << (newname?newname:"<none>") << endl ;
    return clone(newname?newname:GetName()) ; 
  }

  // Accessors to client-server relation information 
  virtual Bool_t isDerived() const { return _serverList.GetSize()?kTRUE:kFALSE; }
  Bool_t isCloneOf(const RooAbsArg& other) const ; 
  Bool_t dependsOn(const RooAbsCollection& serverList, const RooAbsArg* ignoreArg=0) const ;
  Bool_t dependsOn(const RooAbsArg& server, const RooAbsArg* ignoreArg=0) const ;
  Bool_t overlaps(const RooAbsArg& testArg) const ;
  inline TIterator* clientIterator() const { return _clientList.MakeIterator() ; }
  inline TIterator* valueClientIterator() const { return _clientListValue.MakeIterator() ; }
  inline TIterator* shapeClientIterator() const { return _clientListShape.MakeIterator() ; }
  inline TIterator* serverIterator() const { return _serverList.MakeIterator() ; }
  inline RooAbsArg* findServer(const char *name) const { return (RooAbsArg*)_serverList.FindObject(name); }
  inline RooAbsArg* findServer(const RooAbsArg& arg) const { return (RooAbsArg*)_serverList.FindObject(&arg); }
  inline RooAbsArg* findServer(Int_t index) const { return (RooAbsArg*)_serverList.At(index); }
  inline Bool_t isValueServer(const RooAbsArg& arg) const { return _clientListValue.FindObject(&arg)?kTRUE:kFALSE ; }
  inline Bool_t isValueServer(const char* name) const { return _clientListValue.FindObject(name)?kTRUE:kFALSE ; }
  inline Bool_t isShapeServer(const RooAbsArg& arg) const { return _clientListShape.FindObject(&arg)?kTRUE:kFALSE ; }
  inline Bool_t isShapeServer(const char* name) const { return _clientListShape.FindObject(name)?kTRUE:kFALSE ; }
  void leafNodeServerList(RooAbsCollection* list, const RooAbsArg* arg=0) const ;
  void branchNodeServerList(RooAbsCollection* list, const RooAbsArg* arg=0) const ;
  void treeNodeServerList(RooAbsCollection* list, const RooAbsArg* arg=0, 
			  Bool_t doBranch=kTRUE, Bool_t doLeaf=kTRUE, Bool_t valueOnly=kFALSE) const ;
  
  // Is this object a fundamental type that can be added to a dataset?
  // Fundamental-type subclasses override this method to return kTRUE.
  // Note that this test is subtlely different from the dynamic isDerived()
  // test, e.g. a constant is not derived but is also not fundamental.
  inline virtual Bool_t isFundamental() const { return kFALSE; }
  // Create a fundamental-type object that stores our type of value. The
  // created object will have a valid value, but not necessarily the same
  // as our value. The caller is responsible for deleting the returned object.
  virtual RooAbsArg *createFundamental(const char* newname=0) const = 0;

  // Is this argument an l-value, ie, can it appear on the left-hand side
  // of an assignment expression? LValues are also special since they can
  // potentially be analytically integrated and generated.
  inline virtual Bool_t isLValue() const { return kFALSE; }

  // Parameter & observable interpretation of servers
  friend class RooProdPdf ;
  friend class RooAddPdf ;
  RooArgSet* getVariables() const ;
  RooArgSet* getParameters(const RooAbsData* data) const ;
  RooArgSet* getParameters(const RooAbsData& data) const { return getParameters(&data) ; }
  RooArgSet* getParameters(const RooArgSet& set) const { return getParameters(&set) ; }
  virtual RooArgSet* getParameters(const RooArgSet* depList) const ;
  RooArgSet* getObservables(const RooArgSet& set) const { return getObservables(&set) ; }
  RooArgSet* getObservables(const RooAbsData* data) const ;
  RooArgSet* getObservables(const RooAbsData& data) const { return getObservables(&data) ; }
  virtual RooArgSet* getObservables(const RooArgSet* depList) const ;
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

  // I/O streaming interface (machine readable)
  virtual Bool_t readFromStream(istream& is, Bool_t compact, Bool_t verbose=kFALSE) = 0 ;
  virtual void writeToStream(ostream& os, Bool_t compact) const = 0 ;

  // Printing interface (human readable)
  virtual void printToStream(ostream& os, PrintOption opt= Standard, TString indent= "") const;
  inline virtual void Print(Option_t *options= 0) const {
    printToStream(defaultStream(),parseOptions(options));
  }

  // Accessors to attributes
  void setAttribute(const Text_t* name, Bool_t value=kTRUE) ;
  Bool_t getAttribute(const Text_t* name) const ;
  inline TIterator* attribIterator() const { return _attribList.MakeIterator() ; }
  inline Bool_t isConstant() const { return getAttribute("Constant") ; }
  RooLinkedList getCloningAncestors() const ;

  // Sorting
  Int_t Compare(const TObject* other) const ;
  virtual Bool_t IsSortable() const { return kTRUE ; }

  //Debug hooks
  static void verboseDirty(Bool_t flag) { _verboseDirty = flag ; }
  static void copyList(TList& dest, const TList& source) ;
  void printDirty(Bool_t depth=kTRUE) const ;
  static void setDirtyInhibit(Bool_t flag) { _inhibitDirty = flag ; }

  virtual Bool_t operator==(const RooAbsArg& other) = 0 ;

  // Formatting control
  static void nameFieldLength(Int_t newLen) { _nameLength = newLen>0 ? newLen : 0 ; }

  // Range management
  virtual Bool_t inRange(const char*) const { return kTRUE ; }
  virtual Bool_t hasRange(const char*) const { return kFALSE ; }

  enum ConstOpCode { Activate=0, DeActivate=1, ConfigChange=2, ValueChange=3 } ;
  

  friend class RooMinuit ;
  virtual void constOptimize(ConstOpCode opcode) ;


  void printCompactTree(const char* indent="",const char* fileName=0, const char* namePat=0) ;
  void printCompactTree(ostream& os, const char* indent="", const char* namePat=0) ;
  virtual void printCompactTreeHook(ostream& os, const char *ind="") ;

  inline void setDeleteWatch(Bool_t flag=kTRUE) { _deleteWatch = flag ; } ;
  Bool_t deleteWatch() const { return _deleteWatch ; }


protected:

  friend class RooExtendPdf ;
  friend class RooRealIntegral ;
  friend class RooAbsReal ;

  enum OperMode { Auto=0, AClean=1, ADirty=2 } ;
  void setOperMode(OperMode mode, Bool_t recurseADirty=kTRUE) ; 
  virtual void operModeHook() {} ;
  inline OperMode operMode() const { return _operMode ; }

  virtual Bool_t isValid() const ;

  virtual void getParametersHook(const RooArgSet* /*nset*/, RooArgSet* /*list*/) const {} ;
  virtual void getObservablesHook(const RooArgSet* /*nset*/, RooArgSet* /*list*/) const {} ;

  // Dirty state accessor/modifiers
  inline Bool_t isShapeDirty() const { return isDerived()?_shapeDirty:kFALSE ; } 
  inline Bool_t isValueDirty() const { 
    if (_inhibitDirty) return kTRUE ;
    switch(_operMode) {
    case AClean: return kFALSE ;
    case ADirty: return kTRUE ;
    case Auto: return (isDerived()?_valueDirty:kFALSE) ;
    }
    return kTRUE ; // we should never get here
  }
  inline void setValueDirty() const { setValueDirty(0) ; }
  inline void setShapeDirty() const { setShapeDirty(0) ; } 
  inline void clearValueDirty() const { 
    if (_verboseDirty) cout << "RooAbsArg::clearValueDirty(" << GetName() 
			    << "): dirty flag " << (_valueDirty?"":"already ") << "cleared" << endl ;
    _valueDirty=kFALSE ; 
  }
  inline void clearShapeDirty() const { 
    if (_verboseDirty) cout << "RooAbsArg::clearShapeDirty(" << GetName() 
			    << "): dirty flag " << (_shapeDirty?"":"already ") << "cleared" << endl ;
    _shapeDirty=kFALSE ; 
  }

  // Client-Server relatation and Proxy management 
  friend class RooArgSet ;
  friend class RooAbsCollection ;
  friend class RooCustomizer ;
  RooRefCountList _serverList       ; //! list of server objects
  RooRefCountList _clientList       ; //! list of client objects
  RooRefCountList _clientListShape  ; //! subset of clients that requested shape dirty flag propagation
  RooRefCountList _clientListValue  ; //! subset of clients that requested value dirty flag propagation
  TList _proxyList        ; //! list of proxies
  TIterator* _clientShapeIter ; //! Iterator over _clientListShape 
  TIterator* _clientValueIter ; //! Iterator over _clientListValue 

  // Server redirection interface
  friend class RooAcceptReject;
  friend class RooGenContext;
  friend class RooResolutionModel ;
  friend class RooSimultaneous ;
  friend class RooSimGenContext ;  
  friend class RooEffGenContext ;  
  friend class RooSimPdfBuilder ;
  friend class RooAbsOptGoodnessOfFit ;
  friend class RooAbsPdf ;
  friend class RooGenProdProj ;

  Bool_t redirectServers(const RooAbsCollection& newServerList, Bool_t mustReplaceAll=kFALSE, Bool_t nameChange=kFALSE, Bool_t isRecursionStep=kFALSE) ;
  Bool_t recursiveRedirectServers(const RooAbsCollection& newServerList, Bool_t mustReplaceAll=kFALSE, Bool_t nameChange=kFALSE) ;
  virtual Bool_t redirectServersHook(const RooAbsCollection& /*newServerList*/, Bool_t /*mustReplaceAll*/, Bool_t /*nameChange*/, Bool_t /*isRecursive*/) { return kFALSE ; } ;
  virtual void serverNameChangeHook(const RooAbsArg* /*oldServer*/, const RooAbsArg* /*newServer*/) { } ;

  friend class RooFormula ;
  void addServer(RooAbsArg& server, Bool_t valueProp=kTRUE, Bool_t shapeProp=kFALSE) ;
  void addServerList(RooAbsCollection& serverList, Bool_t valueProp=kTRUE, Bool_t shapeProp=kFALSE) ;
  void replaceServer(RooAbsArg& oldServer, RooAbsArg& newServer, Bool_t valueProp, Bool_t shapeProp) ;
  void changeServer(RooAbsArg& server, Bool_t valueProp, Bool_t shapeProp) ;
  void removeServer(RooAbsArg& server, Bool_t force=kFALSE) ;
  RooAbsArg *findNewServer(const RooAbsCollection &newSet, Bool_t nameChange) const;

  // Proxy management
  friend class RooAddModel ;
  friend class RooArgProxy ;
  friend class RooSetProxy ;
  friend class RooListProxy ;
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
  THashList _attribList ; // List of string attributes
  void printAttribList(ostream& os) const;

  // Hooks for RooTreeData interface
  friend class RooTreeData ;
  friend class RooDataSet ;
  friend class RooRealMPFE ;
  virtual void syncCache(const RooArgSet* nset=0) = 0 ;
  virtual void copyCache(const RooAbsArg* source) = 0 ;
  virtual void attachToTree(TTree& t, Int_t bufSize=32000) = 0 ;
  virtual void setTreeBranchStatus(TTree& t, Bool_t active) = 0 ;
  virtual void fillTreeBranch(TTree& t) = 0 ;
  TString cleanBranchName() const ;
  UInt_t crc32(const char* data) const ;

  // Global   
  friend ostream& operator<<(ostream& os, const RooAbsArg &arg);  
  friend istream& operator>>(istream& is, RooAbsArg &arg) ;
  
  // Debug stuff
  static Bool_t _verboseDirty ; // Static flag controlling verbose messaging for dirty state changes
  static Bool_t _inhibitDirty ; // Static flag controlling global inhibit of dirty state propagation
  Bool_t _deleteWatch ; //! Delete watch flag 

  static Int_t _nameLength ;

private:

  // Value and Shape dirty state bits
  void setValueDirty(const RooAbsArg* source) const ; 
  void setShapeDirty(const RooAbsArg* source) const ; 
  mutable Bool_t _valueDirty ;  // Flag set if value needs recalculating because input values modified
  mutable Bool_t _shapeDirty ;  // Flag set if value needs recalculating because input shapes modified
  mutable OperMode _operMode ; // Dirty state propagation mode

  ClassDef(RooAbsArg,1) // Abstract variable
};

ostream& operator<<(ostream& os, const RooAbsArg &arg);  
istream& operator>>(istream& is, RooAbsArg &arg) ;

#endif
