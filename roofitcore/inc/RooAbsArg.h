/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooAbsArg.rdl,v 1.63 2001/12/02 23:25:54 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/
#ifndef ROO_ABS_ARG
#define ROO_ABS_ARG

#include <iostream.h>
#include <assert.h>
#include "TNamed.h"
#include "THashList.h"
#include "TObjArray.h"
#include "TStopwatch.h"
#include "RooFitCore/RooPrintable.hh"

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
  Bool_t isDerived() const { return _serverList.First()?kTRUE:kFALSE; }
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
			  Bool_t doBranch=kTRUE, Bool_t doLeaf=kTRUE) const ;
  
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

  // Parameter & dependents interpretation of servers
  friend class RooProdPdf ;
  RooArgSet* getParameters(const RooAbsData* set) const ;
  RooArgSet* getParameters(const RooArgSet& set) const { return getParameters(&set) ; }
  virtual RooArgSet* getParameters(const RooArgSet* depList) const ;
  RooArgSet* getDependents(const RooArgSet& set) const { return getDependents(&set) ; }
  RooArgSet* getDependents(const RooAbsData* set) const ;
  virtual RooArgSet* getDependents(const RooArgSet* depList) const ;
  RooArgSet* getComponents() const ;	
  Bool_t dependentOverlaps(const RooAbsData* dset, const RooAbsArg& testArg) const ;
  Bool_t dependentOverlaps(const RooArgSet* depList, const RooAbsArg& testArg) const ;
  virtual Bool_t checkDependents(const RooArgSet* nset) const ;
  Bool_t recursiveCheckDependents(const RooArgSet* nset) const ;
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
  inline TIterator* attribIterator() { return _attribList.MakeIterator() ; }
  inline Bool_t isConstant() const { return getAttribute("Constant") ; }

  // Sorting
  Int_t Compare(const TObject* other) const ;
  virtual Bool_t IsSortable() const { return kTRUE ; }

  //Debug hooks
  static void verboseDirty(Bool_t flag) { _verboseDirty = flag ; }
  static void copyList(TList& dest, const TList& source) ;
  void printDirty(Bool_t depth=kTRUE) const ;

  // Universal assignment operators to fundamentals
  virtual RooAbsArg& operator=(Int_t ival) ;
  virtual RooAbsArg& operator=(Double_t fval) ;
  virtual RooAbsArg& operator=(const char* cval) ;

  // Formatting control
  static void nameFieldLength(Int_t newLen) { _nameLength = newLen>0 ? newLen : 0 ; }
  
protected:

  friend class RooExtendPdf ;
  friend class RooRealIntegral ;
  friend class RooAbsReal ;
  enum OperMode { Auto=0, AClean=1, ADirty=2 } ;
  void setOperMode(OperMode mode) { _operMode = mode ; operModeHook() ; }
  virtual void operModeHook() {} ;
  inline OperMode operMode() const { return _operMode ; }

  virtual Bool_t isValid() const ;

  virtual void getParametersHook(const RooArgSet* nset, RooArgSet* list) const {} ;
  virtual void getDependentsHook(const RooArgSet* nset, RooArgSet* list) const {} ;

  // Dirty state accessor/modifiers
  inline Bool_t isShapeDirty() const { return isDerived()?_shapeDirty:kFALSE ; } 
  inline Bool_t isValueDirty() const { 
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
  friend class RooFitContext ;
  TList _serverList       ; //! list of server objects
  TList _clientList       ; //! list of client objects
  TList _clientListShape  ; //! subset of clients that requested shape dirty flag propagation
  TList _clientListValue  ; //! subset of clients that requested value dirty flag propagation
  TList _proxyList        ; //! list of proxies
  TIterator* _clientShapeIter ; //! Iterator over _clientListShape 
  TIterator* _clientValueIter ; //! Iterator over _clientListValue 

  // Server redirection interface
  friend class RooAcceptReject;
  friend class RooGenContext;
  friend class RooResolutionModel ;
  friend class RooSimultaneous ;
  friend class RooSimGenContext ;
  friend class RooSimPdfBuilder ;
  Bool_t redirectServers(const RooAbsCollection& newServerList, Bool_t mustReplaceAll=kFALSE, Bool_t nameChange=kFALSE) ;
  Bool_t recursiveRedirectServers(const RooAbsCollection& newServerList, Bool_t mustReplaceAll=kFALSE, Bool_t nameChange=kFALSE) ;
  virtual Bool_t redirectServersHook(const RooAbsCollection& newServerList, Bool_t mustReplaceAll, Bool_t nameChange) { return kFALSE ; } ;
  virtual void serverNameChangeHook(const RooAbsArg* oldServer, const RooAbsArg* newServer) { } ;

  friend class RooFormula ;
  void addServer(RooAbsArg& server, Bool_t valueProp=kTRUE, Bool_t shapeProp=kFALSE) ;
  void addServerList(RooAbsCollection& serverList, Bool_t valueProp=kTRUE, Bool_t shapeProp=kFALSE) ;
  void changeServer(RooAbsArg& server, Bool_t valueProp, Bool_t shapeProp) ;
  void removeServer(RooAbsArg& server) ;
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
  virtual void syncCache(const RooArgSet* nset=0) = 0 ;
  virtual void copyCache(const RooAbsArg* source) = 0 ;
  virtual void attachToTree(TTree& t, Int_t bufSize=32000) = 0 ;
  virtual void fillTreeBranch(TTree& t) = 0 ;

  // Global   
  friend ostream& operator<<(ostream& os, const RooAbsArg &arg);  
  friend istream& operator>>(istream& is, RooAbsArg &arg) ;
  
  // Debug stuff
  static Bool_t _verboseDirty ; // Static flag controlling verbose messaging for dirty state changes

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
