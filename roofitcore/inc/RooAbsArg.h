/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooAbsArg.rdl,v 1.18 2001/04/20 01:51:38 verkerke Exp $
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
#include "RooFitCore/RooPrintable.hh"

class TTree ;
class RooArgSet ;
class RooDataSet ;
class RooArgProxy ;

class RooAbsArg : public TNamed, public RooPrintable {
public:

  // Constructors, cloning and assignment
  RooAbsArg() ;
  virtual ~RooAbsArg();
  RooAbsArg(const char *name, const char *title);
  RooAbsArg(const RooAbsArg& other, const char* name=0) ;
  virtual TObject* clone() const = 0 ;
  virtual TObject* Clone(const char* newname=0) const { return clone() ; }

  // Accessors to client-server relation information 
  Bool_t isDerived() const { return _serverList.First()?kTRUE:kFALSE; }
  Bool_t dependsOn(const RooArgSet& serverList) const ;
  Bool_t dependsOn(const RooAbsArg& server) const ;
  Bool_t overlaps(const RooAbsArg& testArg) const ;
  inline TIterator* clientIterator() const { return _clientList.MakeIterator() ; }
  inline TIterator* serverIterator() const { return _serverList.MakeIterator() ; }
  void leafNodeServerList(RooArgSet* list, const RooAbsArg* arg=0) const ;
  void branchNodeServerList(RooArgSet* list, const RooAbsArg* arg=0) const ;
  void treeNodeServerList(RooArgSet* list, const RooAbsArg* arg=0, 
			  Bool_t doBranch=kTRUE, Bool_t doLeaf=kTRUE) const ;
  
  // Parameter & dependents interpretation of servers
  RooArgSet* getParameters(const RooDataSet* set) const ;
  RooArgSet* getDependents(const RooDataSet* set) const ;
  Bool_t dependentOverlaps(const RooDataSet* dset, const RooAbsArg& testArg) const ;
  virtual Bool_t checkDependents(const RooDataSet* set) const ;

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

  //Debug hooks
  static void verboseDirty(Bool_t flag) { _verboseDirty = flag ; }
  static void copyList(TList& dest, const TList& source) ;

protected:

  virtual RooAbsArg& operator=(const RooAbsArg& other) ; 

  // Client-Server relatation and Proxy management 
  friend class RooArgSet ;
  THashList _clientList      ; //! complete client list
  THashList _clientListShape ; //! clients that requested shape dirty flag propagation
  THashList _clientListValue ; //! clients that requested value dirty flag propagation
  THashList _serverList      ; //! do not persist (or clone)
  TObjArray _proxyArray      ; //! do not persist (or clone)

  Bool_t redirectServers(const RooArgSet& newServerList, Bool_t mustReplaceAll=kFALSE) ;
  Bool_t recursiveRedirectServers(const RooArgSet& newServerList, Bool_t mustReplaceAll=kFALSE) ;
  virtual Bool_t redirectServersHook(const RooArgSet& newServerList, Bool_t mustReplaceAll) { return kFALSE ; } ;

  void addServer(RooAbsArg& server, Bool_t valueProp=kTRUE, Bool_t shapeProp=kFALSE) ;
  void addServerList(RooArgSet& serverList, Bool_t valueProp=kTRUE, Bool_t shapeProp=kFALSE) ;
  void changeServer(RooAbsArg& server, Bool_t valueProp, Bool_t shapeProp) ;
  void removeServer(RooAbsArg& server) ;

  friend class RooArgProxy ;
  void registerProxy(RooArgProxy& proxy) ;
  RooArgProxy& getProxy(Int_t index) const ;
  Int_t numProxies() const ;
	
  // Attribute list
  THashList _attribList ;
  void printAttribList(ostream& os) const;

  // Dirty state accessor/modifiers
  // 
  // The dirty state accessors can be overriden by 
  // fundamental subclasses to always return false
  virtual Bool_t isValueDirty() const { return _valueDirty ; } 
  virtual Bool_t isShapeDirty() const { return _shapeDirty ; } 
  void setValueDirty(Bool_t flag=kTRUE) const { setValueDirty(flag,0) ; }
  void setShapeDirty(Bool_t flag=kTRUE) const { setShapeDirty(flag,0) ; } 

  // Hooks for RooDataSet interface
  friend class RooDataSet ;
  virtual void attachToTree(TTree& t, Int_t bufSize=32000) ;
  virtual void postTreeLoadHook() {} ;
  virtual Bool_t isValid() const ;

  // Global   
  friend ostream& operator<<(ostream& os, const RooAbsArg &arg);  
  friend istream& operator>>(istream& is, RooAbsArg &arg) ;
  
  // Debug stuff
  static Bool_t _verboseDirty ;

private:

  // Value and Shape dirty state bits
  void setValueDirty(Bool_t flag, const RooAbsArg* source) const ; 
  void setShapeDirty(Bool_t flag, const RooAbsArg* source) const ; 
  mutable Bool_t _valueDirty ;
  mutable Bool_t _shapeDirty ;

  ClassDef(RooAbsArg,1) // a real-valued variable and its value
};

ostream& operator<<(ostream& os, const RooAbsArg &arg);  
istream& operator>>(istream& is, RooAbsArg &arg) ;

#endif
