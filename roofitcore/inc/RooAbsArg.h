/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooAbsArg.rdl,v 1.11 2001/03/29 22:37:39 verkerke Exp $
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

class TTree ;
class RooArgSet ;
class RooDataSet ;

class RooAbsArg : public TNamed {
public:

  // Constructors, cloning and assignment
  RooAbsArg();
  virtual ~RooAbsArg();
  RooAbsArg(const char *name, const char *title);
  RooAbsArg(const RooAbsArg& other) ;
  RooAbsArg(const char* name, const RooAbsArg& other) ;

  // Accessors to client-server relation information 
  Bool_t isDerived() const { return _serverList.First()?kTRUE:kFALSE; }
  Bool_t dependsOn(const RooArgSet& serverList) const ;
  Bool_t dependsOn(const RooAbsArg& server) const ;
  inline TIterator* clientIterator() const { return _clientList.MakeIterator() ; }
  inline TIterator* serverIterator() const { return _serverList.MakeIterator() ; }
  
  // I/O streaming interface (machine readable)
  virtual Bool_t readFromStream(istream& is, Bool_t compact, Bool_t verbose=kFALSE) = 0 ;
  virtual void writeToStream(ostream& os, Bool_t compact) const = 0 ;

  // Printing interface (human readable)
  enum PrintOption { OneLine=0, Standard=1, Shape=2, Verbose=3 } ;
  virtual void printToStream(ostream& stream, PrintOption opt=Standard) const ;
  inline void Print(Option_t * = 0) const ;

  // Accessors to attributes
  void setAttribute(Text_t* name, Bool_t value=kTRUE) ;
  Bool_t getAttribute(Text_t* name) const ;
  inline TIterator* attribIterator() { return _attribList.MakeIterator() ; }

  //Debug hooks
  static void verboseDirty(Bool_t flag) { _verboseDirty = flag ; }

protected:

  virtual RooAbsArg& operator=(const RooAbsArg& other) ; 
  void initCopy(const RooAbsArg& other) ;

  // Client-Server relatation management 
  friend class RooArgSet ;
  THashList _clientList      ; //! complete client list
  THashList _clientListShape ; //! clients that requested shape dirty flag propagation
  THashList _clientListValue ; //! clients that requested value dirty flag propagation
  THashList _serverList      ; //! do not persist (or clone)
  Bool_t redirectServers(RooArgSet& newServerList, Bool_t mustReplaceAll=kFALSE) ;
  virtual Bool_t redirectServersHook(RooArgSet& newServerList, Bool_t mustReplaceAll) {} ;
  void addServer(RooAbsArg& server, Bool_t valueProp=kTRUE, Bool_t shapeProp=kTRUE) ;
  void changeServer(RooAbsArg& server, Bool_t valueProp, Bool_t shapeProp) ;
  void removeServer(RooAbsArg& server) ;
	
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
