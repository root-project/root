/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooAbsArg.rdl,v 1.2 2001/03/15 23:19:11 verkerke Exp $
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
#include "TNamed.h"
#include "THashList.h"

class TTree ;
class RooArgSet ;
class RooDataSet ;

class RooAbsArg : public TNamed {
public:

  // Constructors, cloning and assignment
  RooAbsArg() ;
  virtual ~RooAbsArg();
  RooAbsArg(const char *name, const char *title);
  RooAbsArg(const RooAbsArg& other) ;
  virtual TObject* Clone() ;
  virtual RooAbsArg& operator=(RooAbsArg& other) ; 

  // Accessors to client-server relation information 
  Bool_t isDerived() const { return _serverList.First()?kTRUE:kFALSE; }
  Bool_t dependsOn(RooArgSet& serverList) ;
  Bool_t dependsOn(RooAbsArg& server) ;
  inline TIterator* clientIterator() { return _clientList.MakeIterator() ; }
  inline TIterator* serverIterator() { return _serverList.MakeIterator() ; }
  
  // I/O streaming interface (machine readable)
  virtual Bool_t readFromStream(istream& is, Bool_t compact, Bool_t verbose=kFALSE) = 0 ;
  virtual void writeToStream(ostream& os, Bool_t compact) = 0 ;

  // Printing interface (human readable)
  enum PrintOption { OneLine=0, Standard=1, Shape=2, Verbose=3 } ;
  virtual void printToStream(ostream& stream, PrintOption opt=Standard) ;
  void print(PrintOption opt=Standard) { printToStream(cout,opt) ; }

  // Accessors to attributes
  void setAttribute(Text_t* name, Bool_t value=kTRUE) ;
  Bool_t getAttribute(Text_t* name) const ;
  inline TIterator* attribIterator() { return _attribList.MakeIterator() ; }

  //Debug hooks
  void printLinks() ;
  static void verboseDirty(Bool_t flag) { _verboseDirty = flag ; }

protected:

  // Client-Server relatation management 
  friend class RooArgSet ;
  THashList _clientList ; //! do not persist (or clone)
  THashList _serverList ; //! do not persist (or clone)
  Bool_t redirectServers(RooArgSet& newServerList, Bool_t mustReplaceAll=kFALSE) ;
  virtual Bool_t redirectServersHook(RooArgSet& newServerList, Bool_t mustReplaceAll) {} ;
  void addServer(RooAbsArg& server) ;
  void removeServer(RooAbsArg& server) ;
	
  // Attribute list
  THashList _attribList ;
  void printAttribList(ostream& os) ;

  // Value and Shape dirty state information mananagement
  Bool_t _valueDirty ;
  Bool_t _shapeDirty ;
  void setValueDirty(Bool_t flag=kTRUE) ; 
  void setShapeDirty(Bool_t flag=kTRUE) ; 
  virtual Bool_t isValueDirty() const { return _valueDirty ; } 
  virtual Bool_t isShapeDirty() const { return _shapeDirty ; } 
  void raiseClientValueDirtyFlags() ;
  void raiseClientShapeDirtyFlags() ;

  // Hooks for RooDataSet interface
  friend class RooDataSet ;
  virtual void attachToTree(TTree& t, Int_t bufSize=32000) ;
  virtual Bool_t isValid() ;

  // Global   
  friend ostream& operator<<(ostream& os, RooAbsArg &arg);  
  friend istream& operator>>(istream& is, RooAbsArg &arg) ;

  // Debug stuff
  static Bool_t _verboseDirty ;

  ClassDef(RooAbsArg,1) // a real-valued variable and its value
};

ostream& operator<<(ostream& os, RooAbsArg &arg);  
istream& operator>>(istream& is, RooAbsArg &arg) ;

#endif
