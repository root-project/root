/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooAbsArg.rdl,v 1.1 2001/03/14 02:45:47 verkerke Exp $
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
  RooAbsArg() ;
  virtual ~RooAbsArg();

  RooAbsArg(const char *name, const char *title);
  RooAbsArg(const RooAbsArg& other) ;
  virtual TObject* Clone() ;

  virtual RooAbsArg& operator=(RooAbsArg& other) ; 
  virtual Bool_t isDerived() const { return kFALSE; }
  
  // I/O streaming interface
  virtual Bool_t readFromStream(istream& is, Bool_t compact, Bool_t verbose=kFALSE) = 0 ;
  virtual void writeToStream(ostream& os, Bool_t compact) = 0 ;

  virtual void PrintToStream(ostream& stream, Option_t* = 0) ;
  void Print(Option_t* opt= 0) { PrintToStream(cout,opt) ; }
  void printLinks() ;

  void setAttribute(Text_t* name, Bool_t value=kTRUE) ;
  Bool_t getAttribute(Text_t* name) const ;


  inline TIterator* clientIterator() { return _clientList.MakeIterator() ; }
  inline TIterator* serverIterator() { return _serverList.MakeIterator() ; }
  inline TIterator* attribIterator() { return _attribList.MakeIterator() ; }

  static void verboseDirty(Bool_t flag) { _verboseDirty = flag ; }

  virtual Bool_t redirectServers(RooArgSet& newServerList, Bool_t mustReplaceAll=kFALSE) ;
  Bool_t dependsOn(RooArgSet& serverList) ;
  Bool_t dependsOn(RooAbsArg& server) ;

protected:

  // This function allows RooDataSet to directly modify the contents of a RooAbsArg
  virtual void attachToTree(TTree& t, Int_t bufSize=32000) ;
  // This function is used by RooDataSet to check that the above didn't result in an
  // illegal value or undefined state
  virtual Bool_t isValid() ;

  // Classes that don't have dirty state information should override this function 
  Bool_t _dirty ;
  void setDirty(Bool_t flag=kTRUE) ; 
  virtual Bool_t isDirty() const { return _dirty ; } 
  void raiseClientDirtyFlags() ;

  THashList _attribList ;
  THashList _clientList ; //! do not persist (or clone)
  THashList _serverList ; //! do not persist (or clone)

  void addServer(RooAbsArg& server) ;
  void removeServer(RooAbsArg& server) ;
	
  void printAttribList(ostream& os) ;

  friend class RooDataSet ;
  static Bool_t _verboseDirty ;

  ClassDef(RooAbsArg,1) // a real-valued variable and its value
};

#endif
