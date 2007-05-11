/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooCmdArg.rdl,v 1.9 2005/04/18 21:44:42 wverkerke Exp $
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

#ifndef ROO_CMD_ARG
#define ROO_CMD_ARG

#include "TNamed.h"
#include "TString.h"
#include "RooLinkedList.h"
class RooAbsData ;


class RooCmdArg : public TNamed {
public:

  RooCmdArg();
  RooCmdArg(const char* name, 
	    Int_t i1=0, Int_t i2=0, 
	    Double_t d1=0, Double_t d2=0, 
	    const char* s1=0, const char* s2=0, 
	    const TObject* o1=0, const TObject* o2=0, const RooCmdArg* ca=0) ;
  RooCmdArg(const RooCmdArg& other) ;
  RooCmdArg& operator=(const RooCmdArg& other) ;
  void addArg(const RooCmdArg& arg) ;
  void setProcessRecArgs(Bool_t flag) { _procSubArgs = flag ; }
  const RooLinkedList& subArgs() const { return _argList ; }

  virtual TObject* Clone(const char* newName=0) const {
    RooCmdArg* newarg = new RooCmdArg(*this) ;
    if (newName) { newarg->SetName(newName) ; }
    return newarg ;
  }

  ~RooCmdArg();

  static const RooCmdArg none  ;

protected:

  friend class RooCmdConfig ;
  friend class RooProdPdf ;

  const char* opcode() const { return strlen(GetName()) ? GetName() : 0 ; }
  Int_t getInt(Int_t idx) const { return _i[idx] ; }
  Double_t getDouble(Int_t idx) const { return _d[idx] ; }
  const char* getString(Int_t idx) const { return _s[idx] ; }
  const TObject* getObject(Int_t idx) const { return _o[idx] ; }

private:

  friend class RooAbsCollection ;

  // Payload
  Double_t _d[2] ;
  Int_t _i[2] ;
  const char* _s[2] ;
  TObject* _o[2] ;
  Bool_t _procSubArgs ;
  RooLinkedList _argList ;
	
  ClassDef(RooCmdArg,0) // Universal method argument
};

#endif


