/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooWorkspace.h,v 1.1 2007/07/12 20:30:28 wouter Exp $
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
#ifndef ROO_WORKSPACE
#define ROO_WORKSPACE

#include "RooPrintable.h"
#include "RooArgSet.h"
#include "RooLinkedList.h"

class RooAbsPdf ;
class RooAbsData ;
class RooRealVar ;
class RooCategory ;
class RooAbsReal ;
//class RooModelView ;

#include "TNamed.h"

class RooWorkspace : public TNamed {
public:

  RooWorkspace() {} ;
  RooWorkspace(const char* name, const char* title=0) ;
  RooWorkspace(const RooWorkspace& other) ;
  ~RooWorkspace() ;

  // Import functions for dataset, functions
  Bool_t import(const RooAbsArg& arg) ;
  Bool_t import(RooAbsData& data) ;

  // Import other workspaces
  Bool_t merge(const RooWorkspace& other) ;
  Bool_t join(const RooWorkspace& other) ;

  // Accessor functions 
  RooAbsPdf* pdf(const char* name) ;
  RooAbsReal* function(const char* name) ;
  RooRealVar* var(const char* name) ;
  RooCategory* cat(const char* name) ;
  RooAbsData* data(const char* name) ;
  TIterator* componentIterator() { return _allOwnedNodes.createIterator() ; }
  const RooArgSet& components() const { return _allOwnedNodes ; }

  // View management
//RooModelView* addView(const char* name, const RooArgSet& observables) ;
//RooModelView* view(const char* name) ;
//void removeView(const char* name) ;

  // Print function
  void Print(Option_t* opts=0) const ;

 private:

  RooArgSet _allOwnedNodes ; // List of owned pdfs and components
  RooLinkedList _dataList ; // List of owned datasets
  RooLinkedList _views ; // List of model views

  ClassDef(RooWorkspace,1)  // The RooFit Project Workspace 
  
} ;

#endif
