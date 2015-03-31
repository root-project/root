/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooFormula.h,v 1.34 2007/05/11 09:11:30 verkerke Exp $
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
#ifndef ROO_FORMULA
#define ROO_FORMULA

#include "Rtypes.h"
#include "v5/TFormula.h"
#include "RooAbsReal.h"
#include "RooArgSet.h"
#include "RooPrintable.h"
#include "RooLinkedList.h"
#include <vector>

class RooFormula : public ROOT::v5::TFormula, public RooPrintable {
public:
  // Constructors etc.
  RooFormula() ;
  RooFormula(const char* name, const char* formula, const RooArgList& varList);
  RooFormula(const RooFormula& other, const char* name=0) ;
  virtual TObject* Clone(const char* newName=0) const { return new RooFormula(*this,newName) ; }
  virtual ~RooFormula();
	
  // Dependent management
  RooArgSet& actualDependents() const ;
  Bool_t changeDependents(const RooAbsCollection& newDeps, Bool_t mustReplaceAll, Bool_t nameChange) ;

  inline RooAbsArg* getParameter(const char* name) const { 
    // Return pointer to parameter with given name
    return (RooAbsArg*) _useList.FindObject(name) ; 
  }
  inline RooAbsArg* getParameter(Int_t index) const { 
    // Return pointer to parameter at given index
    return (RooAbsArg*) _origList.At(index) ; 
  }

  // Function value accessor
  inline Bool_t ok() { return _isOK ; }
  Double_t eval(const RooArgSet* nset=0) ;

  // Debugging
  void dump() ;
  Bool_t reCompile(const char* newFormula) ;


  virtual void printValue(std::ostream& os) const ;
  virtual void printName(std::ostream& os) const ;
  virtual void printTitle(std::ostream& os) const ;
  virtual void printClassName(std::ostream& os) const ;
  virtual void printArgs(std::ostream& os) const ;
  void printMultiline(std::ostream& os, Int_t contents, Bool_t verbose=kFALSE, TString indent="") const ;

  inline virtual void Print(Option_t *options= 0) const {
    // Printing interface (human readable)
    printStream(defaultPrintStream(),defaultPrintContents(options),defaultPrintStyle(options));
  }

protected:
  
  RooFormula& operator=(const RooFormula& other) ;
  void initCopy(const RooFormula& other) ;

  // Interface to ROOT::v5::TFormula engine
  Int_t DefinedVariable(TString &name, int& action) ; // ROOT 4
  Int_t DefinedVariable(TString &name) ; // ROOT 3
  Double_t DefinedValue(Int_t code) ;

  RooArgSet* _nset ;
  mutable Bool_t    _isOK ;     // Is internal state OK?
  RooLinkedList     _origList ; //! Original list of dependents
  std::vector<Bool_t> _useIsCat;//! Is given slot in _useList a category?
  RooLinkedList _useList ;      //! List of actual dependents 
  mutable RooArgSet _actual;    //! Set of actual dependents
  RooLinkedList _labelList ;    //  List of label names for category objects  
  mutable Bool_t    _compiled ; //  Flag set if formula is compiled

  ClassDef(RooFormula,1)     // ROOT::v5::TFormula derived class interfacing with RooAbsArg objects
};

#endif
