/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooCustomizer.h,v 1.11 2007/05/11 09:11:30 verkerke Exp $
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

#ifndef ROO_PDF_CUSTOMIZER
#define ROO_PDF_CUSTOMIZER

#include "Rtypes.h"
#include "TList.h"
#include "TNamed.h"
#include "TString.h"
#include "RooArgSet.h"
#include "RooPrintable.h"
class RooAbsCategoryLValue ; 
class RooAbsCategory ;
class RooAbsArg ;
class RooAbsPdf ;

class RooCustomizer : public TNamed, public RooPrintable {

public:

  // Constructors, assignment etc
  RooCustomizer(const RooAbsArg& pdf, const RooAbsCategoryLValue& masterCat, RooArgSet& splitLeafListOwned, RooArgSet* splitLeafListAll=0) ;
  RooCustomizer(const RooAbsArg& pdf, const char* name) ;
  virtual ~RooCustomizer() ;
  
  void setOwning(Bool_t flag) { _owning = flag ; }
  
  void splitArgs(const RooArgSet& argSet, const RooAbsCategory& splitCat) ;
  void splitArg(const RooAbsArg& arg, const RooAbsCategory& splitCat) ;
  void replaceArg(const RooAbsArg& orig, const RooAbsArg& subst) ;
  RooAbsArg* build(const char* masterCatState, Bool_t verbose=kFALSE) ;
  RooAbsArg* build(Bool_t verbose=kFALSE) ;

  const RooArgSet& cloneBranchList() const { return *_cloneBranchList ; }
  const RooArgSet& cloneLeafList() const { return *_cloneNodeListOwned ; }

  // Printing interface 
  virtual void printName(ostream& os) const ;
  virtual void printTitle(ostream& os) const ;
  virtual void printClassName(ostream& os) const ;
  virtual void printArgs(ostream& os) const ;
  virtual void printMultiline(ostream& os, Int_t content, Bool_t verbose=kFALSE, TString indent= "") const;

  inline virtual void Print(Option_t *options= 0) const {
    printStream(defaultPrintStream(),defaultPrintContents(options),defaultPrintStyle(options));
  }

  // Releases ownership of list of cloned branch nodes
  void setCloneBranchSet(RooArgSet& cloneBranchSet) ;

protected:
  
  RooCustomizer(const RooCustomizer&) ;
  void initialize() ;
  
  RooAbsArg* doBuild(const char* masterCatState, Bool_t verbose) ;

  Bool_t _sterile ;
  Bool_t _owning ;
  TString _name ;

  TList _splitArgList ;
  TList _splitCatList ;

  TList _replaceArgList ;
  TList _replaceSubList ;

  // Master nodes are not owned
  RooAbsArg* _masterPdf ;
  RooAbsCategoryLValue* _masterCat ;

  TIterator* _masterLeafListIter ;
  TIterator* _masterBranchListIter ;

  RooArgSet  _masterBranchList ;
  RooArgSet  _masterLeafList ;

  RooArgSet  _internalCloneBranchList ;
  RooArgSet* _cloneBranchList ;

  // Cloned leafs are owned by the user supplied list in the ctor
  RooArgSet* _cloneNodeListAll ;
  RooArgSet* _cloneNodeListOwned ;

  ClassDef(RooCustomizer,0) // PDF customizer 
} ;

#endif
