/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooCustomizer.rdl,v 1.10 2005/06/20 15:44:50 wverkerke Exp $
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
  RooCustomizer(const RooAbsArg& pdf, const RooAbsCategoryLValue& masterCat, RooArgSet& splitLeafList) ;
  RooCustomizer(const RooAbsArg& pdf, const char* name) ;
  virtual ~RooCustomizer() ;
  
  void splitArgs(const RooArgSet& argSet, const RooAbsCategory& splitCat) ;
  void splitArg(const RooAbsArg& arg, const RooAbsCategory& splitCat) ;
  void replaceArg(const RooAbsArg& orig, const RooAbsArg& subst) ;
  RooAbsArg* build(const char* masterCatState, Bool_t verbose=kFALSE) ;
  RooAbsArg* build(Bool_t verbose=kFALSE) ;

  const RooArgSet& cloneBranchList() const { return *_cloneBranchList ; }
  const RooArgSet& cloneLeafList() const { return *_cloneNodeList ; }

  // Printing interface 
  virtual void printToStream(ostream& os, PrintOption opt= Standard, TString indent= "") const;
  inline virtual void Print(Option_t *options= 0) const {
    printToStream(defaultStream(),parseOptions(options));
  }

  // Releases ownership of list of cloned branch nodes
  void setCloneBranchSet(RooArgSet& cloneBranchSet) ;

protected:
  
  RooCustomizer(const RooCustomizer&) ;
  void initialize() ;
  
  RooAbsArg* doBuild(const char* masterCatState, Bool_t verbose) ;

  Bool_t _sterile ;
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
  RooArgSet* _cloneNodeList ;

  ClassDef(RooCustomizer,0) // PDF customizer 
} ;

#endif
