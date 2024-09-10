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

#include <RooArgList.h>
#include <RooArgSet.h>
#include <RooPrintable.h>

#include <TNamed.h>
#include <TString.h>

#include <vector>
#include <string>

class RooAbsCategoryLValue ;
class RooAbsCategory ;
class RooAbsArg ;
class RooAbsPdf ;

// Editing tool for RooAbsArg composite object expressions
class RooCustomizer {

public:

  // Constructors, assignment etc
  RooCustomizer(const RooAbsArg& pdf, const RooAbsCategoryLValue& masterCat, RooArgSet& splitLeafListOwned, RooArgSet* splitLeafListAll=nullptr) ;
  RooCustomizer(const RooAbsArg& pdf, const char* name) ;

  /// If flag is true, make customizer own all created components
  void setOwning(bool flag) {
    _owning = flag ;
  }

  void splitArgs(const RooArgSet& argSet, const RooAbsCategory& splitCat) ;
  void splitArg(const RooAbsArg& arg, const RooAbsCategory& splitCat) ;
  void replaceArg(const RooAbsArg& orig, const RooAbsArg& subst) ;
  RooAbsArg* build(const char* masterCatState, bool verbose=false) ;
  RooAbsArg* build(bool verbose=false) ;

  /// Return list of cloned branch nodes
  const RooArgSet& cloneBranchList() const {
    return *_cloneBranchList ;
  }
  /// Return list of cloned leaf nodes
  const RooArgSet& cloneLeafList() const {
    return *_cloneNodeListOwned ;
  }

  // Printing interface
  void printArgs(std::ostream& os) const ;
  void printMultiline(std::ostream& os, Int_t content, bool verbose=false, TString indent= "") const;

  /// Releases ownership of list of cloned branch nodes
  void setCloneBranchSet(RooArgSet& cloneBranchSet) ;

  RooAbsPdf const& pdf() const;

  RooCustomizer(const RooCustomizer &) = delete;
  RooCustomizer &operator=(const RooCustomizer &) = delete;
  RooCustomizer(RooCustomizer &&) = delete;
  RooCustomizer &operator=(RooCustomizer &&) = delete;

protected:

  void initialize() ;

  RooAbsArg* doBuild(const char* masterCatState, bool verbose) ;

  bool _sterile ; ///< If true we do not have as associated master category
  bool _owning ;  ///< If true we own all created components
  TString _name ;   ///< Name of this object

  RooArgList _splitArgList ; ///< List of RooAbsArgs to be split
  RooArgList _splitCatList ; ///< List of categories to be used for above splits

  RooArgList _replaceArgList ; ///< List of RooAbsArgs to be replaced
  RooArgList _replaceSubList ; ///< List of replacement RooAbsArgs

  // Master nodes are not owned
  RooAbsArg* _masterPdf ;             ///< Pointer to input p.d.f
  RooAbsCategoryLValue* _masterCat = nullptr;  ///< Pointer to input master category

  RooArgSet  _masterBranchList ;      ///< List of branch nodes
  RooArgSet  _masterLeafList ;        ///< List of leaf nodes

  RooArgSet  _internalCloneBranchList;   ///< List of branches of internal clone
  RooArgSet* _cloneBranchList = nullptr; ///< Pointer to list of cloned branches used

  // Cloned leaves are owned by the user supplied list in the constructor
  RooArgSet* _cloneNodeListAll = nullptr;  ///< List of all cloned nodes
  RooArgSet* _cloneNodeListOwned = nullptr;///< List of owned cloned nodes
} ;

#endif
