/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooCustomizer.rdl,v 1.1 2001/10/09 01:41:19 verkerke Exp $
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   20-Jul-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

#ifndef ROO_PDF_CUSTOMIZER
#define ROO_PDF_CUSTOMIZER

#include "Rtypes.h"
#include "TList.h"
#include "TNamed.h"
#include "TString.h"
#include "RooFitCore/RooArgSet.hh"
#include "RooFitCore/RooPrintable.hh"
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

  const RooArgSet& cloneBranchList() const { return _cloneBranchList ; }
  const RooArgSet& cloneLeafList() const { return *_cloneLeafList ; }

  RooArgSet* fullParamList(const RooArgSet* depList) const ;

  // Printing interface 
  virtual void printToStream(ostream& os, PrintOption opt= Standard, TString indent= "") const;
  inline virtual void Print(Option_t *options= 0) const {
    printToStream(defaultStream(),parseOptions(options));
  }

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
  RooArgSet  _masterUnsplitLeafList ;

  // Cloned nodes are owned by the customizer
  RooArgSet _cloneBranchList ;

  // Cloned leafs are owned by the user supplied list in the ctor
  RooArgSet* _cloneLeafList ;

  ClassDef(RooCustomizer,0) // PDF customizer 
} ;

#endif
