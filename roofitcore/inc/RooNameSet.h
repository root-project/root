/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooNameSet.rdl,v 1.1 2001/08/01 01:27:55 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/
#ifndef ROO_NAME_SET
#define ROO_NAME_SET

#include "TList.h"
#include "TString.h"
#include "TClass.h"
#include "RooFitCore/RooAbsArg.hh"
#include "RooFitCore/RooPrintable.hh"
class RooArgSet ;

class RooNameSet : public TList, public RooPrintable {
public:

  // Constructors, assignment etc.
  RooNameSet();
  RooNameSet(const RooArgSet& argSet);
  RooNameSet(const RooNameSet& other) ;
  virtual TObject* Clone(const char* newname=0) { return new RooNameSet(*this) ; }
  virtual ~RooNameSet() ;

  void refill(const RooArgSet& argSet) ;
  RooArgSet* select(const RooArgSet& input) ;
  Bool_t operator==(const RooNameSet& other) ;

  virtual void printToStream(ostream &os, PrintOption opt= Standard, TString indent= "") const;
  inline virtual void Print(Option_t *options= 0) const {
    printToStream(defaultStream(),parseOptions(options));
  }

  TList _nameList ;

protected:

  TIterator* _nameIter ; //! do not persist

  ClassDef(RooNameSet,1) // A sterile version of RooArgSet, containing only the names of the contained RooAbsArgs
};

#endif
