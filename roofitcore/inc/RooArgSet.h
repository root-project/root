/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooArgSet.rdl,v 1.11 2001/05/10 21:26:08 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/
#ifndef ROO_ARG_SET
#define ROO_ARG_SET

#include "THashList.h"
#include "TString.h"
#include "TClass.h"
#include "RooFitCore/RooAbsArg.hh"
#include "RooFitCore/RooPrintable.hh"

class RooArgSet : public THashList, public RooPrintable {
public:

  // Constructors, assignment etc.
  RooArgSet();
  RooArgSet(const char *name);
  RooArgSet(const RooAbsArg& var1, 
	    const char *name="");
  RooArgSet(const RooAbsArg& var1, const RooAbsArg& var2, 
	    const char *name="");
  RooArgSet(const RooAbsArg& var1, const RooAbsArg& var2,
	    const RooAbsArg& var3, 
	    const char *name="");
  RooArgSet(const RooAbsArg& var1, const RooAbsArg& var2,
	    const RooAbsArg& var3, const RooAbsArg& var4, 
	    const char *name="");
  RooArgSet(const RooAbsArg& var1, const RooAbsArg& var2,
	    const RooAbsArg& var3, const RooAbsArg& var4, 
	    const RooAbsArg& var5, 
	    const char *name="");
  RooArgSet(const RooAbsArg& var1, const RooAbsArg& var2,
	    const RooAbsArg& var3, const RooAbsArg& var4, 
	    const RooAbsArg& var5, const RooAbsArg& var6, 
	    const char *name="");
  RooArgSet(const RooAbsArg& var1, const RooAbsArg& var2,
            const RooAbsArg& var3, const RooAbsArg& var4, 
	    const RooAbsArg& var5, const RooAbsArg& var6, 
	    const RooAbsArg& var7, 
	    const char *name="");
  RooArgSet(const RooAbsArg& var1, const RooAbsArg& var2,
            const RooAbsArg& var3, const RooAbsArg& var4, 
	    const RooAbsArg& var5, const RooAbsArg& var6, 
	    const RooAbsArg& var7, const RooAbsArg& var8, 
	    const char *name="");

  virtual ~RooArgSet();
  // Create a copy of an existing list. New variables cannot be added
  // to a copied list. The variables in the copied list are independent
  // of the original variables.
  RooArgSet(const RooArgSet& other, const char *name="");
  virtual TObject* Clone(const char* newname=0) const { return new RooArgSet(*this,newname); }
  RooArgSet& operator=(const RooArgSet& other);

  // Copy list and contents (and optionally 'deep' servers)
  RooArgSet *snapshot(Bool_t deepCopy=kTRUE) const ;

  // List content management
  Bool_t add(const RooAbsArg& var, Bool_t silent=kFALSE) ;
  Bool_t add(const RooArgSet& list) ;
  Bool_t remove(const RooAbsArg& var) ;
  Bool_t replace(const RooAbsArg& var1, const RooAbsArg& var2) ;

  // Group operations on AbsArgs
  void setAttribAll(const Text_t* name, Bool_t value=kTRUE) ;

  // List search methods
  RooAbsArg *find(const char *name) const ;
  Bool_t contains(RooAbsArg& var) const { return kTRUE ; }
  RooArgSet* selectByAttrib(const char* name, Bool_t value) ;

  // I/O streaming interface (machine readable)
  virtual Bool_t readFromStream(istream& is, Bool_t compact, Bool_t verbose=kFALSE) ;
  virtual void writeToStream(ostream& os, Bool_t compact) ;
  
  // Printing interface (human readable)
  virtual void printToStream(ostream& os, PrintOption opt= Standard, TString indent= "") const;
  inline virtual void Print(Option_t *options= 0) const {
    printToStream(defaultStream(),parseOptions(options));
  }

  const char* GetName() const { return _name.Data() ; }

protected:

  Bool_t _isCopy; // Copied lists own contents
  TString _name;  // THashList doesn't inherit from TNamed

  // Support for snapshot method 
  void addServerClonesToList(const RooAbsArg& var) ;

  ClassDef(RooArgSet,1) // a list of real-valued variables
};

#endif
