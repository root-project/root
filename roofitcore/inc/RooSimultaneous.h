/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id$
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   25-Jun-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/
#ifndef ROO_SIMULTANEOUS
#define ROO_SIMULTANEOUS

#include "THashList.h"
#include "RooFitCore/RooAbsPdf.hh"
#include "RooFitCore/RooCategoryProxy.hh"
#include "RooFitCore/RooSetProxy.hh"
class RooAbsCategoryLValue ;

class RooSimultaneous : public RooAbsPdf {
public:

  // Constructors, assignment etc
  inline RooSimultaneous() { }
  RooSimultaneous(const char *name, const char *title, RooAbsCategoryLValue& indexCat) ;
  RooSimultaneous(const RooSimultaneous& other, const char* name=0);
  virtual TObject* clone() const { return new RooSimultaneous(*this) ; }
  virtual ~RooSimultaneous() ;

  virtual Double_t evaluate(const RooDataSet* dset=0) const ;
  virtual Bool_t selfNormalized(const RooArgSet& dependents) const { return kTRUE ; }
  Bool_t addPdf(const RooAbsPdf& pdf, const char* catLabel) ;
  
protected:

  RooCategoryProxy _indexCat ; // Index category
  THashList    _pdfProxyList ; // List of PDF proxies (named after applicable category state)

  ClassDef(RooSimultaneous,1)  // Description goes here
};

#endif
