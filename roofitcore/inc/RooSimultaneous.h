/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooSimultaneous.rdl,v 1.2 2001/07/31 05:54:22 verkerke Exp $
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   25-Jun-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/
#ifndef ROO_SIMULTANEOUS
#define ROO_SIMULTANEOUS

//#include "THashList.h"
#include "TList.h"
#include "RooFitCore/RooAbsPdf.hh"
#include "RooFitCore/RooCategoryProxy.hh"
#include "RooFitCore/RooSetProxy.hh"
#include "RooFitCore/RooSimFitContext.hh"
class RooAbsCategoryLValue ;

class RooSimultaneous : public RooAbsPdf {
public:

  // Constructors, assignment etc
  inline RooSimultaneous() { }
  RooSimultaneous(const char *name, const char *title, RooAbsCategoryLValue& indexCat) ;
  RooSimultaneous(const RooSimultaneous& other, const char* name=0);
  virtual TObject* clone(const char* newname) const { return new RooSimultaneous(*this,newname) ; }
  virtual ~RooSimultaneous() ;

  virtual Double_t evaluate(const RooDataSet* dset=0) const ;
  virtual Bool_t selfNormalized() const { return kTRUE ; }
  Bool_t addPdf(const RooAbsPdf& pdf, const char* catLabel) ;

  virtual Int_t fitTo(RooDataSet& data, Option_t *options, Double_t *minValue=0) ;
  
protected:

  friend class RooSimFitContext ;
  RooCategoryProxy _indexCat ; // Index category
  TList    _pdfProxyList ; // List of PDF proxies (named after applicable category state)
  Double_t _numPdf ;

  ClassDef(RooSimultaneous,1)  // Description goes here
};

#endif
