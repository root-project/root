/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooSimultaneous.rdl,v 1.9 2001/09/24 23:06:01 verkerke Exp $
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
#include "RooFitCore/RooAICRegistry.hh"
class RooAbsCategoryLValue ;
class RooFitResult ;
class RooPlot ;
class RooAbsData ;

class RooSimultaneous : public RooAbsPdf {
public:

  // Constructors, assignment etc
  inline RooSimultaneous() { }
  RooSimultaneous(const char *name, const char *title, RooAbsCategoryLValue& indexCat) ;
  RooSimultaneous(const RooSimultaneous& other, const char* name=0);
  virtual TObject* clone(const char* newname) const { return new RooSimultaneous(*this,newname) ; }
  virtual ~RooSimultaneous() ;

  virtual Double_t evaluate() const ;
  virtual Bool_t selfNormalized() const { return kTRUE ; }
  Bool_t addPdf(const RooAbsPdf& pdf, const char* catLabel) ;

  virtual Bool_t forceAnalyticalInt(const RooAbsArg& dep) const { return kTRUE ; }
  Int_t getAnalyticalIntegralWN(RooArgSet& allVars, RooArgSet& numVars, const RooArgSet* normSet) const ;
  Double_t analyticalIntegralWN(Int_t code, const RooArgSet* normSet) const ;

  virtual const RooFitResult* fitTo(RooAbsData& data, Option_t *fitOpt = "", Option_t *optOpt = "cpds" ) ;
  virtual RooPlot *plotOn(RooPlot *frame, RooAbsData* wdata, Option_t* drawOptions="L", Double_t scaleFactor= 1.0) const;
  
protected:

  mutable RooAICRegistry _codeReg ; 

  friend class RooSimFitContext ;
  RooCategoryProxy _indexCat ; // Index category
  TList    _pdfProxyList ; // List of PDF proxies (named after applicable category state)
  Double_t _numPdf ;

  ClassDef(RooSimultaneous,1)  // Description goes here
};

#endif
