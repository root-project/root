/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooSimultaneous.rdl,v 1.23 2002/05/16 00:01:36 verkerke Exp $
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
  RooSimultaneous(const char *name, const char *title, const RooArgList& pdfList, RooAbsCategoryLValue& indexCat) ;
  RooSimultaneous(const RooSimultaneous& other, const char* name=0);
  virtual TObject* clone(const char* newname) const { return new RooSimultaneous(*this,newname) ; }
  virtual ~RooSimultaneous() ;

  virtual Double_t evaluate() const ;
  virtual Bool_t selfNormalized() const { return kTRUE ; }
  Bool_t addPdf(const RooAbsPdf& pdf, const char* catLabel) ;

  virtual ExtendMode extendMode() const { 
    if (_anyMustExtend) return MustBeExtended ;
    if (_anyCanExtend) return CanBeExtended ;
    return CanNotBeExtended ; 
  }

  virtual Double_t expectedEvents() const ;

  virtual Bool_t forceAnalyticalInt(const RooAbsArg& dep) const { return kTRUE ; }
  Int_t getAnalyticalIntegralWN(RooArgSet& allVars, RooArgSet& numVars, const RooArgSet* normSet) const ;
  Double_t analyticalIntegralWN(Int_t code, const RooArgSet* normSet) const ;

  virtual RooPlot *plotOn(RooPlot *frame, Option_t* drawOptions="L", Double_t scaleFactor= 1.0, 
			  ScaleType stype=Relative, const RooAbsData* projData=0, const RooArgSet* projSet=0) const ; 
  
  virtual RooFitContext* fitContext(const RooAbsData& dset, const RooArgSet* projDeps=0) const ;

  RooAbsPdf* getPdf(const char* catName) const ;
  const RooAbsCategory& indexCat() const { return _indexCat.arg() ; }
  
protected:

  virtual void selectNormalization(const RooArgSet* depSet=0, Bool_t force=kFALSE) ;
  mutable RooSetProxy _plotCoefNormSet ;

  friend class RooSimGenContext ;
  virtual RooAbsGenContext* genContext(const RooArgSet &vars, 
				       const RooDataSet *prototype=0, Bool_t verbose= kFALSE) const ;

  mutable RooAICRegistry _codeReg ;  // Auxiliary class keeping tracking of composite analytical integration codes
 
  friend class RooSimFitContext ;
  RooCategoryProxy _indexCat ; // Index category
  TList    _pdfProxyList ;     // List of PDF proxies (named after applicable category state)
  Double_t _numPdf ;           // Number of registered PDFs
  Bool_t   _anyCanExtend ;     // Flag set if all component PDFs are extendable
  Bool_t   _anyMustExtend ;    // Flag set if all component PDFs are extendable

  ClassDef(RooSimultaneous,1)  // Description goes here
};

#endif
