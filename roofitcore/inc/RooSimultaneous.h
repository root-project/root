/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooSimultaneous.rdl,v 1.41 2005/07/12 11:29:38 wverkerke Exp $
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
#ifndef ROO_SIMULTANEOUS
#define ROO_SIMULTANEOUS

//#include "THashList.h"
#include "TList.h"
#include "RooAbsPdf.h"
#include "RooCategoryProxy.h"
#include "RooSetProxy.h"
#include "RooAICRegistry.h"
#include "RooNormListManager.h"
class RooAbsCategoryLValue ;
class RooFitResult ;
class RooPlot ;
class RooAbsData ;
class RooLinkedList ;

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

  virtual Double_t expectedEvents(const RooArgSet* nset) const ;
  virtual Double_t expectedEvents(const RooArgSet& nset) const { return expectedEvents(&nset) ; }

  virtual Bool_t forceAnalyticalInt(const RooAbsArg&) const { return kTRUE ; }
  Int_t getAnalyticalIntegralWN(RooArgSet& allVars, RooArgSet& numVars, const RooArgSet* normSet, const char* rangeName=0) const ;
  Double_t analyticalIntegralWN(Int_t code, const RooArgSet* normSet, const char* rangeName=0) const ;

  using RooAbsPdf::plotOn ;
  virtual RooPlot* plotOn(RooPlot* frame, 
			  const RooCmdArg& arg1            , const RooCmdArg& arg2=RooCmdArg(),
			  const RooCmdArg& arg3=RooCmdArg(), const RooCmdArg& arg4=RooCmdArg(),
			  const RooCmdArg& arg5=RooCmdArg(), const RooCmdArg& arg6=RooCmdArg(),
			  const RooCmdArg& arg7=RooCmdArg(), const RooCmdArg& arg8=RooCmdArg(),
			  const RooCmdArg& arg9=RooCmdArg(), const RooCmdArg& arg10=RooCmdArg()) const {
    return RooAbsReal::plotOn(frame,arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9,arg10) ;
  }

  // Backward compatibility function
  virtual RooPlot *plotOn(RooPlot *frame, Option_t* drawOptions, Double_t scaleFactor=1.0, 
			  ScaleType stype=Relative, const RooAbsData* projData=0, const RooArgSet* projSet=0,
			  Double_t precision=1e-3, Bool_t shiftToZero=kFALSE, const RooArgSet* projDataSet=0,
			  Double_t rangeLo=0, Double_t rangeHi=0, RooCurve::WingMode wmode=RooCurve::Extended) const;
  
  RooAbsPdf* getPdf(const char* catName) const ;
  const RooAbsCategory& indexCat() const { return _indexCat.arg() ; }
  
protected:

  virtual RooPlot* plotOn(RooPlot* frame, RooLinkedList& cmdList) const ;

  virtual void selectNormalization(const RooArgSet* depSet=0, Bool_t force=kFALSE) ;
  virtual void selectNormalizationRange(const char* rangeName=0, Bool_t force=kFALSE) ;
  mutable RooSetProxy _plotCoefNormSet ;
  const TNamed* _plotCoefNormRange ;

  mutable RooNormListManager _normListMgr ; // Component normalization list manager
  virtual Bool_t redirectServersHook(const RooAbsCollection& newServerList, Bool_t mustReplaceAll, Bool_t nameChange, Bool_t isRecursive) ;

  friend class RooSimGenContext ;
  virtual RooAbsGenContext* genContext(const RooArgSet &vars, const RooDataSet *prototype=0, 
	                               const RooArgSet* auxProto=0, Bool_t verbose= kFALSE) const ;
 
  RooCategoryProxy _indexCat ; // Index category
  TList    _pdfProxyList ;     // List of PDF proxies (named after applicable category state)
  Int_t    _numPdf ;           // Number of registered PDFs
  Bool_t   _anyCanExtend ;     // Flag set if all component PDFs are extendable
  Bool_t   _anyMustExtend ;    // Flag set if all component PDFs are extendable

  ClassDef(RooSimultaneous,1)  // Description goes here
};

#endif
