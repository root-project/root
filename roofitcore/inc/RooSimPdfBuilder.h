/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id$                                                             *
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2002, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/
#ifndef ROO_SIM_PDF_BUILDER
#define ROO_SIM_PDF_BUILDER

#include "Rtypes.h"
#include "TObject.h"
#include "RooFitCore/RooArgSet.hh"
#include "RooFitCore/RooArgList.hh"
#include "RooFitCore/RooAbsData.hh"
class RooAbsPdf ;
class RooCategory ;

class RooSimPdfBuilder : public TObject {
public:

  RooSimPdfBuilder(const RooArgSet& pdfProtoList) ;
  ~RooSimPdfBuilder() ;

  RooArgSet* createProtoBuildConfig() ;

  const RooSimultaneous* buildPdf(const RooArgSet& buildConfig, const RooArgSet& dependents, 
				  const RooArgSet* auxSplitCats=0, Bool_t verbose=kFALSE) ;

  const RooSimultaneous* buildPdf(const RooArgSet& buildConfig, const RooAbsData* dataSet, 
				  const RooArgSet& auxSplitCats, Bool_t verbose=kFALSE) {
    return buildPdf(buildConfig,*dataSet->get(),&auxSplitCats,verbose) ;
  }

  const RooSimultaneous* buildPdf(const RooArgSet& buildConfig, const RooArgSet& dependents,
				  const RooArgSet& auxSplitCats, Bool_t verbose=kFALSE) {
    return buildPdf(buildConfig,dependents,&auxSplitCats,verbose) ;
  }

  const RooSimultaneous* buildPdf(const RooArgSet& buildConfig, const RooAbsData* dataSet, 
				  const RooArgSet* auxSplitCats=0, Bool_t verbose=kFALSE) {
    return buildPdf(buildConfig,*dataSet->get(),auxSplitCats,verbose) ;
  }
  
  const RooArgSet& splitLeafList() { return _splitNodeList; }

  void addSpecializations(const RooArgSet& specSet) ;
 
protected:

  RooArgSet _protoPdfSet ;       // Set of prototype PDFS

  RooArgSet _compSplitCatSet ;   // List of owned composite splitting categories
  RooArgSet _splitNodeList ;     // List of owned split nodes
  TList     _retiredCustomizerList ; // Retired customizer from previous builds (own their PDF branch nodes)

private:
  RooSimPdfBuilder(const RooSimPdfBuilder&) ; // No copying allowed

protected:
  ClassDef(RooSimPdfBuilder,0) // RooSimultaneous PDF Builder 
};

#endif
