/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitTools
 *    File: $Id: RooSimPdfBuilder.rdl,v 1.6 2002/03/07 06:22:24 verkerke Exp $
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   17-Oct-2001 WV Created initial version
 *
 * Copyright (C) 2000 University of California
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
