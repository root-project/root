/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitTools
 *    File: $Id: RooProdPdf.rdl,v 1.1 2001/05/10 00:16:08 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   06-Jan-2000 DK Created initial version
 *   19-Apr-2000 DK Add the printEventStats() method
 *   26-Jun-2000 DK Add support for extended likelihood fits
 *   02-Jul-2000 DK Add support for multiple terms (instead of only 2)
 *   05-Jul-2000 DK Add support for extended maximum likelihood and a
 *                  new method for this: setNPar()
 *   03-May02001 WV Port to RooFitCore/RooFitModels
 *
 * Copyright (C) 2000 Stanford University
 *****************************************************************************/
#ifndef ROO_PROD_PDF
#define ROO_PROD_PDF

#include "RooFitCore/RooAbsPdf.hh"
#include "TList.h"

class RooProdPdf : public RooAbsPdf {
public:
  RooProdPdf(const char *name, const char *title);
  RooProdPdf(const char *name, const char *title,
	    RooAbsPdf& pdf1, RooAbsPdf& pdf2) ;
  RooProdPdf(const RooProdPdf& other, const char* name=0) ;
  virtual TObject* clone() const { return new RooProdPdf(*this) ; }
  virtual ~RooProdPdf() ;

  void addPdf(RooAbsPdf& pdf) ;

  Double_t evaluate() const ;

  Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& numVars) const ;
  Double_t analyticalIntegral(Int_t code) const ;
  Bool_t selfNormalized(const RooArgSet& dependents) const { return kTRUE ; }

protected:

  TList _pdfProxyList ;

private:

  ClassDef(RooProdPdf,0) // a non-persistent sum of PDFs
};

#endif
