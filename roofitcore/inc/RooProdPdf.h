/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitTools
 *    File: $Id: RooProdPdf.rdl,v 1.11 2001/09/18 04:13:48 verkerke Exp $
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
#include "RooFitCore/RooListProxy.hh"
#include "RooFitCore/RooAICRegistry.hh"

class RooProdPdf : public RooAbsPdf {
public:
  RooProdPdf(const char *name, const char *title, Double_t cutOff=0);
  RooProdPdf(const char *name, const char *title,
	    RooAbsPdf& pdf1, RooAbsPdf& pdf2, Double_t cutOff=0) ;
  RooProdPdf(const char* name, const char* title, RooArgList& pdfList, Double_t cutOff=0) ;
  RooProdPdf(const RooProdPdf& other, const char* name=0) ;
  virtual TObject* clone(const char* newname) const { return new RooProdPdf(*this,newname) ; }
  virtual ~RooProdPdf() ;

  void addPdf(RooAbsPdf& pdf) ;

  Double_t evaluate() const ;

  virtual Bool_t forceAnalyticalInt(const RooAbsArg& dep) const { return kTRUE ; }
  Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& numVars, const RooArgSet* normSet) const ;
  Double_t analyticalIntegral(Int_t code, const RooArgSet* normSet) const ;
  virtual Bool_t selfNormalized() const { return kTRUE ; }

protected:

  mutable RooAICRegistry _codeReg ;

  Double_t _cutOff ;
  RooListProxy _pdfList ;
  TIterator* _pdfIter ; //! do not persist

private:

  ClassDef(RooProdPdf,0) // PDF representing a product of PDFs
};

#endif
