/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitTools
 *    File: $Id: RooAddPdf.rdl,v 1.5 2001/07/31 05:54:17 verkerke Exp $
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
#ifndef ROO_ADD_PDF
#define ROO_ADD_PDF

#include "RooFitCore/RooAbsPdf.hh"
#include "TList.h"

class RooAddPdf : public RooAbsPdf {
public:
  RooAddPdf(const char *name, const char *title);
  RooAddPdf(const char *name, const char *title,
	    RooAbsPdf& pdf1, RooAbsPdf& pdf2, RooAbsReal& coef1) ;
  RooAddPdf(const RooAddPdf& other, const char* name=0) ;
  virtual TObject* clone(const char* newname) const { return new RooAddPdf(*this,newname) ; }
  virtual ~RooAddPdf() ;

  void addPdf(RooAbsPdf& pdf, RooAbsReal& coef) ;
  void addLastPdf(RooAbsPdf& pdf) ;

  Double_t evaluate(const RooDataSet* dset) const ;
  virtual Bool_t checkDependents(const RooDataSet* set) const ;	

  Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& numVars) const ;
  Double_t analyticalIntegral(Int_t code) const ;
  virtual Bool_t selfNormalized() const { return kTRUE ; }

protected:

  TList _pdfProxyList ;
  TList _coefProxyList ;

private:

  ClassDef(RooAddPdf,0) // PDF representing a sum of PDFs
};

#endif
