/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitTools
 *    File: $Id: RooHistPdf.rdl,v 1.2 2001/10/08 05:20:16 verkerke Exp $
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   26-Sep-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/
#ifndef ROO_HIST_PDF
#define ROO_HIST_PDF

#include "RooFitCore/RooAbsPdf.hh"
#include "RooFitCore/RooRealProxy.hh"
#include "RooFitCore/RooSetProxy.hh"
#include "RooFitCore/RooAICRegistry.hh"

class RooRealVar;
class RooAbsReal;
class RooDataHist ;

class RooHistPdf : public RooAbsPdf {
public:
  RooHistPdf(const char *name, const char *title, const RooArgSet& vars, const RooDataHist& dhist, Int_t intOrder=0);
  RooHistPdf(const RooHistPdf& other, const char* name=0);
  virtual TObject* clone(const char* newname) const { return new RooHistPdf(*this,newname); }
  inline virtual ~RooHistPdf() { }

  Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars) const ;
  Double_t analyticalIntegral(Int_t code) const ;

protected:

  Double_t evaluate() const;

  RooSetProxy  _depList ;   // List of dependents defining dimensions of histogram
  RooDataHist* _dataHist ;  // Unowned pointer to underlying histogram
  mutable RooAICRegistry _codeReg ; // Auxiliary class keeping tracking of analytical integration code
  Int_t        _intOrder ; // Interpolation order

  ClassDef(RooHistPdf,0) // Histogram based PDF
};

#endif
