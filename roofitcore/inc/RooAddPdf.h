/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitTools
 *    File: $Id: RooAddPdf.rdl,v 1.26 2002/03/22 22:43:54 verkerke Exp $
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
#include "RooFitCore/RooListProxy.hh"
#include "RooFitCore/RooAICRegistry.hh"

class RooAddPdf : public RooAbsPdf {
public:

  RooAddPdf(const char *name, const char *title);
  RooAddPdf(const char *name, const char *title,
	    RooAbsPdf& pdf1, RooAbsPdf& pdf2, RooAbsReal& coef1) ;
  RooAddPdf(const char *name, const char *title, const RooArgList& pdfList) ;
  RooAddPdf(const char *name, const char *title, const RooArgList& pdfList, const RooArgList& coefList) ;
  RooAddPdf(const RooAddPdf& other, const char* name=0) ;
  virtual TObject* clone(const char* newname) const { return new RooAddPdf(*this,newname) ; }
  virtual ~RooAddPdf() ;

  Double_t evaluate() const ;
  virtual Bool_t checkDependents(const RooArgSet* nset) const ;	

  virtual Bool_t forceAnalyticalInt(const RooAbsArg& dep) const { return kTRUE ; }
  Int_t getAnalyticalIntegralWN(RooArgSet& allVars, RooArgSet& numVars, const RooArgSet* normSet) const ;
  Double_t analyticalIntegralWN(Int_t code, const RooArgSet* normSet) const ;
  virtual Bool_t selfNormalized() const { return kTRUE ; }

  virtual ExtendMode extendMode() const { return (_haveLastCoef || _allExtendable) ? MustBeExtended : CanNotBeExtended; }
  virtual Double_t expectedEvents() const ;

  const RooArgList& pdfList() const { return _pdfList ; }
  const RooArgList& coefList() const { return _coefList ; }

  void fixCoefNormalization(const RooArgSet& refCoefNorm) ;

protected:

  virtual void selectNormalization(const RooArgSet* depSet=0) ;

  mutable RooSetProxy _refCoefNorm ;
  Bool_t _projectCoefs ;
  mutable Bool_t _doProjectCoefs ;

  void syncCoefProjList(const RooArgSet* nset, const RooArgSet* iset=0) const ;
  mutable RooArgSet* _lastCoefProjSet ;
  mutable RooArgSet* _lastCoefProjIntSet ;
  mutable RooListProxy _pdfProjList ;

  void updateCoefCache(const RooArgSet* nset) const ;
  mutable Double_t* _coefCache ;
  
  friend class RooAddGenContext ;
  virtual RooAbsGenContext* genContext(const RooArgSet &vars, 
				       const RooDataSet *prototype=0, Bool_t verbose= kFALSE) const ;

  void syncSuppNormList(const RooArgSet* nset) const ;
  mutable RooArgSet* _lastSupNormSet ;

  mutable RooAICRegistry _codeReg ;  // Registry of component analytical integration codes

  RooListProxy _pdfList ;   //  List of component PDFs
  RooListProxy _coefList ;  //  List of coefficients
  mutable RooArgList _snormList ;  //  List of supplemental normalization factors
  TIterator* _pdfIter ;     //! Iterator over PDF list
  TIterator* _coefIter ;    //! Iterator over coefficient list
  TIterator* _snormIter ;   //! Iterator over supplemental normalizations
  
  Bool_t _haveLastCoef ;   //  Flag indicating if last PDFs coefficient was supplied in the ctor
  Bool_t _allExtendable ;   //  Flag indicating if all PDF components are extendable

private:

  ClassDef(RooAddPdf,0) // PDF representing a sum of PDFs
};

#endif
