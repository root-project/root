/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitModels
 *    File: $Id: RooDstD0BG.rdl,v 1.3 2001/08/25 01:23:51 chcheng Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   UE, Ulrik Egede, RAL, U.Egede@rl.ac.uk
 *   MT, Max Turri, UC Santa Cruz
 *   CC, Chih-hsiang Cheng, Stanford University
 * History:
 *   07-Feb-2000 DK Created initial version from RooGaussianProb
 *   29-Feb-2000 UE Created as copy of RooArgusBG.rdl
 *   12-Jul-2000 MT Implement alpha parameter
 *   21-Aug-2001 CC Migrate from RooFitTool  
 *
 * Description : Background shape for D*-D0 mass difference
 *
 * Copyright (C) 2000 RAL
 *****************************************************************************/
#ifndef ROO_DstD0_BG
#define ROO_DstD0_BG

#include "RooFitCore/RooAbsPdf.hh"
#include "RooFitCore/RooRealProxy.hh"

class RooRealVar;

class RooDstD0BG : public RooAbsPdf {
public:
  RooDstD0BG(const char *name, const char *title,
	     RooAbsReal& _dm, RooAbsReal& _dm0, RooAbsReal& _c,
	     RooAbsReal& _a, RooAbsReal& _b);

  RooDstD0BG(const RooDstD0BG& other, const char *name=0) ;
  virtual TObject *clone(const char *newname) const { 
    return new RooDstD0BG(*this,newname); }
  inline virtual ~RooDstD0BG() { };
  
  Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const RooArgSet* normSet=0) const ;
  Double_t analyticalIntegral(Int_t code) const ;
  
protected:

  RooRealProxy dm ;
  RooRealProxy dm0 ;
  RooRealProxy c,a,b ;

  Double_t evaluate() const;
  
private:
  
  ClassDef(RooDstD0BG,0) // D*-D0 mass difference bg PDF
};

#endif
