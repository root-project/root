/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitTools
 *    File: $Id: RooArgusBG.rdl,v 1.5 2001/08/23 01:23:34 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Feb-2000 DK Created initial version from RooGaussianProb
 *   02-May-2001 WV Port to RooFitModels/RooFitCore
 *
 * Copyright (C) 2000 Stanford University
 *****************************************************************************/
#ifndef ROO_ARGUS_BG
#define ROO_ARGUS_BG

#include "RooFitCore/RooAbsPdf.hh"
#include "RooFitCore/RooRealProxy.hh"

class RooRealVar;
class RooAbsReal;

class RooArgusBG : public RooAbsPdf {
public:
  RooArgusBG(const char *name, const char *title, 
	     RooAbsReal& _m, RooAbsReal& _m0, RooAbsReal& _c);
  RooArgusBG(const char *name, const char *title, 
	     RooAbsReal& _m, RooAbsReal& _m0, RooAbsReal& _c, RooAbsReal& _p);
  RooArgusBG(const RooArgusBG& other,const char* name=0) ;
  virtual TObject* clone(const char* newname) const { return new RooArgusBG(*this,newname); }
  inline virtual ~RooArgusBG() { }

protected:
  RooRealProxy m ;
  RooRealProxy m0 ;
  RooRealProxy c ;
  RooRealProxy p ;

  Double_t evaluate() const ;
//   void initGenerator();

private:
  ClassDef(RooArgusBG,0) // Argus background shape PDF
};

#endif
