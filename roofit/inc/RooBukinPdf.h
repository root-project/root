/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
 *    File: $Id: RooBukinPdf.h,v 1.5 2007/07/12 20:30:49 wouter Exp $
 * Authors:                                                                  *
 *   RW, Ruddick William  UC Colorado        wor@slac.stanford.edu           *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/


// -- CLASS DESCRIPTION [PDF] --
// RooBukinPdf implements the NovosibirskA function 

// Original Fortran Header below
/*****************************************************************************
 * Fitting function for asymmetric peaks with 6 free parameters:	     *
 *     Ap   - peak value						     *
 *     Xp   - peak position						     *
 *     sigp - FWHM divided by 2*sqrt(2*log(2))=2.35			     *
 *     xi   - peak asymmetry parameter					     *
 *     rho1 - parameter of the "left tail"				     *
 *     rho2 - parameter of the "right tail"				     *
 *   ---------------------------------------------			     *
 *       May 26, 2003							     *
 *       A.Bukin, Budker INP, Novosibirsk				     *
 *       Documentation:							     *
 *       http://www.slac.stanford.edu/BFROOT/www/Organization/CollabMtgs/2003/detJuly2003/Tues3a/bukin.ps 
 *   -------------------------------------------			     *
 *****************************************************************************/
#ifndef ROO_BUKINPDF
#define ROO_BUKINPDF

#include "RooAbsPdf.h"
#include "RooRealProxy.h"

class RooRealVar;
class RooAbsReal;

class RooBukinPdf : public RooAbsPdf {
public:

  RooBukinPdf() {} ;
  RooBukinPdf(const char *name, const char *title,
	      RooAbsReal& _x, RooAbsReal& _Xp,
	      RooAbsReal& _sigp, RooAbsReal& _xi,
              RooAbsReal& _rho1, RooAbsReal& _rho2);

  RooBukinPdf(const RooBukinPdf& other,const char* name=0) ;	

  virtual TObject* clone(const char* newname) const { return new RooBukinPdf(*this,newname);	}
  inline virtual ~RooBukinPdf() { }

protected:
  RooRealProxy x;
  RooRealProxy Xp;
  RooRealProxy sigp;
  RooRealProxy xi;
  RooRealProxy rho1;
  RooRealProxy rho2;
  Double_t evaluate() const;

private:

  ClassDef(RooBukinPdf,1) // Variation of Novosibirsk PDF
  double consts;
};

#endif
