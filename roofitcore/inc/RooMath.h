/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooMath.rdl,v 1.2 2001/07/31 05:54:20 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   20-Jun-2000 DK Created initial version
 *   18-Jun-2001 WV Imported from RooFitTools
 *
 * Copyright (C) 2000 Stanford University
 *****************************************************************************/
#ifndef ROO_MATH
#define ROO_MATH

#include "RooFitCore/RooComplex.hh"

#include <math.h>

typedef RooComplex* pRooComplex ;
typedef Double_t* pDouble_t ;

class RooMath {
public:

  // CERNLIB complex error function
  static RooComplex ComplexErrFunc(Double_t re, Double_t im= 0);
  static RooComplex ComplexErrFunc(const RooComplex& z);

  // Interpolated CERF with automatic interpolation order selection
  static inline RooComplex FastComplexErrFunc(const RooComplex& z){
    return ITPComplexErrFunc(z,z.im()>0?3:4) ;
  }
  
  // Interpolated Re(CERF) with automatic interpolation order selection
  static inline Double_t FastComplexErrFuncRe(const RooComplex& z) {
    return ITPComplexErrFuncRe(z,z.im()>0?3:4) ;
  }

  // Interpolated Im(CERF) with automatic interpolation order selection
  static inline Double_t FastComplexErrFuncIm(const RooComplex& z) {
    return ITPComplexErrFuncIm(z,z.im()>0?3:4) ;
  }

  // Interpolated complex error function at specified interpolation order
  static RooComplex ITPComplexErrFunc(const RooComplex& z, Int_t nOrder) ;
  static Double_t ITPComplexErrFuncRe(const RooComplex& z, Int_t nOrder) ;
  static Double_t ITPComplexErrFuncIm(const RooComplex& z, Int_t nOrder) ;

private:

  // Allocate and initialize CERF lookup grid
  static void initFastCERF(Int_t reBins= 800, Double_t reMin=-4.0, Double_t reMax=4.0, 
			   Int_t imBins=1000, Double_t imMin=-4.0, Double_t imMax=6.0) ;

  // 1-D nth order polynomial interpolation routine
  static Double_t interpolate(Double_t yArr[],Int_t nOrder, Double_t x) ;

  // CERF lookup grid
  static pDouble_t* _imCerfArray ;
  static pDouble_t* _reCerfArray ;

  // CERF grid dimensions and parameters
  static Int_t _reBins ;
  static Double_t _reMin ;
  static Double_t _reMax ;
  static Double_t _reRange ;
  static Double_t _reStep ;

  static Int_t _imBins ;
  static Double_t _imMin ;
  static Double_t _imMax ;
  static Double_t _imRange ;
  static Double_t _imStep ;
  
  ClassDef(RooMath,0) // math utility routines
};

#endif
