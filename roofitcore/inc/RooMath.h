/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooMath.h,v 1.16 2007/05/11 09:11:30 verkerke Exp $
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/
#ifndef ROO_MATH
#define ROO_MATH

#include "RooComplex.h"

#include <math.h>
#include <fstream>

typedef RooComplex* pRooComplex ;
typedef Double_t* pDouble_t ;

class RooMath {
public:

  virtual ~RooMath() {} ;

  // CERNLIB complex error function
  static RooComplex ComplexErrFunc(Double_t re, Double_t im= 0);
  static RooComplex ComplexErrFunc(const RooComplex& z);

  // Interpolated CERF with automatic interpolation order selection
  static RooComplex FastComplexErrFunc(const RooComplex& z) ;
  
  // Interpolated Re(CERF) with automatic interpolation order selection
  static Double_t FastComplexErrFuncRe(const RooComplex& z) ;

  // Interpolated Im(CERF) with automatic interpolation order selection
  static Double_t FastComplexErrFuncIm(const RooComplex& z) ;

  // Interpolated complex error function at specified interpolation order
  static RooComplex ITPComplexErrFunc(const RooComplex& z, Int_t nOrder) ;
  static Double_t ITPComplexErrFuncRe(const RooComplex& z, Int_t nOrder) ;
  static Double_t ITPComplexErrFuncIm(const RooComplex& z, Int_t nOrder) ;

  // Switch to use file cache for CERF lookup table
  static void cacheCERF(Bool_t flag=kTRUE) ;

  // 1-D nth order polynomial interpolation routines
  static Double_t interpolate(Double_t yArr[],Int_t nOrder, Double_t x) ;
  static Double_t interpolate(Double_t xa[], Double_t ya[], Int_t n, Double_t x) ;

  static Double_t erf(Double_t x) ;
  static Double_t erfc(Double_t x) ;
  
  static void cleanup() ;

private:

  static Bool_t loadCache() ;
  static void storeCache() ;
  static const char* cacheFileName() ;

  // Allocate and initialize CERF lookup grid
  static void initFastCERF(Int_t reBins= 800, Double_t reMin=-4.0, Double_t reMax=4.0, 
			   Int_t imBins=1000, Double_t imMin=-4.0, Double_t imMax=6.0) ;

  // CERF lookup grid
  static pDouble_t* _imCerfArray ; // Lookup table for Im part of complex error function
  static pDouble_t* _reCerfArray ; // Lookup table for Re part of complex error function

  // CERF grid dimensions and parameters
  static Int_t _reBins ;      // Number of grid points in real dimension of CERF-LUT
  static Double_t _reMin ;    // Low edge of real dimension of CERF-LUT
  static Double_t _reMax ;    // High edge of real dimension of CERF-LUT
  static Double_t _reRange ;  // Range in real dimension of CERF-LUT
  static Double_t _reStep ;   // Grid spacing in real dimension of CERF-LUT

  static Int_t _imBins ;      // Number of grid points in imaginary dimension of CERF-LUT    
  static Double_t _imMin ;    // Low edge of imaginary dimension of CERF-LUT
  static Double_t _imMax ;    // High edge of imaginary dimension of CERF-LUT
  static Double_t _imRange ;  // Range in imaginary dimension of CERF-LUT
  static Double_t _imStep ;   // Grid spacing in imaginary dimension of CERF-LUT

  static Bool_t _cacheTable ; // Switch activating use of file cache for CERF-LUT
  
  ClassDef(RooMath,0) // math utility routines
};

#endif
