/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooMath.rdl,v 1.5 2001/10/08 05:20:18 verkerke Exp $
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
#include <fstream.h>

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

  // Switch to use file cache for CERF lookup table
  static void cacheCERF(Bool_t flag=kTRUE) { _cacheTable = flag ; }

  // 1-D nth order polynomial interpolation routine
  static Double_t interpolate(Double_t yArr[],Int_t nOrder, Double_t x) ;

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
