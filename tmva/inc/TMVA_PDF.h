// @(#)root/tmva $Id: TMVA_PDF.h,v 1.4 2006/04/29 23:55:41 andreas.hoecker Exp $
// Author: Andreas Hoecker, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA_PDF                                                              *
 *                                                                                *
 * Description:                                                                   *
 *      PDF wrapper for histograms; uses user-defined spline interpolation        *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Xavier Prudent  <prudent@lapp.in2p3.fr>  - LAPP, France                   *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-KP Heidelberg, Germany     *
 *      Kai Voss        <Kai.Voss@cern.ch>       - U. of Victoria, Canada         *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland,                                                        * 
 *      U. of Victoria, Canada,                                                   * 
 *      MPI-KP Heidelberg, Germany,                                               * 
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://mva.sourceforge.net/license.txt)                                       *
 *                                                                                *
 * File and Version Information:                                                  *
 * $Id: TMVA_PDF.h,v 1.4 2006/04/29 23:55:41 andreas.hoecker Exp $
 **********************************************************************************/

#ifndef ROOT_TMVA_PDF
#define ROOT_TMVA_PDF

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMVA_PDF                                                             //
//                                                                      //
// PDF wrapper for histograms; uses user-defined spline interpolation   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TSpline.h"
#include "TH1.h"
#include "TGraph.h"

class TMVA_PDF : public TObject {
  
 public:

  enum SmoothMethod { Spline1, Spline2, Spline3, Spline5 };
  
  TMVA_PDF( const TH1* theHist, 
	   TMVA_PDF::SmoothMethod method = Spline2,
	   Int_t nsmooth = 0 );
  
  ~TMVA_PDF( void );
  
  Double_t GetVal( const Double_t x );

  TH1*     GetPDFHist ( void ){ return fPDFHist; }
  Double_t GetIntegral( Double_t xmin, Double_t xmax );

  // accessors
  TSpline* GetSpline( void ) const { return fSpline; }
  Double_t GetXmin  ( void ) const { return fXmin;   }
  Double_t GetXmax  ( void ) const { return fXmax;   }

 private:

  void     checkHist(void);
  void     fillSplineToHist( void );
  Double_t Integral  ( Double_t xmin, Double_t xmax );
  Double_t Integral  ( void );
  
  Int_t      fNbinsPDFHist;

  Int_t      fNsmooth; // # times the histogram is smoothed
  Double_t   fXmin, fXmax;
  Int_t      fNbins;

  TSpline*   fSpline;
  TH1*       fPDFHist;
  TH1*       fHist; // copy of input histogram
  TGraph*    fGraph;
  Double_t   fIntegral;

  ClassDef(TMVA_PDF,0)  //PDF wrapper for histograms
};

#endif 
