// @(#)root/tmva $Id: PDF.h,v 1.20 2006/11/17 14:59:24 stelzer Exp $
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : PDF                                                                   *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      PDF wrapper for histograms; uses user-defined spline interpolation        *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Xavier Prudent  <prudent@lapp.in2p3.fr>  - LAPP, France                   *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *      Kai Voss        <Kai.Voss@cern.ch>       - U. of Victoria, Canada         *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland,                                                        * 
 *      U. of Victoria, Canada,                                                   * 
 *      MPI-K Heidelberg, Germany ,                                               * 
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#ifndef ROOT_TMVA_PDF
#define ROOT_TMVA_PDF

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// PDF                                                                  //
//                                                                      //
// PDF wrapper for histograms; uses user-defined spline interpolation   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TSpline.h"
#include "TH1.h"
#include "TGraph.h"

#ifndef ROOT_TMVA_MsgLogger
#include "TMVA/MsgLogger.h"
#endif

namespace TMVA {

   class PDF : public TObject {
      
   public:

      enum ESmoothMethod { kSpline0, kSpline1, kSpline2, kSpline3, kSpline5 };
  
      PDF( const TH1* theHist, 
           ESmoothMethod method = kSpline2,
           Int_t nsmooth = 0 );
  
      virtual ~PDF( void );
  
      // returns probability density at given abscissa
      Double_t GetVal( const Double_t x );

      // histogram underlying the PDF
      TH1*     GetPDFHist ( void ) { return fPDFHist; }

      // integral of PDF within given range
      Double_t GetIntegral( Double_t xmin, Double_t xmax );

      // accessors
      TSpline* GetSpline( void ) const { return fSpline; }
      Double_t GetXmin  ( void ) const { return fXmin;   }
      Double_t GetXmax  ( void ) const { return fXmax;   }

      // do we use the original histogram as reference ?
      Bool_t   UseHistogram() { return fUseHistogram; }

   private:

      // sanity check of PDF quality (after smoothing): comparison with 
      // original histogram
      void     CheckHist(void);
      void     FillSplineToHist( void );
      Double_t GetIntegral( void );

      // flag that indicates that no splines are produced, and the original 
      // histogram is used as reference instead; this is useful for discrete
      // variables
      Bool_t   fUseHistogram;  // spline0 uses histogram as reference
  
      // to increase computation speed, the final PDF is filled in 
      // a high-binned histogram; "GetValue" then returns the histogram
      // entry, linearized between adjacent bins
      Int_t    fNbinsPDFHist;  // number of bins in high-binned reference histogram

      Int_t    fNsmooth;       // number of times the histogram is smoothed
      Double_t fXmin, fXmax;   // minimum and maximum of histogram

      TSpline* fSpline;        // the used spline type
      TH1*     fPDFHist;       // the high-binned histogram corresponding to the PDF
      TH1*     fHist;          // copy of input histogram
      TGraph*  fGraph;         // needed to create PDF from histogram

      mutable MsgLogger fLogger;  // message logger

      ClassDef(PDF,0)  // PDF wrapper for histograms
         ;
   };

} // namespace TMVA

#endif 
