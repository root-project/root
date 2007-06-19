// @(#)root/tmva $Id: PDF.h,v 1.13 2007/04/19 06:53:01 brun Exp $
// Author: Asen Christov, Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : PDF                                                                   *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      PDF wrapper for histograms; uses user-defined spline interpolation        *
 *      and kernel density estimation                                             *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Asen Christov   <christov@physik.uni-freiburg.de> - Freiburg U., Germany  *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Xavier Prudent  <prudent@lapp.in2p3.fr>  - LAPP, France                   *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *      Kai Voss        <Kai.Voss@cern.ch>       - U. of Victoria, Canada         *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland                                                         * 
 *      U. of Victoria, Canada                                                    * 
 *      MPI-K Heidelberg, Germany                                                 * 
 *      LAPP, Annecy, France,                                                     *
 *      Freiburg U., Germany                                                      * 
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

#include <iostream>

#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TMVA_MsgLogger
#include "TMVA/MsgLogger.h"
#endif
#ifndef ROOT_TMVA_KDEKernel
#include "TMVA/KDEKernel.h"
#endif

#ifndef ROOT_TH1
#include "TH1.h"
#endif

class TSpline;
class TGraph;
class TF1;

namespace TMVA {

   class PDF;
   ostream& operator<< ( ostream& os, const PDF& tree );
   istream& operator>> ( istream& istr, PDF& tree);

   class PDF : public TObject {

      friend ostream& operator<< ( ostream& os, const PDF& tree );
      friend istream& operator>> ( istream& istr, PDF& tree);
      
   public:

      enum EInterpolateMethod { kSpline0, kSpline1, kSpline2, kSpline3, kSpline5, kKDE };

      PDF();
      PDF( const TH1* theHist, EInterpolateMethod method = kSpline2, Int_t nsmooth = 0,
           Bool_t checkHist = kFALSE );
      PDF( const TH1* theHist, 
           KDEKernel::EKernelType ktype, KDEKernel::EKernelIter kiter, KDEKernel::EKernelBorder 
           kborder, Float_t FineFactor );

      virtual ~PDF();
  
      // returns probability density at given abscissa
      Double_t GetVal( Double_t x ) const;

      // histogram underlying the PDF
      TH1*     GetPDFHist()      const { return fPDFHist; }
      TH1*     GetOriginalHist() const { return fHistOriginal; }
      TH1*     GetSmoothedHist() const { return fHist; }

      // integral of PDF within given range
      Double_t GetIntegral( Double_t xmin, Double_t xmax );

      // accessors
      TSpline* GetSpline() const { return fSpline; }
      Int_t    GetNBins () const { return fHist->GetNbinsX(); }
      Double_t GetXmin  () const { return fHist->GetXaxis()->GetXmin();   }
      Double_t GetXmax  () const { return fHist->GetXaxis()->GetXmax();   }

      // perform series of validation tests
      void     ValidatePDF( TH1* original = 0 ) const;

      // modified name (remove TMVA::)
      const char* GetName() const { return "PDF"; }

      // TMVA version control (for weight files)
      void   SetReadingVersion( UInt_t rv ) { fReadingVersion = rv; }      
      UInt_t GetReadingVersion() const { return fReadingVersion; }

   private:

      // sanity check of PDF quality (after smoothing): comparison with 
      // original histogram
      void     CheckHist() const;
      void     FillSplineToHist();
      void     FillKDEToHist();
      Double_t GetIntegral() const;
      Double_t GetPdfHistBinWidth() const { 
         TH1* h = GetPDFHist();
         return (fPDFHist) ? (h->GetXaxis()->GetXmax() - h->GetXaxis()->GetXmin())/h->GetNbinsX() : 1;
      }

      // do we use the original histogram as reference ?
      Bool_t   UseHistogram() const { return fUseHistogram; }

      void     BuildPDF( Bool_t checkHist = kFALSE );

      // flag that indicates that no splines are produced and no smoothing
      // is applied, i.e., the original histogram is used as reference
      // this is useful for discrete variables      
      Bool_t   fUseHistogram;  // spline0 uses histogram as reference
  
      // static configuration variables ----------------------------
      // to increase computation speed, the final PDF is filled in 
      // a high-binned histogram; "GetValue" then returns the histogram
      // entry, linearized between adjacent bins
      static const Int_t    fgNbin_PdfHist;           // number of bins in high-binned reference histogram
      static const Bool_t   fgManualIntegration;      // manual integration (sum over bins) or DGAUSS
      static const Double_t fgEpsilon;                // minimum PDF return
      // -----------------------------------------------------------

      Int_t    fNsmooth;                              // number of times the histogram is smoothed
      TMVA::PDF::EInterpolateMethod fInterpolMethod;  // interpolation method
      TSpline* fSpline;                               //! the used spline type
      TH1*     fPDFHist;                              //  the high-binned histogram corresponding to the PDF
      TH1*     fHist;                                 //  copy of input histogram
      TH1*     fHistOriginal;                         //  the input histogram
      TGraph*  fGraph;                                //! needed to create PDF from histogram
      TF1*     fIGetVal;                              // integration interface

      KDEKernel::EKernelType   fKDEtype;              // Kernel type to use for KDE
      KDEKernel::EKernelIter   fKDEiter;              // Number of iterations (adaptive or not)
      KDEKernel::EKernelBorder fKDEborder;            // The method to take care about "border" effects (string)
      Float_t                  fFineFactor;           // fine tuning factor for Adaptive KDE

      UInt_t                   fReadingVersion;       // the TMVA version of the weight file

      mutable MsgLogger fLogger;                      //! message logger

      // static pointer to this object
      static PDF* fgThisPDF;                          // this PDF pointer
      static PDF* ThisPDF( void ) { return fgThisPDF; }

      // external auxiliary functions 
      static Double_t IGetVal( Double_t*, Double_t* );

      ClassDef(PDF,1)  // PDF wrapper for histograms
   };

} // namespace TMVA

#endif 
