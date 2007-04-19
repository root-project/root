// @(#)root/tmva $Id: PDF.cxx,v 1.44 2007/04/02 08:13:32 andreas.hoecker Exp $
// Author: Asen Christov, Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : PDF                                                                   *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation (see header for description)                               *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Asen Christov   <christov@physik.uni-freiburg.de> - Freiburg U., Germany  *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Xavier Prudent  <prudent@lapp.in2p3.fr>  - LAPP, France                   *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *      Kai Voss        <Kai.Voss@cern.ch>       - U. of Victoria, Canada         *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland,                                                        * 
 *      U. of Victoria, Canada,                                                   * 
 *      MPI-K Heidelberg, Germany,                                                * 
 *      LAPP, Annecy, France,                                                     *
 *      Freiburg U., Germany                                                      * 
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#include <iomanip>
#include "Riostream.h"

#include "TMath.h"
#include "TMVA/PDF.h"
#include "TMVA/TSpline1.h"
#include "TMVA/TSpline2.h"
#include "TH1F.h"

namespace TMVA {
   const Bool_t   DEBUG_PDF=kFALSE;
   const Double_t PDF_epsilon_=1.0e-07;
}

Int_t TMVA::PDF::NBIN_PdfHist=10000;

using namespace std;

ClassImp(TMVA::PDF)

//_______________________________________________________________________
TMVA::PDF::PDF()
   : fUseHistogram  ( kFALSE ),
     fNsmooth       ( 0 ),
     fInterpolMethod( PDF::kSpline0 ),
     fSpline        ( 0 ),
     fPDFHist       ( 0 ),
     fHist          ( 0 ),
     fHistOriginal  ( 0 ),
     fGraph         ( 0 ),
     fKDEtype       ( KDEKernel::kNone ),
     fKDEiter       ( KDEKernel::kNonadaptiveKDE ),
     fLogger        ( this )
{
   // default constructor needed for ROOT I/O
}

//_______________________________________________________________________
TMVA::PDF::PDF( const TH1 *hist, TMVA::PDF::EInterpolateMethod method, Int_t nsmooth )
   : fUseHistogram  ( kFALSE ),
     fNsmooth       ( nsmooth ),
     fInterpolMethod( method ),
     fSpline        ( 0 ),
     fPDFHist       ( 0 ),
     fHist          ( 0 ),
     fHistOriginal  ( 0 ),
     fGraph         ( 0 ),
     fKDEtype       ( KDEKernel::kNone ),
     fKDEiter       ( KDEKernel::kNonadaptiveKDE ),
     fKDEborder     ( KDEKernel::kNoTreatment ),
     fFineFactor    ( 0. ),
     fLogger        ( this )
{  
   // constructor of spline based PDF: 
   // - default Spline method is: Spline2 (quadratic)
   // - default smoothing is none

   // sanity check
   if (hist == NULL) fLogger << kFATAL << "called without valid histogram pointer!" << Endl;

   // histogram should be non empty
   if (hist->GetEntries() <= 0) 
      fLogger << kFATAL << "number of entries <= 0 in histogram: " << hist->GetTitle() << Endl;

   // another sanity check (nsmooth<0 indicated build with KDE)
   if (nsmooth<0) fLogger << kFATAL << "PDF construction called with nsmooth<0" << Endl;

   fHistOriginal = (TH1*)hist->Clone( TString(hist->GetName()) + "_original" );
   fHist         = (TH1*)hist->Clone( TString(hist->GetName()) + "_smoothed" );
   fHistOriginal->SetDirectory(0);
   fHist        ->SetDirectory(0);

   BuildPDF();

}


//_______________________________________________________________________
TMVA::PDF::PDF( const TH1* hist, KDEKernel::EKernelType ktype, KDEKernel::EKernelIter kiter, 
                KDEKernel::EKernelBorder kborder, Float_t FineFactor)
   : fUseHistogram  ( kFALSE ),
     fNsmooth       (-1 ),
     fInterpolMethod( PDF::kSpline0 ),
     fSpline        ( 0 ),
     fPDFHist       ( 0 ),
     fHist          ( 0 ),
     fHistOriginal  ( 0 ),
     fGraph         ( 0 ),
     fKDEtype       ( ktype ),
     fKDEiter       ( kiter ),
     fKDEborder     ( kborder ),
     fFineFactor    ( FineFactor),
     fLogger        ( this )
{
   // constructor of kernel based PDF:
   // - default kernel type is Gaussian
   // - default number of iterations is 1 (i.e. nonadaptive KDE)
   // sanity check
   if (hist == NULL) fLogger << kFATAL << "called without valid histogram pointer!" << Endl;

   // histogram should be non empty
   if (hist->GetEntries() <= 0) 
      fLogger << kFATAL << "number of entries <= 0 in histogram: " << hist->GetTitle() << Endl;

   fLogger << "create " 
           << ((kiter == KDEKernel::kNonadaptiveKDE) ? "nonadaptive " : 
               (kiter == KDEKernel::kAdaptiveKDE) ? "adaptive " : "??? ")
           << ((ktype == KDEKernel::kGauss) ? "Gauss " : "??? ")
           << "type KDE kernel for histogram: \"" << hist->GetName() << "\""
           << Endl;
      
   fHistOriginal = (TH1*)hist->Clone( TString(hist->GetName()) + "_original" );
   fHist         = (TH1*)hist->Clone( TString(hist->GetName()) + "_smoothed" );   
   fHistOriginal->SetDirectory(0);
   fHist        ->SetDirectory(0);


   FillKDEToHist();
  
}


//_______________________________________________________________________
TMVA::PDF::~PDF()
{
   // destructor
   if (fSpline       != NULL) delete fSpline; 
   if (fHist         != NULL) delete fHist;
   if (fPDFHist      != NULL) delete fPDFHist;
   if (fHistOriginal != NULL) delete fHistOriginal;
}


void TMVA::PDF::BuildPDF() {
   // build the PDF from the original histograms

   // (not useful for discrete distributions, or if no splines are requested)
   if (fInterpolMethod != TMVA::PDF::kSpline0) CheckHist();
    
   // use ROOT TH1 smooth methos
   if (fNsmooth > 0) fHist->Smooth( fNsmooth );
  
   // fill histogramm to graph
   fGraph = new TGraph( fHist );
    
   switch (fInterpolMethod) {

   case TMVA::PDF::kSpline0:
      // use original histogram as reference
      // this is useful, eg, for discrete variables
      fUseHistogram = kTRUE;
      break;

   case TMVA::PDF::kSpline1:
      fSpline = new TMVA::TSpline1( "spline1", fGraph );
      break;

   case TMVA::PDF::kSpline2:
      fSpline = new TMVA::TSpline2( "spline2", fGraph );
      break;

   case TMVA::PDF::kSpline3:
      fSpline = new TSpline3( "spline3", fGraph );
      break;
    
   case TMVA::PDF::kSpline5:
      fSpline = new TSpline5( "spline5", fGraph );
      break;

   default:
      fLogger << kWARNING << "no valid interpolation method given! Use Spline3" << Endl;
      fSpline = new TMVA::TSpline2( "spline2", fGraph );
   }

   // fill into histogram 
   FillSplineToHist();

   if (!UseHistogram()) {
      fSpline->SetTitle( (TString)fHist->GetTitle() + fSpline->GetTitle() );
      fSpline->SetName ( (TString)fHist->GetName()  + fSpline->GetName()  );
   }

   // sanity check
   Double_t integral = GetIntegral();
   if (integral < 0) fLogger << kFATAL << "integral: " << integral << " <= 0" << Endl;

   // normalize
   if (integral>0)  fPDFHist->Scale( 1.0/integral );
}


//_______________________________________________________________________
void TMVA::PDF::FillKDEToHist()
{
   // creates high-binned reference histogram to be used instead of the
   // PDF for speed reasons

   fPDFHist = new TH1F( "", "", NBIN_PdfHist, GetXmin(), GetXmax() );
   fPDFHist->SetTitle( (TString)fHist->GetTitle() + "_hist from_KDE" );
   fPDFHist->SetName ( (TString)fHist->GetName()  + "_hist_from_KDE" );
   
   // create the kernel object  
   TMVA::KDEKernel *kern = new TMVA::KDEKernel(fKDEiter,
                                               fHist,
                                               fPDFHist->GetBinLowEdge(1),
                                               fPDFHist->GetBinLowEdge(fPDFHist->GetNbinsX()+1),
                                               fKDEborder,
                                               fFineFactor);
   kern->SetKernelType(fKDEtype);
   
   Float_t histoLowEdge=fHist->GetBinLowEdge(1);
   Float_t histoUpperEdge=fHist->GetBinLowEdge(fHist->GetNbinsX()+1);

   for(Int_t i=1;i<fHist->GetNbinsX();i++) {
   // loop over the bins of the original histo
      for(Int_t j=1;j<fPDFHist->GetNbinsX();j++) {
         // loop over the bins of the new histo and fill it
         fPDFHist->AddBinContent(j,fHist->GetBinContent(i)*
                                   kern->GetBinKernelIntegral(fPDFHist->GetBinLowEdge(j),
                                                              fPDFHist->GetBinLowEdge(j+1),
                                                              fHist->GetBinCenter(i),
                                                              i)
                                );
      }
   if (fKDEborder == 3) { // mirror the saples and fill them again
   // in order to save time do the mirroring only for the first (the lowwer) 1/5 of the histo to the left; 
   // and the last (the higher) 1/5 of the histo to the right.
   // the middle part of the histo, which is not mirrored, has no influence on the border effects anyway ...
      if (i < fHist->GetNbinsX()/5  ) {  // the first (the lowwer) 1/5 of the histo
         for(Int_t j=1;j<fPDFHist->GetNbinsX();j++) {
            // loop over the bins of the PDF histo and fill it
            fPDFHist->AddBinContent(j,fHist->GetBinContent(i)*
                                    kern->GetBinKernelIntegral(fPDFHist->GetBinLowEdge(j),
                                                               fPDFHist->GetBinLowEdge(j+1),
                                                               2*histoLowEdge-fHist->GetBinCenter(i), //  mirroring to the left
                                                               i)
                                    );
         }
      }
      if (i > 4*fHist->GetNbinsX()/5) { // the last (the higher) 1/5 of the histo
         for(Int_t j=1;j<fPDFHist->GetNbinsX();j++) {
            // loop over the bins of the PDF histo and fill it
            fPDFHist->AddBinContent(j,fHist->GetBinContent(i)*
                                    kern->GetBinKernelIntegral(fPDFHist->GetBinLowEdge(j),
                                                               fPDFHist->GetBinLowEdge(j+1),
                                                               2*histoUpperEdge-fHist->GetBinCenter(i), // mirroring to the right
                                                               i)
                                    );
            }            
         }
      }
   }
          
   fPDFHist->SetEntries(fHist->GetEntries());

   delete kern;
 
   // sanity check
   Double_t integral = GetIntegral();
   if (integral < 0) fLogger << kFATAL << "integral: " << integral << " <= 0" << Endl; 

   // normalize 
   if (integral>0) fPDFHist->Scale( 1.0/integral );
 
}

//_______________________________________________________________________
void TMVA::PDF::FillSplineToHist()
{
   // creates high-binned reference histogram to be used instead of the 
   // PDF for speed reasons 

   if (UseHistogram()) {
      // no spline given, use the original histogram
      fPDFHist = (TH1*)fHist->Clone();
      fPDFHist->SetTitle( (TString)fHist->GetTitle() + "_hist from_spline0" );
      fPDFHist->SetName ( (TString)fHist->GetName()  + "_hist_from_spline0" );      
   }
   else {
      // create new reference histogram
      fPDFHist = new TH1F( "", "", NBIN_PdfHist, GetXmin(), GetXmax() );
      fPDFHist->SetTitle( (TString)fHist->GetTitle() + "_hist from_" + fSpline->GetTitle() );
      fPDFHist->SetName ( (TString)fHist->GetName()  + "_hist_from_" + fSpline->GetTitle() );
      
      for (Int_t bin=1; bin <= NBIN_PdfHist; bin++) {
         Double_t x = fPDFHist->GetBinCenter( bin );
         Double_t y = fSpline->Eval( x );
         // sanity correction: in cases where strong slopes exist, accidentally, the 
         // splines can go to zero; in this case we set the corresponding bin content
         // equal to the bin content of the original histogram
         if (y <= TMVA::PDF_epsilon_) y = fHist->GetBinContent( fHist->FindBin( x ) );
         fPDFHist->SetBinContent( bin, TMath::Max(y, TMVA::PDF_epsilon_) );
      }
   }
   fPDFHist->SetDirectory(0);
}

//_______________________________________________________________________
void TMVA::PDF::CheckHist() const
{
   // sanity check: compare PDF with original histogram
   if (fHist == NULL) {
      fLogger << kFATAL << "<CheckHist> called without valid histogram pointer!" << Endl;
   }

   Int_t nbins = fHist->GetNbinsX();

   Int_t emptyBins=0;
   // count number of empty bins
   for(Int_t bin=1; bin<=nbins; bin++) 
      if (fHist->GetBinContent(bin) == 0) emptyBins++;

   if (((Float_t)emptyBins/(Float_t)nbins) > 0.5) {
      fLogger << kWARNING << "more than 50% (" << (((Float_t)emptyBins/(Float_t)nbins)*100)
              <<"%) of the bins in hist '" 
              << fHist->GetName() << "' are empty!" << Endl;
      fLogger << kWARNING << "X_min=" << GetXmin() 
              << " mean=" << fHist->GetMean() << " X_max= " << GetXmax() << Endl;  
   }

   if (DEBUG_PDF) 
      fLogger << kINFO << GetXmin() << " < x < " << GetXmax() << " in " << nbins << " bins" << Endl;
}

//_______________________________________________________________________
void TMVA::PDF::ValidatePDF( TH1* originalHist ) const
{
   // comparison of original histogram with reference PDF

   // if no histogram is given, use the original one (the one the PDF was made from)
   if (!originalHist) originalHist = fHistOriginal;

   Int_t    nbins = originalHist->GetNbinsX();

   // treat errors properly
   if (originalHist->GetSumw2()->GetSize() == 0) originalHist->Sumw2();

   // ---- first validation: simple(st) possible chi2 test
   // count number of empty bins
   Double_t chi2 = 0;
   Int_t    ndof = 0;
   Int_t    nc1  = 0; // deviation counters
   Int_t    nc2  = 0; // deviation counters
   Int_t    nc3  = 0; // deviation counters
   Int_t    nc6  = 0; // deviation counters
   for(Int_t bin=1; bin<=nbins; bin++) {
      Double_t x  = originalHist->GetBinCenter( bin );
      Double_t y  = originalHist->GetBinContent( bin );
      Double_t ey = originalHist->GetBinError( bin );

      Int_t binPdfHist = fPDFHist->FindBin( x );

      Double_t yref = GetVal( x );      
      Double_t rref = ( originalHist->GetSumOfWeights()/fPDFHist->GetSumOfWeights() * 
                        originalHist->GetBinWidth( bin )/fPDFHist->GetBinWidth( binPdfHist ) );

      if (y > 0) {
         ndof++;
         Double_t d = TMath::Abs( (y - yref*rref)/ey );
         chi2 += d*d;
         if (d > 1) { nc1++; if (d > 2) { nc2++; if (d > 3) { nc3++; if (d > 6) nc6++; } } }
      }
   }
   
   fLogger << "validation result for PDF \"" << originalHist->GetTitle() << "\"" << ": " << Endl;
   fLogger << Form( "    chi2/ndof(!=0) = %.1f/%i = %.2f (Prob = %.2f)", 
                    chi2, ndof, chi2/ndof, TMath::Prob( chi2, ndof ) ) << Endl;
   fLogger << Form( "    #bins-found(#expected-bins) deviating > [1,2,3,6] sigmas: " \
                    "[%i(%i),%i(%i),%i(%i),%i(%i)]", 
                    nc1, Int_t(TMath::Prob(1.0,1)*ndof), nc2, Int_t(TMath::Prob(4.0,1)*ndof),
                    nc3, Int_t(TMath::Prob(9.0,1)*ndof), nc6, Int_t(TMath::Prob(36.0,1)*ndof) ) << Endl;
}

//_______________________________________________________________________
Double_t  TMVA::PDF::GetIntegral() const
{
   // computes normalisation
   return GetIntegral( GetXmin(), GetXmax() );
}

//_______________________________________________________________________
Double_t TMVA::PDF::GetIntegral( Double_t xmin, Double_t xmax ) const
{  
   // computes PDF integral within given ranges
   Double_t  integral = 0;

   if (UseHistogram()) {
      integral = fPDFHist->GetSumOfWeights();
   }
   else {
      Int_t     nsteps = 10000;
      Double_t  intBin = (xmax - xmin)/nsteps; // bin width for integration
      for (Int_t bini=0; bini < nsteps; bini++) {
         Double_t x = (bini + 0.5)*intBin + xmin;
         integral += GetVal( x );
      }
      integral *= intBin;
   }
  
   return integral;
}

//_______________________________________________________________________
Double_t TMVA::PDF::GetVal( Double_t x ) const
{  
   // returns value PDF(x)

   // check which is filled
   Int_t bin = fPDFHist->FindBin(x);
   bin = TMath::Max(bin,1);
   bin = TMath::Min(bin,fPDFHist->GetNbinsX());

   Double_t retval = 0;

   if (UseHistogram()) 
      retval = fPDFHist->GetBinContent( bin );

   else {
      Int_t nextbin = bin;
      if ((x > fPDFHist->GetBinCenter(bin) && bin != fPDFHist->GetNbinsX()) || bin == 1)
         nextbin++;
      else
         nextbin--;  
      
      // linear interpolation between adjacent bins
      Double_t dx = fPDFHist->GetBinCenter( bin )  - fPDFHist->GetBinCenter( nextbin );
      Double_t dy = fPDFHist->GetBinContent( bin ) - fPDFHist->GetBinContent( nextbin );
      retval = fPDFHist->GetBinContent( bin ) + (x - fPDFHist->GetBinCenter( bin ))*dy/dx;
   }

   return TMath::Max( retval, TMVA::PDF_epsilon_ );
}

//_______________________________________________________________________
ostream& TMVA::operator<< (ostream& os, const TMVA::PDF& pdf)
{ 
   // write the pdf
   os << "NSmooth         " << pdf.fNsmooth << endl;
   os << "InterpolMethod  " << pdf.fInterpolMethod << endl;
   os << "KDE_type        " << pdf.fKDEtype << endl;
   os << "KDE_iter        " << pdf.fKDEiter << endl;
   os << "KDE_border      " << pdf.fKDEborder << endl;
   os << "KDE_finefactor  " << pdf.fFineFactor << endl;

   TH1 * histToWrite = pdf.GetOriginalHist();

   const Int_t nBins = histToWrite->GetNbinsX();
   os << nBins                     // nbins
      << "   " << histToWrite->GetXaxis()->GetXmin()    // x_min
      << "   " << histToWrite->GetXaxis()->GetXmax()    // x_max
      << endl;

   // write the smoothed hist
   //   os.std::setf(ios::left);
   for(Int_t i=0; i<nBins; i++) {
      os << std::setw(15) << std::left << histToWrite->GetBinContent(i+1);
      if ((i+1)%5==0) os << endl;
   }
   return os; // Return the output stream.
}

//_______________________________________________________________________
istream& TMVA::operator>> (istream& istr, TMVA::PDF& pdf)
{ 

   // read the tree from an istream
   TString devnullS;
   Int_t   valI;
   istr >> devnullS >> pdf.fNsmooth;
   // have to do this with strings to be more stable with developing code
   istr >> devnullS >> valI; pdf.fInterpolMethod = TMVA::PDF::EInterpolateMethod(valI);
   istr >> devnullS >> valI; pdf.fKDEtype        = TMVA::KDEKernel::EKernelType(valI);  
   istr >> devnullS >> valI; pdf.fKDEiter        = TMVA::KDEKernel::EKernelIter(valI);
   istr >> devnullS >> valI; pdf.fKDEborder      = TMVA::KDEKernel::EKernelBorder(valI);
   istr >> devnullS >> pdf.fFineFactor;
  
   Int_t   nbins;
   Float_t xmin, xmax;
   istr >> nbins >> xmin >> xmax;

   // write the smoothed hist
   TH1 * newhist = new TH1F("_original","", nbins, xmin, xmax);
   newhist->SetDirectory(0);

   Float_t val;
   for(Int_t i=0; i<nbins; i++) {
      istr >> val;
      newhist->SetBinContent(i+1,val);
   }
   
   if (pdf.fHistOriginal != 0) delete pdf.fHistOriginal;
   pdf.fHistOriginal = newhist;
   pdf.fHist = (TH1*)pdf.fHistOriginal->Clone( "_smoothed" );
   pdf.fHist ->SetDirectory(0);

   if (pdf.fNsmooth>=0) pdf.BuildPDF();
   else                 pdf.FillKDEToHist();

   return istr;
}
