// Author: Kerstin Tackmann, Andreas Hoecker, Heiko Lacker

/**********************************************************************************
 *                                                                                *
 * Project: TSVDUnfold - data unfolding based on Singular Value Decomposition     *
 * Package: ROOT                                                                  *
 * Class  : TSVDUnfold                                                            *
 *                                                                                *
 * Description:                                                                   *
 *      Single class implementation of SVD data unfolding based on:               *
 *          A. Hoecker, V. Kartvelishvili,                                        *
 *          "SVD approach to data unfolding"                                      *
 *          NIM A372, 469 (1996) [hep-ph/9509307]                                 *
 *                                                                                *
 * Authors:                                                                       *
 *      Kerstin Tackmann <Kerstin.Tackmann@cern.ch>   - CERN, Switzerland         *
 *      Andreas Hoecker  <Andreas.Hoecker@cern.ch>    - CERN, Switzerland         *
 *      Heiko Lacker     <lacker@physik.hu-berlin.de> - Humboldt U, Germany       *
 *                                                                                *
 * Copyright (c) 2010:                                                            *
 *      CERN, Switzerland                                                         *
 *      Humboldt University, Germany                                              *
 *                                                                                *
 **********************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TSVDUnfold                                                           //
//                                                                      //
// Data unfolding using Singular Value Decomposition (hep-ph/9509307)   //
// Authors: Kerstin Tackmann, Andreas Hoecker, Heiko Lacker             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

//_______________________________________________________________________
/* Begin_Html
<center><h2>SVD Approach to Data Unfolding</h2></center>
<p>
Reference: <a href="http://arXiv.org/abs/hep-ph/9509307">Nucl. Instrum. Meth. A372, 469 (1996) [hep-ph/9509307]</a>
<p>
TSVDUnfold implements the singular value decomposition based unfolding method (see reference). Currently, the unfolding of one-dimensional histograms is supported, with the same number of bins for the measured and the unfolded spectrum.
<p>
The unfolding procedure is based on singular value decomposition of the response matrix. The regularisation of the unfolding is implemented via a discrete minimum-curvature condition.
<p>
Monte Carlo inputs:
<ul>
<li><tt>xini</tt>: true underlying spectrum (TH1D, n bins)
<li><tt>bini</tt>: reconstructed spectrum (TH1D, n bins)
<li><tt>Adet</tt>: response matrix (TH2D, nxn bins)
</ul>
Consider the unfolding of a measured spectrum <tt>bdat</tt>. The corresponding spectrum in the Monte Carlo is given by <tt>bini</tt>, with the true underlying spectrum given by <tt>xini</tt>. The detector response is described by <tt>Adet</tt>, with <tt>Adet</tt> filled with events (not probabilities) with the true observable on the y-axis and the reconstructed observable on the x-axis.
<p>
The measured distribution can be unfolded for any combination of resolution, efficiency and acceptance effects, provided an appropriate definition of <tt>xini</tt> and <tt>Adet</tt>.<br><br>
<p>
The unfolding can be performed by
<ul>
<pre>
TSVDUnfold *tsvdunf = new TSVDUnfold();
tsvdunf->Init( bdat, bini, xini, Adet );
TH1D* unfresult = tsvdunf->Unfold(kreg);
</pre>
</ul>
where <tt>kreg</tt> determines the regularisation of the unfolding. In general, overregularisation (too small <tt>kreg</tt>) will bias the unfolded spectrum towards the Monte Carlo input, while underregularisation (too large <tt>kreg</tt>) will lead to large fluctuations in the unfolded spectrum. The optimal regularisation can be determined following guidelines in <a href="http://arXiv.org/abs/hep-ph/9509307">Nucl. Instrum. Meth. A372, 469 (1996) [hep-ph/9509307]</a> using the distribution of the <tt>|d_i|<\tt> that can be obtained by <tt>tsvdunf->GetD()</tt> and/or using pseudo-experiments.
<p>
Covariance matrices on the measured spectrum can be propagated to covariance matrices using the <tt>GetUnfoldCovMatrix</tt> method, which uses pseudo experiments for the propagation. In addition, <tt>GetAdetCovMatrix</tt> allows for the propagation of the statistical uncertainties on the response matrix using pseudo experiments. In addition, the distribution of singular values can be retrieved using <tt>tsvdunf->GetSV()<\tt>.
End_Html */
//_______________________________________________________________________


#include <iostream>

#include "TSVDUnfold.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TDecompSVD.h"
#include "TRandom3.h"
#include "TMath.h"

ClassImp(TSVDUnfold)

using namespace std;

//_______________________________________________________________________
TSVDUnfold::TSVDUnfold( const TH1D *bdat, const TH1D *bini, const TH1D *xini, const TH2D *Adet )
   : TObject     (),
     fNdim       (0),
     fDdim       (2),
     fNormalize  (kFALSE),
     fKReg       (-1),
     fDHist      (NULL),
     fSVHist     (NULL),
     fBdat       (bdat), 
     fBini       (bini),
     fXini       (xini),
     fAdet       (Adet), 
     fToyhisto   (NULL),
     fToymat     (NULL),
     fToyMode    (kFALSE),
     fMatToyMode (kFALSE) 
{
   // Default constructor

   // Initialisation of TSVDUnfold
   // User provides data and MC test spectra, as well as detector response matrix
   if (bdat->GetNbinsX() != bini->GetNbinsX() || 
       bdat->GetNbinsX() != xini->GetNbinsX() ||
       bdat->GetNbinsX() != Adet->GetNbinsX() ||
       bdat->GetNbinsX() != Adet->GetNbinsY()) {
      TString msg = "All histograms must have equal dimension.\n";
      msg += Form( "  Found: dim(bdat)=%i\n",    bdat->GetNbinsX() );
      msg += Form( "  Found: dim(bini)=%i\n",    bini->GetNbinsX() );
      msg += Form( "  Found: dim(xini)=%i\n",    xini->GetNbinsX() );
      msg += Form( "  Found: dim(Adet)=%i,%i\n", Adet->GetNbinsX(), Adet->GetNbinsY() );
      msg += "Please start again!";

      Fatal( "Init", msg, "%s" );
   }

   // Get the input histos
   fNdim = bdat->GetNbinsX();
   fDdim = 2; // This is the derivative used to compute the curvature matrix
}

//_______________________________________________________________________
TSVDUnfold::TSVDUnfold( const TSVDUnfold& other )
   : TObject     ( other ),
     fNdim       (other.fNdim),
     fDdim       (other.fDdim),
     fNormalize  (other.fNormalize),
     fKReg       (other.fKReg),
     fDHist      (other.fDHist),
     fSVHist     (other.fSVHist),
     fBdat       (other.fBdat),
     fBini       (other.fBini),
     fXini       (other.fXini),
     fAdet       (other.fAdet),
     fToyhisto   (other.fToyhisto),
     fToymat     (other.fToymat),
     fToyMode    (other.fToyMode),
     fMatToyMode (other.fMatToyMode) 
{
   // Copy constructor
}

//_______________________________________________________________________
TSVDUnfold::~TSVDUnfold()
{
   // Destructor
}

//_______________________________________________________________________
TH1D* TSVDUnfold::Unfold( Int_t kreg )
{
   // Perform the unfolding   
   fKReg = kreg;
   
   // Make the histos
   if (!fToyMode && !fMatToyMode) InitHistos( );

   // Create vectors and matrices
   TVectorD vb(fNdim), vbini(fNdim), vxini(fNdim), vberr(fNdim);
   TMatrixD mA(fNdim, fNdim), mCurv(fNdim, fNdim), mC(fNdim, fNdim);

   Double_t eps = 1e-12;
   Double_t sreg;

   // Copy histogams entries into vector
   if (fToyMode) { H2V( fToyhisto, vb ); H2Verr( fToyhisto, vberr ); }
   else          { H2V( fBdat,     vb ); H2Verr( fBdat,     vberr ); }

   H2V( fBini, vbini );
   H2V( fXini, vxini );
   if (fMatToyMode) H2M( fToymat, mA );
   else        H2M( fAdet,   mA );

   // Scale the MC vectors to data norm
   Double_t scale = vb.Sum()/vbini.Sum();
   vbini *= scale;
   vxini *= scale;
   mA    *= scale;
  
   // Fill and invert the second derivative matrix
   FillCurvatureMatrix( mCurv, mC );

   // Inversion of mC by help of SVD
   TDecompSVD CSVD(mC);
   TMatrixD CUort = CSVD.GetU();
   TMatrixD CVort = CSVD.GetV();
   TVectorD CSV   = CSVD.GetSig();

   TMatrixD CSVM(fNdim, fNdim);
   for (Int_t i=0; i<fNdim; i++) CSVM(i,i) = 1/CSV(i);

   CUort.Transpose( CUort );
   TMatrixD mCinv = (CVort*CSVM)*CUort;

   // Rescale matrix and vectors by error of data vector
   vbini = VecDiv   ( vbini, vberr );
   vb    = VecDiv   ( vb,    vberr, 1 );
   mA    = MatDivVec( mA,    vberr, 1 );
   vberr = VecDiv   ( vberr, vberr, 1 );

   // Singular value decomposition and matrix operations
   TDecompSVD ASVD( mA*mCinv );
   TMatrixD Uort = ASVD.GetU();
   TMatrixD Vort = ASVD.GetV();
   TVectorD ASV  = ASVD.GetSig();

   if (!fToyMode && !fMatToyMode) {
      V2H(ASV, *fSVHist);
   }

   TMatrixD Vreg = mCinv*Vort;

   Uort.Transpose(Uort);
   TVectorD vdini = Uort*vbini;
   TVectorD vd    = Uort*vb;

   if (!fToyMode && !fMatToyMode) {
      V2H(vd, *fDHist);
   }

   // Damping coefficient
   Int_t k = GetKReg()-1; 

   TVectorD vx(fNdim); // Return variable

   // Damping factors
   TVectorD vdz(fNdim);
   for (Int_t i=0; i<fNdim; i++) {
     if (ASV(i)<ASV(0)*eps) sreg = ASV(0)*eps;
     else                   sreg = ASV(i);
     vdz(i) = sreg/(sreg*sreg + ASV(k)*ASV(k));
   }
   TVectorD vz = CompProd( vd, vdz );
   
   // Compute the weights
   TVectorD vw = Vreg*vz;
   
   // Rescale by xini
   vx = CompProd( vw, vxini );
   
   if(fNormalize){ // Scale result to unit area
     scale = vx.Sum();
     if (scale > 0) vx *= 1.0/scale;
   }
   
   // Get Curvature and also chi2 in case of MC unfolding
   if (!fToyMode && !fMatToyMode) {
     Info( "Unfold", "Unfolding param: %i",k+1 );
     Info( "Unfold", "Curvature of weight distribution: %f", GetCurvature( vw, mCurv ) );
   }

   TH1D* h = (TH1D*)fBdat->Clone("unfoldingresult");
   for(int i=1; i<=fNdim; i++){
      h->SetBinContent(i,0.);
      h->SetBinError(i,0.);
   }
   V2H( vx, *h );

   return h;
}

//_______________________________________________________________________
TH2D* TSVDUnfold::GetUnfoldCovMatrix( const TH2D* cov, Int_t ntoys, Int_t seed )
{

   fToyMode = true;
   TH1D* unfres = 0;
   TH2D* unfcov = (TH2D*)fAdet->Clone("unfcovmat");
   for(int i=1; i<=fNdim; i++)
      for(int j=1; j<=fNdim; j++)
         unfcov->SetBinContent(i,j,0.);
  
   // Code for generation of toys (taken from RooResult and modified)
   // Calculate the elements of the upper-triangular matrix L that
   // gives Lt*L = C, where Lt is the transpose of L (the "square-root method")  
   TMatrixD L(fNdim,fNdim); L *= 0;

   for (Int_t iPar= 0; iPar < fNdim; iPar++) {

      // Calculate the diagonal term first
      L(iPar,iPar) = cov->GetBinContent(iPar+1,iPar+1);
      for (Int_t k=0; k<iPar; k++) L(iPar,iPar) -= TMath::Power( L(k,iPar), 2 );
      if (L(iPar,iPar) > 0.0) L(iPar,iPar) = TMath::Sqrt(L(iPar,iPar));
      else                    L(iPar,iPar) = 0.0;

      // ...then the off-diagonal terms
      for (Int_t jPar=iPar+1; jPar<fNdim; jPar++) {
         L(iPar,jPar) = cov->GetBinContent(iPar+1,jPar+1);
         for (Int_t k=0; k<iPar; k++) L(iPar,jPar) -= L(k,iPar)*L(k,jPar);
         if (L(iPar,iPar)!=0.) L(iPar,jPar) /= L(iPar,iPar);
         else                  L(iPar,jPar) = 0;
      }
   }

   // Remember it
   TMatrixD *Lt = new TMatrixD(TMatrixD::kTransposed,L);
   TRandom3 random(seed);

   fToyhisto = (TH1D*)fBdat->Clone("toyhisto");
   TH1D *toymean = (TH1D*)fBdat->Clone("toymean");
   for (Int_t j=1; j<=fNdim; j++) toymean->SetBinContent(j,0.);

   // Get the mean of the toys first
   for (int i=1; i<=ntoys; i++) {

      // create a vector of unit Gaussian variables
      TVectorD g(fNdim);
      for (Int_t k= 0; k < fNdim; k++) g(k) = random.Gaus(0.,1.);

      // Multiply this vector by Lt to introduce the appropriate correlations
      g *= (*Lt);

      // Add the mean value offsets and store the results
      for (int j=1; j<=fNdim; j++) {
         fToyhisto->SetBinContent(j,fBdat->GetBinContent(j)+g(j-1));
         fToyhisto->SetBinError(j,fBdat->GetBinError(j));
      }

      unfres = Unfold(GetKReg());

      for (Int_t j=1; j<=fNdim; j++) {
         toymean->SetBinContent(j, toymean->GetBinContent(j) + unfres->GetBinContent(j)/ntoys);
      }
   }

   // Reset the random seed
   random.SetSeed(seed);

   //Now the toys for the covariance matrix
   for (int i=1; i<=ntoys; i++) {

      // Create a vector of unit Gaussian variables
      TVectorD g(fNdim);
      for (Int_t k= 0; k < fNdim; k++) g(k) = random.Gaus(0.,1.);

      // Multiply this vector by Lt to introduce the appropriate correlations
      g *= (*Lt);

      // Add the mean value offsets and store the results
      for (int j=1; j<=fNdim; j++) {
         fToyhisto->SetBinContent( j, fBdat->GetBinContent(j)+g(j-1) );
         fToyhisto->SetBinError  ( j, fBdat->GetBinError(j) );
      }
      unfres = Unfold(GetKReg());

      for (Int_t j=1; j<=fNdim; j++) {
         for (Int_t k=1; k<=fNdim; k++) {
            unfcov->SetBinContent(j,k,unfcov->GetBinContent(j,k) + ( (unfres->GetBinContent(j) - toymean->GetBinContent(j))* (unfres->GetBinContent(k) - toymean->GetBinContent(k))/(ntoys-1)) );
         }
      }
   }
   delete Lt;
   delete unfres;
   fToyMode = kFALSE;
   
   return unfcov;
}

//_______________________________________________________________________
TH2D* TSVDUnfold::GetAdetCovMatrix( Int_t ntoys, Int_t seed )
{

   fMatToyMode = true;
   TH1D* unfres = 0;
   TH2D* unfcov = (TH2D*)fAdet->Clone("unfcovmat");
   for(int i=1; i<=fNdim; i++)
      for(int j=1; j<=fNdim; j++)
         unfcov->SetBinContent(i,j,0.);

   //Now the toys for the detector response matrix
   TRandom3 random(seed);

   fToymat = (TH2D*)fAdet->Clone("toymat");
   TH1D *toymean = (TH1D*)fXini->Clone("toymean");
   for (Int_t j=1; j<=fNdim; j++) toymean->SetBinContent(j,0.);

   for (int i=1; i<=ntoys; i++) {    
      for (Int_t k=1; k<=fNdim; k++) {
         for (Int_t m=1; m<=fNdim; m++) {
            if (fAdet->GetBinContent(k,m)) {
               fToymat->SetBinContent(k, m, random.Poisson(fAdet->GetBinContent(k,m)));
            }
         }
      }

      unfres = Unfold(GetKReg());

      for (Int_t j=1; j<=fNdim; j++) {
         toymean->SetBinContent(j, toymean->GetBinContent(j) + unfres->GetBinContent(j)/ntoys);
      }
   }

   // Reset the random seed
   random.SetSeed(seed);

   for (int i=1; i<=ntoys; i++) {
      for (Int_t k=1; k<=fNdim; k++) {
         for (Int_t m=1; m<=fNdim; m++) {
            if (fAdet->GetBinContent(k,m))
               fToymat->SetBinContent(k, m, random.Poisson(fAdet->GetBinContent(k,m)));
         }
      }

      unfres = Unfold(GetKReg());

      for (Int_t j=1; j<=fNdim; j++) {
         for (Int_t k=1; k<=fNdim; k++) {
            unfcov->SetBinContent(j,k,unfcov->GetBinContent(j,k) + ( (unfres->GetBinContent(j) - toymean->GetBinContent(j))*(unfres->GetBinContent(k) - toymean->GetBinContent(k))/(ntoys-1)) );
         }
      }
   }
   delete unfres;
   fMatToyMode = kFALSE;
   
   return unfcov;
}

//_______________________________________________________________________
TH1D* TSVDUnfold::GetD() const 
{ 
   // Returns d vector
   for (int i=1; i<=fDHist->GetNbinsX(); i++) {
      if (fDHist->GetBinContent(i)<0.) fDHist->SetBinContent(i, TMath::Abs(fDHist->GetBinContent(i))); 
   }
   return fDHist; 
}

//_______________________________________________________________________
TH1D* TSVDUnfold::GetSV() const 
{ 
   // Returns singular values vector
   return fSVHist; 
}

//_______________________________________________________________________
void TSVDUnfold::H2V( const TH1D* histo, TVectorD& vec )
{
   // Fill 1D histogram into vector
   for (Int_t i=0; i<histo->GetNbinsX(); i++) vec(i) = histo->GetBinContent(i+1);
}

//_______________________________________________________________________
void TSVDUnfold::H2Verr( const TH1D* histo, TVectorD& vec )
{
   // Fill 1D histogram errors into vector
   for (Int_t i=0; i<histo->GetNbinsX(); i++) vec(i) = histo->GetBinError(i+1);
}

//_______________________________________________________________________
void TSVDUnfold::V2H( const TVectorD& vec, TH1D& histo )
{
   // Fill vector into 1D histogram
   for(Int_t i=0; i<vec.GetNrows(); i++) histo.SetBinContent(i+1, vec(i));
}

//_______________________________________________________________________
void TSVDUnfold::H2M( const TH2D* histo, TMatrixD& mat )
{
   // Fill 2D histogram into matrix
   for (Int_t j=0; j<histo->GetNbinsX(); j++) {
      for (Int_t i=0; i<histo->GetNbinsY(); i++) {
         mat(i,j) = histo->GetBinContent(i+1,j+1);
      }
   }
}

//_______________________________________________________________________
TVectorD TSVDUnfold::VecDiv( const TVectorD& vec1, const TVectorD& vec2, Int_t zero )
{
   // Divide entries of two vectors
   TVectorD quot(vec1.GetNrows());
   for (Int_t i=0; i<vec1.GetNrows(); i++) {
      if (vec2(i) != 0) quot(i) = vec1(i) / vec2(i);
      else {
         if   (zero) quot(i) = 0;
         else        quot(i) = vec1(i);
      }
   }
   return quot;
}

//_______________________________________________________________________
TMatrixD TSVDUnfold::MatDivVec( const TMatrixD& mat, const TVectorD& vec, Int_t zero )
{
   // Divide matrix entries by vector
   TMatrixD quotmat(mat.GetNrows(), mat.GetNcols());
   for (Int_t i=0; i<mat.GetNrows(); i++) {
      for (Int_t j=0; j<mat.GetNcols(); j++) {
         if (vec(i) != 0) quotmat(i,j) = mat(i,j) / vec(i);
         else {
            if   (zero) quotmat(i,j) = 0;
            else        quotmat(i,j) = mat(i,j);
         }
      }
   }
   return quotmat;
}

//_______________________________________________________________________
TVectorD TSVDUnfold::CompProd( const TVectorD& vec1, const TVectorD& vec2 )
{
   // Multiply entries of two vectors
   TVectorD res(vec1.GetNrows());
   for (Int_t i=0; i<vec1.GetNrows(); i++) res(i) = vec1(i) * vec2(i);
   return res;
}

//_______________________________________________________________________
Double_t TSVDUnfold::GetCurvature(const TVectorD& vec, const TMatrixD& curv) 
{      
   // Compute curvature of vector
   return vec*(curv*vec);
}

//_______________________________________________________________________
void TSVDUnfold::FillCurvatureMatrix( TMatrixD& tCurv, TMatrixD& tC ) const
{
   Double_t eps = 0.00001;

   Int_t ndim = tCurv.GetNrows();

   // Init
   tCurv *= 0;
   tC    *= 0;

   if (fDdim == 0) for (Int_t i=0; i<ndim; i++) tC(i,i) = 1;
   else if (ndim == 1) {
      for (Int_t i=0; i<ndim; i++) {
         if (i < ndim-1) tC(i,i+1) = 1.0;
         tC(i,i) = 1.0;
      }
   }
   else if (fDdim == 2) {
      for (Int_t i=0; i<ndim; i++) {
         if (i > 0)      tC(i,i-1) = 1.0;
         if (i < ndim-1) tC(i,i+1) = 1.0;
         tC(i,i) = -2.0;
      }
      tC(0,0) = -1.0;
      tC(ndim-1,ndim-1) = -1.0;
   }
   else if (fDdim == 3) {
      for (Int_t i=1; i<ndim-2; i++) {
         tC(i,i-1) =  1.0;
         tC(i,i)   = -3.0;
         tC(i,i+1) =  3.0;
         tC(i,i+2) = -1.0;
      }
   }
   else if (fDdim==4) {
      for (Int_t i=0; i<ndim; i++) {
         if (i > 0)      tC(i,i-1) = -4.0;
         if (i < ndim-1) tC(i,i+1) = -4.0;
         if (i > 1)      tC(i,i-2) =  1.0;
         if (i < ndim-2) tC(i,i+2) =  1.0;
         tC(i,i) = 6.0;
      }
      tC(0,0) = 2.0;
      tC(ndim-1,ndim-1) = 2.0;
      tC(0,1) = -3.0;
      tC(ndim-2,ndim-1) = -3.0;
      tC(1,0) = -3.0;
      tC(ndim-1,ndim-2) = -3.0;
      tC(1,1) =  6.0;
      tC(ndim-2,ndim-2) =  6.0;
   }
   else if (fDdim == 5) {
      for (Int_t i=2; i < ndim-3; i++) {
         tC(i,i-2) = 1.0;
         tC(i,i-1) = -5.0;
         tC(i,i)   = 10.0;
         tC(i,i+1) = -10.0;
         tC(i,i+2) = 5.0;
         tC(i,i+3) = -1.0;
      }
   }
   else if (fDdim == 6) {
      for (Int_t i = 3; i < ndim - 3; i++) {
         tC(i,i-3) = 1.0;
         tC(i,i-2) = -6.0;
         tC(i,i-1) = 15.0;
         tC(i,i)   = -20.0;
         tC(i,i+1) = 15.0;
         tC(i,i+2) = -6.0;
         tC(i,i+3) = 1.0;
      }
   }

   // Add epsilon to avoid singularities
   for (Int_t i=0; i<ndim; i++) tC(i,i) = tC(i,i) + eps;

   //Get curvature matrix
   for (Int_t i=0; i<ndim; i++) {
      for (Int_t j=0; j<ndim; j++) {
         for (Int_t k=0; k<ndim; k++) {
            tCurv(i,j) = tCurv(i,j) + tC(k,i)*tC(k,j);
         }
      }
   }
}

//_______________________________________________________________________
void TSVDUnfold::InitHistos( )
{

   fDHist = new TH1D( "dd", "d vector after orthogonal transformation", fNdim, 0, fNdim );  
   fDHist->Sumw2();

   fSVHist = new TH1D( "sv", "Singular values of AC^-1", fNdim, 0, fNdim );  
   fSVHist->Sumw2();
}

//_______________________________________________________________________
void TSVDUnfold::RegularisedSymMatInvert( TMatrixDSym& mat, Double_t eps )
{
   // naive regularised inversion cuts off small elements

   // init reduced matrix
   const UInt_t n = mat.GetNrows();
   UInt_t nn = 0;   

   UInt_t *ipos = new UInt_t[n];
   //   UInt_t ipos[n];

   // find max diagonal entries
   Double_t ymax = 0;
   for (UInt_t i=0; i<n; i++) if (TMath::Abs(mat[i][i]) > ymax) ymax = TMath::Abs(mat[i][i]);

   for (UInt_t i=0; i<n; i++) {

         // save position of accepted entries
      if (TMath::Abs(mat[i][i])/ymax > eps) ipos[nn++] = i;
   }

   // effective matrix
   TMatrixDSym matwork( nn );
   for (UInt_t in=0; in<nn; in++) for (UInt_t jn=0; jn<nn; jn++) matwork(in,jn) = 0;

   // fill non-zero effective working matrix
   for (UInt_t in=0; in<nn; in++) {

      matwork[in][in] = mat[ipos[in]][ipos[in]];
      for (UInt_t jn=in+1; jn<nn; jn++) {
         matwork[in][jn] = mat[ipos[in]][ipos[jn]];
         matwork[jn][in] = matwork[in][jn];
      }
   }

   // invert
   matwork.Invert();

   // reinitialise old matrix
   for (UInt_t i=0; i<n; i++) for (UInt_t j=0; j<n; j++) mat[i][j] = 0;

   // refill inverted matrix in old one
   for (UInt_t in=0; in<nn; in++) {
      mat[ipos[in]][ipos[in]] = matwork[in][in];
      for (UInt_t jn=in+1; jn<nn; jn++) {
         mat[ipos[in]][ipos[jn]] = matwork[in][jn];
         mat[ipos[jn]][ipos[in]] = mat[ipos[in]][ipos[jn]];
      }
   }
   delete []  ipos;
}

//_______________________________________________________________________
Double_t TSVDUnfold::ComputeChiSquared( const TH1D& truspec, const TH1D& unfspec, const TH2D& covmat, Double_t regpar  )
{
   // helper routine to compute chi-squared between distributions
   UInt_t n = truspec.GetNbinsX();
   TMatrixDSym mat( n );
   for (UInt_t i=0; i<n; i++) {
      for (UInt_t j=i; j<n; j++) {
         mat[i][j] = covmat.GetBinContent( i+1, j+1 );
         mat[j][i] = mat[i][j];
      }
   }

   RegularisedSymMatInvert( mat, regpar );

   // compute chi2
   Double_t chi2 = 0;
   for (UInt_t i=0; i<n; i++) {
      for (UInt_t j=0; j<n; j++) {
         chi2 += ( (truspec.GetBinContent( i+1 )-unfspec.GetBinContent( i+1 )) *
                   (truspec.GetBinContent( j+1 )-unfspec.GetBinContent( j+1 )) * mat[i][j] );
      }
   }

   return chi2;
}

