// Author: Stefan Schmitt
// DESY, 13/10/08


//  Version 6, replace class XY by std::pair
//
//  History:
//    Version 5, replace run-time dynamic arrays by new and delete[]
//    Version 4, fix new bug from V3 with initial regularisation condition
//    Version 3, fix bug with initial regularisation condition
//    Version 2, with improved ScanLcurve() algorithm
//    Version 1, added ScanLcurve() method
//    Version 0, stable version of basic unfolding algorithm

////////////////////////////////////////////////////////////////////////
//
// TUnfold solves the inverse problem
//
//   chi**2 = 1/2 * (y-Ax)# V (y-Ax) + tau (L(x-x0))# L(x-x0)
//
// where # means that the matrix is transposed
//
// Monte Carlo input
// -----------------
//   y: vector of measured quantities  (dimension ny)
//   V: inverse of covariance matrix for y (dimension ny x ny)
//      in many cases V is diagonal and calculated from the errors of y
//   A: migration matrix               (dimension ny x nx)
//   x: unknown underlying distribution (dimension nx)
//
// Regularisation
// --------------
//   tau: parameter, defining the regularisation strength
//   L: matrix of regularisation conditions (dimension nl x nx)
//   x0: bias distribution
//                                                                     
// and chi**2 is minimized as a function of x                        
//                                                                      
// This applies to a very large number of problems, where the measured
// distribution y is a linear superposition of several Monte Carlo shapes
// and the sum of these shapes gives the output distribution x
//
//
//
// Some random examples:
// ======================
//   (1) measure a cross-section as a function of, say, E_T(detector)
//        and unfold it to obtain the underlying distribution E_T(generator)
//   (2) measure a lifetime distribution and unfold the contributions from
//        different flavours
//   (3) measure the transverse mass and decay angle
//        and unfold for the true mass distribution plus background
//
//
//
// References:
// ===========
// Talk by V. Blobel, Terascale Statistics school
//   https://indico.desy.de/contributionDisplay.py?contribId=23&confId=1149
// References quoted in Blobel's talk:
//   Per Chistian Hansen, Rank-Deficient and Discrete Ill-posed Problems,
//        Siam (1998)
//   Jari Kaipio and Erkki Somersalo, Statistical and Computational 
//        Inverse problems, Springer (2005)
//
//
//
// Implementation
// ==============
// The result of the unfolding is calculated as follows:
//
//    Lsquared = L#L                 regularisation conditions squared
//
//    Einv  = ((A# V A)+tau Lsquared)    is the inverse covariance matrix of x
//
//    E = inverse(Einv)                  is the covariance matrix of x
//
//    x = E A V y + tau Lsquared x0   is the result
//
//
//
// Warning:
// ========
//  The algorithm is based on "standard" matrix inversion, with the
//  known limitations in numerical accuracy and computing cost for
//  matrices with large dimensions.
//
//  Thus the algorithm should not used for large dimensions of x and y
//    nx should not be much larger than 200
//    ny should not be much larger than 1000
//
//
//
// Example of using TUnfold:
// =========================
// imagine a 2-dimensional histogram is filled, named A
//    y-axis: generated quantity (e.g. 10 bins)
//    x-axis: reconstructed quantity (e.g. 20 bin)
// The data are filled in a 1-dimensional histogram, named y
// Note1: ALWAYS choose a higher number of bins on the reconstructed side
//         as compared to the generated size!
// Note2: the events which are generated but not reconstructed
//         have to be added to the appropriate overflow bins of A
//
// code fragment (with histograms A and y filled):
//
//      TUnfold unfold(A,TUnfold::kHistMapOutputHoriz);
//      Double_t tau=1.E-4;
//      Double_t biasScale=0.0;
//      unfold.DoUnfold(tau,y,biasScale);
//      TH1D *x=unfold.GetOutput("x","myVariable");
//      TH2D *rhoij=unfold.GetRhoIJ("correlation","myVariable");
//
// will create histograms "x" and "correlation" from A and y.
// if tau is very large, the output is biased to the generated distribution
// if tau is very small, the output will show oscillations
// and large entries in the correlation matrix
//
//
//
// Proper choice of tau
// ====================
// One of the most difficult questions is about the choice of tau. The most
// common method is the L-curve method: a two-dimensional curve is plotted
//   x-axis: log10(chisquare)
//   y-axis: log10(regularisation condition)
// In many cases this curve has an L-shape. The best choice of tau is in the
// kink of the L
//
// Within TUnfold a simple version of the L-curve analysis is included.
// It tests a given number of points in a predefined tau-range and searches
// for the maximum of the curvature in the L-curve (kink position). 
//
// Example: scan tau and produce the L-curve plot
//
// Code fragment: assume A and y are filled
//
//      TUnfold unfold(A,TUnfold::kHistMapOutputHoriz);
//
//      unfold.SetInput(y);
//
//      Int_t nScan=30;
//      Double_t tauMin=1.E-10;
//      Double_t tauMax=1.0;
//      Int_t iBest;
//      TSpline *logTauX,*logTauY;
//      TGraph *lCurve;
//
//      iBest=unfold.ScanLcurve(nScan,tauMin,tauMax,&lCurve);
//
//      std::cout<<"tau="<<unfold.GetTau()<<"\n";
//
//      TH1D *x=unfold.GetOutput("x","myVariable");
//      TH2D *rhoij=unfold.GetRhoIJ("correlation","myVariable");
//
// This creates
//    logTauX: the L-curve's x-coordinate as a function of log(tau) 
//    logTauY: the L-curve's y-coordinate as a function of log(tau) 
//    lCurve: a graph of the L-curve
//    x,rhoij: unfolding result for best choice of tau
//    iBest: the coordinate/spline knot number with the best choice of tau 
//
// Note: always check the L curve after unfolding. The algorithm is not
//    perfect
//
//
// Bin averaging of the output
// ===========================
// Sometimes it is useful to unfold for a fine binning in x and
// calculate the final result with a smaller number of bins. The advantage
// is a reduction in the correlation coefficients if bins are averaged.
// For this type of averaging the full error matrix has to be used.
// There are methods in TUnfold to support this type of calculation
// Example:
//    The vector x has dimension 49, it consists of 7x7 bins
//      in two variables (Pt,Eta)
//    The unfolding result is to be presented as one-dimensional projections
//    in (Pt) and (Eta)
//    The bins of x are mapped as: bins 1..7 the first Eta bin
//                                 bins 2..14 the second Eta bin
//                                 ...
//                                 bins 1,8,15,... the first Pt bin
//                                 ...
// code fragment:
//
//      TUnfold unfold(A,TUnfold::kHistMapOutputHoriz);
//      Double_t tau=1.E-4;
//      Double_t biasScale=0.0;
//      unfold.DoUnfold(tau,y,biasScale);
//      Int_t binMapEta[49+2];
//      Int_t binMapPt[49+2];
//      // overflow and underflow bins are not used
//      binMapEta[0]=-1;
//      binMapEta[49+1]=-1;
//      binMapPt[0]=-1;
//      binMapPt[49+1]=-1;
//      for(Int_t i=1;i<=49;i++) {
//         // all bins (i) with the same (i-1)/7 are added
//         binMapEta[i] = (i-1)/7 +1;
//         // all bins (i) with the same (i-1)%7 are added
//         binMapPt[i]  = (i-1)%7 +1;
//      }
//      TH1D *etaHist=new TH1D("eta(unfolded)",";eta",7,etamin,etamax);
//      TH1D *etaCorr=new TH2D("eta(unfolded)",";eta;eta",7,etamin,etamax,7,etamin,etamax);
//      TH1D *ptHist=new TH1D("pt(unfolded)",";pt",7,ptmin,ptmax);
//      TH1D *ptCorr=new TH2D("pt(unfolded)",";pt;pt",7,ptmin,ptmax,7,ptmin,ptmax);
//      unfold.GetOutput(etaHist,binMapEta);
//      unfold.GetRhoIJ(etaCorrt,binMapEta);
//      unfold.GetOutput(ptHist,binMapPt);
//      unfold.GetRhoIJ(ptCorrt,binMapPt);
//
//
//
// Alternative Regularisation conditions
// =====================================
// Regularisation is needed for most unfolding problems, in order to avoid
// large oscillations and large correlations on the output bins.
// It means that some extra conditions are applied on the output bins
//
// Within TUnfold these conditions are posed on the difference (x-x0), where
//    x:  unfolding output
//    x0: the bias distribution, by default calculated from
//        the input matrix A. There is a method SetBias() to change the
//        bias distribution. 
//        The 3rd argument to DoUnfold() is a scale factor applied to the bias
//          bias_default[j] = sum_i A[i][j]
//          x0[j] = scaleBias*bias[j]
//        The scale factor can be used to
//         (a) completely suppress the bias by setting it to zero
//         (b) compensate differences in the normalisation between data
//             and Monte Carlo
//
// If the regularisation is strong, i.e. large parameter tau,
// then the distribution x or its derivatives will look like the bias 
// distribution. If the parameter tau is small, the distribution x is 
// independent of the bias.
//
// Three basic types of regularisation are implemented in TUnfold
//
//    condition            regularisation
//  ------------------------------------------------------
//    kRegModeNone         none
//    kRegModeSize         minimize the size of (x-x0)
//    kRegModeDerivative   minimize the 1st derivative of (x-x0)
//    kRegModeCurvature    minimize the 2nd derivative of (x-x0)
//
// kRegModeSize is the regularisation scheme which usually is found in
// literature. In addition, the bias usually is not present
// (bias scale factor is zero).
//
// The non-standard regularisation schemes kRegModeDerivative and 
// kRegModeCurvature have the nice feature that they create correlations
// between x-bins, whereas the non-regularized unfolding tends to create 
// negative correlations between bins. For these regularisation schemes the
// parameter tau could be tuned such that the correlations are smallest, 
// as an alternative to the L-curve method.
//
// If kRegModeSize is chosen or if x is a smooth function through all bins,
// the regularisation condition can be set on all bins together by giving
// the appropriate argument in the constructor (see examples above).
//
// If x is composed of independent groups of bins (for example,
// signal and background binning in two variables), it may be necessary to
// set regularisation conditions for the individual groups of bins.
// In this case,  give  kRegModeNone  in the constructor and specify
// the bin grouping with calls to
//          RegularizeBins()   specify a 1-dimensional group of bins
//          RegularizeBins2D() specify a 2-dimensional group of bins
//
// For ultimate flexibility, the regularisation condition can be set on each
// bin individually
//  -> give  kRegModeNone  in the constructor and use
//      RegularizeSize()        regularize one bin   
//      RegularizeDerivative()  regularize the slope given by two bins
//      RegularizeCurvature()   regularize the curvature given by three bins

#include <iostream>
#include <TMatrixD.h>
#include <TMatrixDSparse.h>
#include <TMath.h>

#include "TUnfold.h"

#include <map>

ClassImp(TUnfold)
//______________________________________________________________________________
void TUnfold::ClearTUnfold(void)
{
   // reset all data members
   fXToHist.Set(0);
   fHistToX.Set(0);
   fA = 0;
   fLsquared = 0;
   fV = 0;
   fY = 0;
   fX0 = 0;
   fTau = 0.0;
   fBiasScale = 0.0;
   // output
   fEinv = 0;
   fE = 0;
   fX = 0;
   fAx = 0;
   fChi2A = 0.0;
   fChi2L = 0.0;
   fRhoMax = 999.0;
   fRhoAvg = -1.0;
}

TUnfold::TUnfold(void)
{
   // set all matrix pointers to zero
   ClearTUnfold();
}

Double_t TUnfold::DoUnfold(void)
{
   // main unfolding algorithm. Declared virtual, because other algorithms
   // could be implemented
   //
   // Purpose: unfold y -> x
   // Data members required:
   //     fA:  matrix to relate x and y
   //     fY:  measured data points
   //     fX0: bias on x
   //     fBiasScale: scale factor for fX0
   //     fV:  inverse of covariance matrix for y
   //     fLsquared: regularisation conditions
   //     fTau: regularisation strength
   // Data members modified:
   //     fEinv: inverse of the covariance matrix of x
   //     fE:    covariance matrix of x
   //     fX:    unfolded data points
   //     fAx:   estimate of y from x
   //     fChi2A:  contribution to chi**2 from y-fAx
   //     fChi2L:  contribution to chi**2 from L*(x-x0)
   //     fRhoMax: maximum global correlation coefficient
   //     fRhoAvg: average global correlation coefficient
   // return code:
   //     fRhoMax   if(fRhoMax>=1.0) then the unfolding has failed!

   // delete old results (if any)
   if (fEinv)
      delete fEinv;
   if (fE)
      delete fE;
   if (fX)
      delete fX;
   if (fAx)
      delete fAx;

   //
   // get matrix
   //              T
   //            fA fV  = mAt_V
   //
   TMatrixDSparse mAt_V(TMatrixDSparse(TMatrixDSparse::kTransposed, *fA),
                        TMatrixDSparse::kMult, *fV);
   //
   // get   T
   //     fA fV fY + fTau fBiasScale Lsquared fX0 = rhs
   //
   TMatrixDSparse rhs(mAt_V, TMatrixDSparse::kMult, *fY);
   if (fBiasScale != 0.0) {
      rhs +=
          (fTau * fBiasScale) * TMatrixDSparse(*fLsquared,
                                               TMatrixDSparse::kMult,
                                               *fX0);
   }
   //
   // get matrix
   //              T
   //           (fA fV)fA + fTau*fLsquared  = fEinv
   //
   fEinv = new TMatrixDSparse(mAt_V, TMatrixDSparse::kMult, *fA);
   *fEinv += fTau * (*fLsquared);
   //
   // get matrix
   //             -1
   //        fEinv    = fE
   //
#ifdef MEASURE_TIMING
   struct timespec t0, t1;
   clockid_t type;
   clock_getcpuclockid(0, &type);
   clock_gettime(type, &t0);
#endif
   fE = new TMatrixD(TMatrixD::kInverted, *fEinv);
   //
   // get result
   //        fE rhs  = x
   //
   fX = new TMatrixD(*fE, TMatrixD::kMult, rhs);
   //
   // get result
   //        fA x  =  fAx
   //
   fAx = new TMatrixDSparse(*fA, TMatrixDSparse::kMult, *fX);
   //
   // calculate chi**2 etc
   //
   CalculateChi2Rho();

   return fRhoMax;
}

void TUnfold::CalculateChi2Rho(void)
{
   // Calculate chi**2 and maximum global correlation
   // Data members required:
   //   fY,fAx,fV,fX,fX0,fBiasScale,fTau,fLsquared
   // Data members modified:
   //   fChi2A,fChi2L,fRhoMax,fRhoAvg

   // chi**2 contribution from (y-Ax)V(y-Ax)
   fChi2A = 0.0;
   TMatrixD dy(*fY, TMatrixD::kMinus, *fAx);
   // Vd is made sparse, because it is made from sparse matrix V
   TMatrixDSparse Vd(*fV, TMatrixDSparse::kMult, dy);
   for (int iy = 0; iy < GetNy(); iy++) {
      fChi2A += dy(iy, 0) * Vd(iy, 0);
   }

   // chi**2 contribution from tau(x-s*x0)Lsquared(x-s*x0)
   fChi2L = 0.0;
   TMatrixD dx(*fX, TMatrixD::kMinus, fBiasScale * (*fX0));
   // LsquaredDx is made sparse, because it is made from sparse matrix
   // fLsquared
   TMatrixDSparse LsquaredDx(*fLsquared, TMatrixDSparse::kMult, dx);
   for (int ix = 0; ix < GetNx(); ix++) {
      fChi2L += fTau * dx(ix, 0) * LsquaredDx(ix, 0);
   }

   // maximum global correlation coefficient
   Double_t rho_squared_max = 0.0;
   Double_t rho_sum = 0.0;
   Int_t n_rho=0;
   for (int ix = 0; ix < GetNx(); ix++) {
      Double_t rho_squared =
          1. - 1. / ((*fE) (ix, ix) * (*fEinv) (ix, ix));
      if (rho_squared > rho_squared_max)
         rho_squared_max = rho_squared;
      if(rho_squared>0.0) {
        rho_sum += TMath::Sqrt(rho_squared);
        n_rho++;
      }
   }
   fRhoMax = TMath::Sqrt(rho_squared_max);
   fRhoAvg = (n_rho>0) ? (rho_sum/n_rho) : -1.0;
}

TMatrixDSparse *TUnfold::CreateSparseMatrix(Int_t nr, Int_t nc,
                                            Int_t * row, Int_t * col,
                                            Double_t const *data)
{
   // Create a sparse matrix, to set up fA or fV.
   //   nr,nc: number of rows and columns
   //   row:   row index array
   //   col:   column index array
   //   data:  non-zero matrix elements
   // Return value:
   //   a new matrix
   //
   // This implementation cirumvents certain problems with TMatrixDSparse
   // constructors. Eventually calls to this method should be replaced
   // by something like
   //    new TMatrixDSparse( some_arguments )
   TMatrixDSparse *m = new TMatrixDSparse(nr, nc);
   m->SetSparseIndex(row[nr]);
   m->SetRowIndexArray(row);
   m->SetColIndexArray(col);
   m->SetMatrixArray(data);
   return m;
}

TUnfold::TUnfold(TH2 const *hist_A, EHistMap histmap, ERegMode regmode)
{
   // set up unfolding matrix and initial regularisation scheme
   //    hist_A:  matrix that describes the migrations
   //    histmap: mapping of the histogram axes to unfolding output 
   //    regmode: global regularisation mode
   // data members initialized to something different from zero:
   //    fA: filled from hist_A
   //    fX0: filled from hist_A
   //    fLsquared: filled depending on the regularisation scheme
   // Treatment of overflow bins
   //    Bins where the unfolding input (Detector level) is in overflow
   //    are used for the efficiency correction. They have to be filled
   //    properly!
   //    Bins where the unfolding output (Generator level) is in overflow
   //    are treated as a part of the generator level distribution.
   //    I.e. the unfolding output could have non-zero overflow bins if the
   //    input matrix does have such bins.

   ClearTUnfold();
   Int_t nx0, nx, ny;
   if (histmap == kHistMapOutputHoriz) {
     // include overflow bins on the X axis
      nx0 = hist_A->GetNbinsX()+2;
      ny = hist_A->GetNbinsY();
   } else {
     // include overflow bins on the X axis
      nx0 = hist_A->GetNbinsY()+2;
      ny = hist_A->GetNbinsX();
   }
   nx = 0;
   // fNx is expected to be nx0, but the input matrix may be ill-formed
   // -> all columns with zero events have to be removed
   //    (because y does not contain any information on that bin in x)
   TArrayD sum_over_y(nx0);
   fXToHist.Set(nx0);
   fHistToX.Set(nx0);
   TArrayI colCount_A(ny);
   // loop
   //  - calculate bias distribution
   //      sum_over_y
   //  - count those generator bins which can be unfolded
   //      fNx
   //  - histogram bins are added to the lookup-tables
   //      fHistToX[] and fXToHist[]
   //    these convert histogram bins to matrix indices and vice versa
   //  - the number of columns per row of the final matrix is counted
   //      colCount_A
   for (Int_t ix = 0; ix < nx0; ix++) {
      // calculate sum over all detector bins
      // excluding the overflow bins
      Double_t sum = 0.0;
      for (Int_t iy = 0; iy < ny; iy++) {
         Double_t z;
         if (histmap == kHistMapOutputHoriz) {
            z = hist_A->GetBinContent(ix, iy + 1);
         } else {
            z = hist_A->GetBinContent(iy + 1, ix);
         }
         if (z > 0.0) {
            colCount_A[iy]++;
            sum += z;
         }
      }
      // check whether there is any sensitivity to this generator bin
      if (sum > 0.0) {
         // update mapping tables to convert bin number to matrix index
         fXToHist[nx] = ix;
         fHistToX[ix] = nx;
         // add overflow bins -> important to keep track of the
         // non-reconstructed events
         sum_over_y[nx] = sum;
         if (histmap == kHistMapOutputHoriz) {
            sum_over_y[nx] +=
                hist_A->GetBinContent(ix, 0) +
                hist_A->GetBinContent(ix, ny + 1);
         } else {
            sum_over_y[nx] += hist_A->GetBinContent(0, ix);
            hist_A->GetBinContent(ny + 1, ix);
         }
         nx++;
      } else {
         // histogram bin pointing to -1 (non-existing matrix entry)
         fHistToX[ix] = -1;
         std::cout << "TUnfold: bin " << ix
             << " has no influence on the data -> skipped!\n";
      }
   }
   // store bias as matrix
   fX0 = new TMatrixD(nx, 1, sum_over_y.GetArray());
   // store non-zero elements in sparse matrix fA
   // construct sparse matrix fA
   // row structure
   Int_t *row_A = new Int_t[ny + 1];
   row_A[0] = 0;
   for (Int_t iy = 0; iy < ny; iy++) {
      row_A[iy + 1] = row_A[iy] + colCount_A[iy];
   }
   // column structure and data points
   Int_t *col_A= new Int_t[row_A[ny]];
   Double_t *data_A=new Double_t[row_A[ny]];
   for (Int_t iy = 0; iy < ny; iy++) {
      Int_t index = row_A[iy];
      for (Int_t ix = 0; ix < nx; ix++) {
         Int_t ibinx = fXToHist[ix];
         Double_t z;
         if (histmap == kHistMapOutputHoriz) {
            z = hist_A->GetBinContent(ibinx, iy + 1);
         } else {
            z = hist_A->GetBinContent(iy + 1, ibinx);
         }
         if (z > 0.0) {
            col_A[index] = ix;
            data_A[index] = z / sum_over_y[ix];
            index++;
         }
      }
   }
   fA = CreateSparseMatrix(ny, nx, row_A, col_A, data_A);
   delete[] row_A;
   delete[] col_A;
   delete[] data_A;
   // regularisation conditions squared
   fLsquared = new TMatrixDSparse(GetNx(), GetNx());
   if (regmode != kRegModeNone) {
      RegularizeBins(0, 1, nx0, regmode);
   }
}

TUnfold::~TUnfold(void)
{
   // delete all data members

   if (fA)
      delete fA;
   if (fLsquared)
      delete fLsquared;
   if (fV)
      delete fV;
   if (fY)
      delete fY;
   if (fX0)
      delete fX0;
   if (fEinv)
      delete fEinv;
   if (fE)
      delete fE;
   if (fX)
      delete fX;
   if (fAx)
      delete fAx;
}

void TUnfold::SetBias(TH1 const *bias)
{
   // initialize alternative bias from histogram
   // modifies data member fX0

   if (fX0)
      delete fX0;
   fX0 = new TMatrixD(GetNx(), 1);
   for (Int_t i = 0; i < GetNx(); i++) {
      (*fX0) (i, 0) = bias->GetBinContent(fXToHist[i]);
   }
}

void TUnfold::RegularizeSize(int bin, Double_t const &scale)
{
   // add regularisation on the size of bin i
   //    bin: bin number
   //    scale: size of the regularisation
   // modifies data member fLsquared

   Int_t i = fHistToX[bin];
   if (i < 0) {
      std::cout << "TUnfold::RegularizeSize skip bin " << bin << "\n";
      return;
   }
   (*fLsquared) (i, i) = scale * scale;
}

void TUnfold::RegularizeDerivative(int left_bin, int right_bin,
                                   Double_t const &scale)
{
   // add regularisation on the difference of two bins
   //   left_bin: 1st bin
   //   right_bin: 2nd bin
   //   scale: size of the regularisation
   // modifies data member fLsquared
   Int_t il = fHistToX[left_bin];
   Int_t ir = fHistToX[right_bin];
   if ((il < 0) || (ir < 0)) {
      std::cout << "TUnfold::RegularizeDerivative skip bins "
          << left_bin << "," << right_bin << "\n";
      return;
   }
   Double_t scale_squared = scale * scale;
   (*fLsquared) (il, il) += scale_squared;
   (*fLsquared) (il, ir) -= scale_squared;
   (*fLsquared) (ir, il) -= scale_squared;
   (*fLsquared) (ir, ir) += scale_squared;
}

void TUnfold::RegularizeCurvature(int left_bin, int center_bin,
                                  int right_bin,
                                  Double_t const &scale_left,
                                  Double_t const &scale_right)
{
   // add regularisation on the curvature through 3 bins (2nd derivative)
   //   left_bin: 1st bin
   //   center_bin: 2nd bin
   //   right_bin: 3rd bin
   //   scale_left: scale factor on center-left difference
   //   scale_right: scale factor on right-center difference
   // modifies data member fLsquared

   Int_t il, ic, ir;
   il = fHistToX[left_bin];
   ic = fHistToX[center_bin];
   ir = fHistToX[right_bin];
   if ((il < 0) || (ic < 0) || (ir < 0)) {
      std::cout << "TUnfold::RegularizeCurvature skip bins "
          << left_bin << "," << center_bin << "," << right_bin << "\n";
      return;
   }
   Double_t r[3];
   r[0] = -scale_left;
   r[2] = -scale_right;
   r[1] = -r[0] - r[2];
   // diagonal elements
   (*fLsquared) (il, il) += r[0] * r[0];
   (*fLsquared) (il, ic) += r[0] * r[1];
   (*fLsquared) (il, ir) += r[0] * r[2];
   (*fLsquared) (ic, il) += r[1] * r[0];
   (*fLsquared) (ic, ic) += r[1] * r[1];
   (*fLsquared) (ic, ir) += r[1] * r[2];
   (*fLsquared) (ir, il) += r[2] * r[0];
   (*fLsquared) (ir, ic) += r[2] * r[1];
   (*fLsquared) (ir, ir) += r[2] * r[2];
}

void TUnfold::RegularizeBins(int start, int step, int nbin,
                             ERegMode regmode)
{
   // set regulatisation on a 1-dimensional curve
   //   start: first bin
   //   step:  distance between neighbouring bins
   //   nbin:  total number of bins
   //   regmode:  regularisation mode
   // modifies data member fLsquared

   Int_t i0, i1, i2;
   i0 = start;
   i1 = i0 + step;
   i2 = i1 + step;
   Int_t nSkip = 0;
   if (regmode == kRegModeDerivative)
      nSkip = 1;
   else if (regmode == kRegModeCurvature)
      nSkip = 2;
   for (Int_t i = nSkip; i < nbin; i++) {
      if (regmode == kRegModeSize) {
         RegularizeSize(i0);
      } else if (regmode == kRegModeDerivative) {
         RegularizeDerivative(i0, i1);
      } else if (regmode == kRegModeCurvature) {
         RegularizeCurvature(i0, i1, i2);
      }
      i0 = i1;
      i1 = i2;
      i2 += step;
   }
}

void TUnfold::RegularizeBins2D(int start_bin, int step1, int nbin1,
                               int step2, int nbin2, ERegMode regmode)
{
   // set regularisation on a 2-dimensional grid of bins
   //     start: first bin
   //     step1: distance between bins in 1st direction
   //     nbin1: number of bins in 1st direction
   //     step2: distance between bins in 2nd direction
   //     nbin2: number of bins in 2nd direction
   // modifies data member fLsquared

   for (Int_t i1 = 0; i1 < nbin1; i1++) {
      RegularizeBins(start_bin + step1 * i1, step2, nbin2, regmode);
   }
   for (Int_t i2 = 0; i2 < nbin2; i2++) {
      RegularizeBins(start_bin + step2 * i2, step1, nbin1, regmode);
   }
}

Double_t TUnfold::DoUnfold(Double_t const &tau_reg, TH1 const *input,
                           Double_t const &scaleBias)
{
   // Do unfolding of an input histogram
   //   tau_reg: regularisation parameter
   //   input:   input distribution with errors
   //   scaleBias:  scale factor applied to the bias
   // Data members required:
   //   fA, fX0, fLsquared
   // Data members modified:
   //   those documented in SetInput()
   //   and those documented in DoUnfold(Double_t)
   // Return value:
   //   maximum global correlation coefficient
   //   NOTE!!! return value >=1.0 means error, and the result is junk
   //
   // Overflow bins of the input distribution are ignored!

   SetInput(input,scaleBias);
   return DoUnfold(tau_reg);
}

void TUnfold::SetInput(TH1 const *input, Double_t const &scaleBias) {
  // Define the input data for subsequent calls to DoUnfold(Double_t)
  //   input:   input distribution with errors
  //   scaleBias:  scale factor applied to the bias
  // Data members modified:
  //   fY, fV, fBiasScale
  fBiasScale = scaleBias;
  // construct inverted error matrix of measured quantities
  // from errors of input histogram
  Int_t *row_V=new Int_t[GetNy() + 1];
  Int_t *col_V=new Int_t[GetNy()];
  Double_t *data_V=new Double_t[GetNy()];
  // fV is set up in increasing row/column order
  // to avoid too much memory management
  for (Int_t iy = 0; iy < GetNy(); iy++) {
    Double_t dy = input->GetBinError(iy + 1);
    if (dy <= 0.0)
      dy = 1.0;
    row_V[iy] = iy;
    col_V[iy] = iy;
    data_V[iy] = 1 / dy / dy;
  }
  row_V[GetNy()] = GetNy();
  if (fV)
    delete fV;
  fV = CreateSparseMatrix(GetNy(), GetNy(), row_V, col_V, data_V);
  delete[] row_V;
  delete[] col_V;
  delete[] data_V;
  //
  // get input vector
  if (fY)
    delete fY;
  fY = new TMatrixD(GetNy(), 1);
  for (Int_t i = 0; i < GetNy(); i++) {
    (*fY) (i, 0) = input->GetBinContent(i + 1);
  }  
}

Double_t TUnfold::DoUnfold(Double_t const &tau) {
  // Unfold with given value of regularisation parameter tau
  //     tau: new tau parameter
  // required data members:
  //     fA:  matrix to relate x and y
  //     fY:  measured data points
  //     fX0: bias on x
  //     fBiasScale: scale factor for fX0
  //     fV:  inverse of covariance matrix for y
  //     fLsquared: regularisation conditions
  // modified data members:
  //     fTau and those documented in DoUnfold(void)
  fTau=tau;
  return DoUnfold();
}

Int_t TUnfold::ScanLcurve(Int_t nPoint,
                          Double_t const &tauMin,Double_t const &tauMax,
			  TGraph **lCurve,TSpline **logTauX,
			  TSpline **logTauY) {
  // scan the L curve
  //   nPoint: number of points to scan
  //   tauMin: smallest tau value to study
  //   tauMax: largest tau value to study
  //   lCurve: the L curve as graph
  //   logTauX: output spline of x-coordinates vs tau for the L curve
  //   logTauY: output spline of y-coordinates vs tau for the L curve
  // return value: the coordinate number (0..nPoint-1) with the "best" choice
  //     of tau
  typedef std::map<Double_t,std::pair<Double_t,Double_t> > XYtau_t;

  Int_t bestChoice=-1;
  XYtau_t curve;
  Double_t logTauMin=-10.;
  Double_t logTauMax=0.0;
  Double_t logTau=logTauMin;
  if(tauMin>0.0) logTauMin=TMath::Log10(tauMin);
  if(tauMax>0.0) logTauMax=TMath::Log10(tauMax);
  if(logTauMax<=logTauMin) logTauMax=logTauMin+10.;
  if(nPoint>0) {
    if(nPoint>1) {
      // initialisation for two or more points
      DoUnfold(TMath::Power(10.,logTauMax));
      curve[logTauMax]=std::make_pair(GetLcurveX(),GetLcurveY());
    }
    // initialisation for one or more points (if nPoint<3 tau is set to tauMin)
    DoUnfold(TMath::Power(10.,logTauMin));
    curve[logTauMin]=std::make_pair(GetLcurveX(),GetLcurveY());
  }

  while(int(curve.size())<nPoint-1) {
    // insert additional points, such that the sizes of the delta(XY) vectors
    // are getting smaller and smaller
    XYtau_t::const_iterator i0,i1;
    i0=curve.begin();
    i1=i0;
    Double_t distMax=0.0;
    for(i1++;i1!=curve.end();i1++) {
      std::pair<Double_t,Double_t> const &xy0=(*i0).second;
      std::pair<Double_t,Double_t> const &xy1=(*i1).second;
      Double_t dx=xy1.first-xy0.first;
      Double_t dy=xy1.second-xy0.second;
      Double_t d=TMath::Sqrt(dx*dx+dy*dy);
      if(d>=distMax) {
        distMax=d;
        logTau=0.5*((*i0).first+(*i1).first);
      }
      i0=i1;
    }
    DoUnfold(TMath::Power(10.,logTau));
    curve[logTau]=std::make_pair(GetLcurveX(),GetLcurveY());
  }
  XYtau_t::const_iterator i0,i1;
  i0=curve.begin();
  i1=i0;
  i1++;
  if(curve.size()>2) {
    // set up splines and determine (x,y) curvature in each point
    Double_t *cTi=new Double_t[curve.size()-1];
    Double_t *cCi=new Double_t[curve.size()-1];
    Int_t n=0;
    {
      Double_t *lXi=new Double_t[curve.size()];
      Double_t *lYi=new Double_t[curve.size()];
      Double_t *lTi=new Double_t[curve.size()];
      for( XYtau_t::const_iterator i=curve.begin();i!=curve.end();i++) {
        lXi[n]=(*i).second.first;
        lYi[n]=(*i).second.second;
        lTi[n]=(*i).first;
        n++;
      }
      TSpline3 *splineX=new TSpline3("x vs tau",lTi,lXi,n);
      TSpline3 *splineY=new TSpline3("y vs tau",lTi,lYi,n);
      // calculate (x,y) curvature for all points
      // the curvature is stored in the array cCi[] as a function of cTi[] 
      for(Int_t i=0;i<n-1;i++) {
        Double_t ltau,xy,bi,ci,di;
        splineX->GetCoeff(i,ltau,xy,bi,ci,di);
        Double_t tauBar=0.5*(lTi[i]+lTi[i+1]);
        Double_t dTau=0.5*(lTi[i+1]-lTi[i]);
        Double_t dx1=bi+dTau*(2.*ci+3.*di*dTau);
        Double_t dx2=2.*ci+6.*di*dTau;
        splineY->GetCoeff(i,ltau,xy,bi,ci,di);
        Double_t dy1=bi+dTau*(2.*ci+3.*di*dTau);
        Double_t dy2=2.*ci+6.*di*dTau;
        cTi[i]=tauBar;
        cCi[i]=(dy2*dx1-dy1*dx2)/TMath::Power(dx1*dx1+dy1*dy1,1.5);
      }
      delete splineX;
      delete splineY;
      delete[] lXi;
      delete[] lYi;
      delete[] lTi;
    }
    // create curvature Spline
    TSpline3 *splineC=new TSpline3("L curve curvature",cTi,cCi,n-1);
    Double_t cCmax=cCi[0];
    Double_t cTmax=cTi[0];
    // find the maximum of the curvature
    for(Int_t i=0;i<n-2;i++) {
      // find maximum on this spline section
      // check boundary conditions for x[i] and x[i+1]
      Double_t xMax=cTi[i+1];
      Double_t yMax=cCi[i+1];
      if(cCi[i]>yMax) {
        yMax=cCi[i];
        xMax=cTi[i];
      }
      // find maximum for x[i]<x<x[i+1]
      // get spline coefficiencts and solve equation
      //   derivative(x)==0
      Double_t x,y,b,c,d;
      splineC->GetCoeff(i,x,y,b,c,d);
      // coefficiencts of quadratic equation
      Double_t m_p_half=-c/(3.*d);
      Double_t q=b/(3.*d);
      Double_t discr=m_p_half*m_p_half-q;
      if(discr>=0.0) {
        // solution found
        discr=TMath::Sqrt(discr);
        Double_t xx;
        if(m_p_half>0.0) {
          xx = m_p_half + discr;
        } else {
          xx = m_p_half - discr;
        }
        Double_t dx=cTi[i+1]-x;
        // check first solution
        if((xx>0.0)&&(xx<dx)) {
          y=splineC->Eval(x+xx);
          if(y>yMax) {
            yMax=y;
            xMax=x+xx;
          }
        }
        // second solution
        if(xx !=0.0) {
          xx= q/xx;
        } else {
          xx=0.0;
        }
        // check second solution
        if((xx>0.0)&&(xx<dx)) {
          y=splineC->Eval(x+xx);
          if(y>yMax) {
            yMax=y;
            xMax=x+xx;
          }
        }
      }
      // check whether this local minimum is a global minimum
      if(yMax>cCmax) {
        cCmax=yMax;
        cTmax=xMax;
      }
    }
    delete splineC;
    delete[] cTi;
    delete[] cCi;
    logTau=cTmax;
    DoUnfold(TMath::Power(10.,logTau));
    curve[logTau]=std::make_pair(GetLcurveX(),GetLcurveY());
  }
  //
  // identify position of the result and save it as splines
  if(curve.size()>0) {
    Double_t *x=new Double_t[curve.size()];
    Double_t *y=new Double_t[curve.size()];
    Double_t *logT=new Double_t[curve.size()];
    int n=0;
    for( XYtau_t::const_iterator i=curve.begin();i!=curve.end();i++) {
      if(logTau==(*i).first) {
        bestChoice=n;
      }
      x[n]=(*i).second.first;
      y[n]=(*i).second.second;
      logT[n]=(*i).first;
      n++;
    }
    if(lCurve) (*lCurve)=new TGraph(n,x,y);
    if(logTauX) (*logTauX)=new TSpline3("log(chi**2)%log(tau)",logT,x,n);
    if(logTauY) (*logTauY)=new TSpline3("L curve",logT,y,n);
    delete[] x;
    delete[] y;
    delete[] logT;
  }

  return bestChoice;
}


TH1D *TUnfold::GetOutput(char const *name, char const *title,
                         Double_t xmin, Double_t xmax) const
{
   // retreive unfolding result as histogram
   //   name:  name of the histogram
   //   title: title of the histogram
   //   x0,x1: lower/upper edge of histogram.
   //          if (x0>=x1) then x0=0 and x1=nbin are used

   int nbins = fHistToX.GetSize() - 2;
   if (xmin >= xmax) {
      xmin = 0.0;
      xmax = nbins;
   }
   TH1D *out = new TH1D(name, title, nbins, xmin, xmax);
#ifdef UNUSED
   for (Int_t i = 0; i < GetNx(); i++) {
      out->SetBinContent(fXToHist[i], (*fX) (i, 0));
      out->SetBinError(fXToHist[i], TMath::Sqrt((*fE) (i, i)));
   }
#else
   GetOutput(out);
#endif

   return out;
}

TH1D *TUnfold::GetBias(char const *name, char const *title,
                       Double_t xmin, Double_t xmax) const
{
   // retreive bias as histogram
   //   name:  name of the histogram
   //   title: title of the histogram
   //   x0,x1: lower/upper edge of histogram.
   //          if (x0>=x1) then x0=0 and x1=nbin are used

   int nbins = fHistToX.GetSize() - 2;
   if (xmin >= xmax) {
      xmin = 0.0;
      xmax = nbins;
   }
   TH1D *out = new TH1D(name, title, nbins, xmin, xmax);
   for (Int_t i = 0; i < GetNx(); i++) {
      out->SetBinContent(fXToHist[i], (*fX0) (i, 0));
   }
   return out;
}

TH1D *TUnfold::GetFoldedOutput(char const *name, char const *title,
                               Double_t y0, Double_t y1) const
{
   // retreive unfolding result folded back by the matrix
   //   name:  name of the histogram
   //   title: title of the histogram
   //   y0,y1: lower/upper edge of histogram.
   //          if (y0>=y1) then y0=0 and y1=nbin are used

   if (y0 >= y1) {
      y0 = 0.0;
      y1 = GetNy();
   }
   TH1D *out = new TH1D(name, title, GetNy(), y0, y1);
   Int_t const *rows=fA->GetRowIndexArray();
   Int_t const *cols=fA->GetColIndexArray();
   Double_t const *data=fA->GetMatrixArray();
   for (Int_t i = 0; i < GetNy(); i++) {
      out->SetBinContent(i + 1, (*fAx) (i, 0));
      Double_t e2=0.0;
      for( Int_t cindex1=rows[i];cindex1<rows[i+1];cindex1++) {
        for( Int_t cindex2=rows[i];cindex2<rows[i+1];cindex2++) {
          e2 += data[cindex1]*(*fE)(cols[cindex1],cols[cindex2])*data[cindex2];
        }
      }
      out->SetBinError(i + 1,TMath::Sqrt(e2));
   }

   return out;
}

TH1D *TUnfold::GetInput(char const *name, char const *title,
                        Double_t y0, Double_t y1) const
{
   // retreive input distribution
   //   name:  name of the histogram
   //   title: title of the histogram
   //   y0,y1: lower/upper edge of histogram.
   //          if (y0>=y1) then y0=0 and y1=nbin are used

   if (y0 >= y1) {
      y0 = 0.0;
      y1 = GetNy();
   }
   TH1D *out = new TH1D(name, title, GetNy(), y0, y1);
   TMatrixD Vinv(TMatrixD::kInverted, *fV);
   for (Int_t i = 0; i < GetNy(); i++) {
      out->SetBinContent(i + 1, (*fY) (i, 0));
      out->SetBinError(i + 1, TMath::Sqrt(Vinv(i, i)));
   }

   return out;
}

TH2D *TUnfold::GetRhoIJ(char const *name, char const *title,
                        Double_t xmin, Double_t xmax) const
{
   // retreive full matrix of correlation coefficients
   //   name:  name of the histogram
   //   title: title of the histogram
   //   x0,x1: lower/upper edge of histogram.
   //          if (x0>=x1) then x0=0 and x1=nbin are used

   int nbins = fHistToX.GetSize() - 2;
   if (xmin >= xmax) {
      xmin = 0.0;
      xmax = nbins;
   }
   TH2D *out = new TH2D(name, title, nbins, xmin, xmax, nbins, xmin, xmax);
#ifdef UNUSED
   for (Int_t i = 0; i < GetNx(); i++) {
      for (Int_t j = 0; j < GetNx(); j++) {
         out->SetBinContent(fXToHist[i], fXToHist[j],
                            (*fE) (i,
                                   j) / TMath::Sqrt((*fE) (i,
                                                           i) * (*fE) (j,
                                                                       j)));
      }
   }
#else
   GetRhoIJ(out);
#endif
   return out;
}

TH2D *TUnfold::GetEmatrix(char const *name, char const *title,
                          Double_t xmin, Double_t xmax) const
{
   // retreive full error matrix
   //   name:  name of the histogram
   //   title: title of the histogram
   //   x0,x1: lower/upper edge of histogram.
   //          if (x0>=x1) then x0=0 and x1=nbin are used

   int nbins = fHistToX.GetSize() - 2;
   if (xmin >= xmax) {
      xmin = 0.0;
      xmax = nbins;
   }
   TH2D *out = new TH2D(name, title, nbins, xmin, xmax, nbins, xmin, xmax);
#ifdef UNUSED
   for (Int_t i = 0; i < GetNx(); i++) {
      for (Int_t j = 0; j < GetNx(); j++) {
         out->SetBinContent(fXToHist[i], fXToHist[j], (*fE) (i, j));
      }
   }
#else
   GetEmatrix(out);
#endif

   return out;
}

TH1D *TUnfold::GetRhoI(char const *name, char const *title,
                       Double_t xmin, Double_t xmax) const
{
   // retreive matrix of global correlation coefficients
   //   name:  name of the histogram
   //   title: title of the histogram
   //   x0,x1: lower/upper edge of histogram.
   //          if (x0>=x1) then x0=0 and x1=nbin are used

   int nbins = fHistToX.GetSize() - 2;
   if (xmin >= xmax) {
      xmin = 0.0;
      xmax = nbins;
   }
   TH1D *out = new TH1D(name, title, nbins, xmin, xmax);
#ifdef RHOI_FAST
   for (Int_t i = 0; i < GetNx(); i++) {
     // the access (*fEinv) (i, i) is safe, because the diagonal
     // element always is non-zero
      Double_t rho_squared = 1. - 1. / ((*fE) (i, i) * (*fEinv) (i, i));
      Double_t rho;
      if (rho_squared >= 0.0)
         rho = TMath::Sqrt(rho_squared);
      else
         rho = -TMath::Sqrt(-rho_squared);
      out->SetBinContent(fXToHist[i], rho);
   }
#else
   GetRhoI(out);
#endif

   return out;
}

TH2D *TUnfold::GetLsquared(char const *name, char const *title,
                           Double_t xmin, Double_t xmax) const
{
   // retreive ix of regularisation conditions squared
   //   name:  name of the histogram
   //   title: title of the histogram
   //   x0,x1: lower/upper edge of histogram.
   //            if (x0>=x1) then x0=0 and x1=nbin are used

   int nbins = fHistToX.GetSize() - 2;
   if (xmin >= xmax) {
      xmin = 0.0;
      xmax = nbins;
   }
   TH2D *out = new TH2D(name, title, nbins, xmin, xmax, nbins, xmin, xmax);
   out->SetOption("BOX");
   // loop over sparse matrix
   Int_t const *rows=fLsquared->GetRowIndexArray();
   Int_t const *cols=fLsquared->GetColIndexArray();
   Double_t const *data=fLsquared->GetMatrixArray();
   for (Int_t i = 0; i < GetNx(); i++) {
      for (Int_t cindex = rows[i]; cindex < rows[i+1]; cindex++) {
        Int_t j=cols[cindex];
        out->SetBinContent(fXToHist[i], fXToHist[j], fTau * data[cindex]);
      }
   }

  return out;
}

Double_t const &TUnfold::GetTau(void) const
{
   // return regularisation parameter
   return fTau;
}

Double_t const &TUnfold::GetRhoMax(void) const
{
   // return maximum global correlation
   // Note: return value>1.0 means the unfolding has failed
   return fRhoMax;
}

Double_t const &TUnfold::GetRhoAvg(void) const
{
   // return average global correlation
   return fRhoAvg;
}

Double_t const &TUnfold::GetChi2A(void) const
{
   // return chi**2 contribution from matrix A
   return fChi2A;
}

Double_t const &TUnfold::GetChi2L(void) const
{
   // return chi**2 contribution from regularisation conditions
   return fChi2L;
}

Double_t TUnfold::GetLcurveX(void) const {
  // return value on x axis of L curve
  return TMath::Log10(fChi2A+fChi2L);
}

Double_t TUnfold::GetLcurveY(void) const {
  // return value on y axis of L curve
  return TMath::Log10(fChi2L/fTau);
}

void TUnfold::GetOutput(TH1 *output,Int_t const *binMap) const {
   // get output distribution, cumulated over several bins
   //   output: output histogram
   //   binMap: for each bin of the original output distribution
   //           specify the destination bin. A value of -1 means that the bin
   //           is discarded. 0 means underflow bin, 1 first bin, ...
   //        binMap[0] : destination of underflow bin
   //        binMap[1] : destination of first bin
   //          ...

  Int_t nbin=output->GetNbinsX();
  Double_t *c=new Double_t[nbin+2];
  Double_t *e2=new Double_t[nbin+2];
  for(Int_t i=0;i<nbin+2;i++) {
    c[i]=0.0;
    e2[i]=0.0;
  }

  Int_t binMapSize = fHistToX.GetSize();
  for(Int_t i=0;i<binMapSize;i++) {
    Int_t destBinI=binMap ? binMap[i] : i;
    Int_t srcBinI=fHistToX[i];
    if((destBinI>=0)&&(destBinI<nbin+2)&&(srcBinI>=0)) {
      c[destBinI]+=(*fX)(srcBinI,0);
      for(Int_t j=0;j<binMapSize;j++) {
        Int_t destBinJ=binMap ? binMap[j] : j;
        if(destBinI!=destBinJ) continue;
        Int_t srcBinJ=fHistToX[j];
        if(srcBinJ>=0) e2[destBinI]+= (*fE)(srcBinI,srcBinJ);
      }
    }
  }
  for(Int_t i=0;i<nbin+2;i++) {
    output->SetBinContent(i,c[i]);
    output->SetBinError(i,TMath::Sqrt(e2[i]));
  }
  delete[] c;
  delete[] e2;
}

void TUnfold::GetEmatrix(TH2 *ematrix,Int_t const *binMap) const {
   // get output error matrix, cumulated over several bins
   //   ematrix: error matrix histogram
   //   binMap: for each bin of the original output distribution
   //           specify the destination bin. A value of -1 means that the bin
   //           is discarded. 0 means underflow bin, 1 first bin, ...
   //        binMap[0] : destination of underflow bin
   //        binMap[1] : destination of first bin
   //          ...
  Int_t nbin=ematrix->GetNbinsX();
  for(Int_t i=0;i<nbin+2;i++) {
    for(Int_t j=0;j<nbin+2;j++) {
      ematrix->SetBinContent(i,j,0.0);
      ematrix->SetBinError(i,j,0.0);
    }
  }

  Int_t binMapSize = fHistToX.GetSize();
  for(Int_t i=0;i<binMapSize;i++) {
    Int_t destBinI=binMap ? binMap[i] : i;
    Int_t srcBinI=fHistToX[i];
    if((destBinI>=0)&&(destBinI<nbin+2)&&(srcBinI>=0)) {
      for(Int_t j=0;j<binMapSize;j++) {
        Int_t destBinJ=binMap ? binMap[j] : j;
        Int_t srcBinJ=fHistToX[j];
        if((destBinJ>=0)&&(destBinJ<nbin+2)&&(srcBinJ>=0)) {
          Double_t e2=ematrix->GetBinContent(destBinI,destBinJ);
          e2 += (*fE)(srcBinI,srcBinJ);
          ematrix->SetBinContent(destBinI,destBinJ,e2);
        }
      }
    }
  }
}

Double_t TUnfold::GetRhoI(TH1 *rhoi,TH2 *ematrixinv,Int_t const *binMap) const {
  // get global correlation coefficients and inverted error matrix,
  // cumulated over several bins
  //   rhoi: global correlation histogram
  //   ematrixinv: inverse of error matrix (if pointer==0 it is not returned)
  //   binMap: for each bin of the original output distribution
  //           specify the destination bin. A value of -1 means that the bin
  //           is discarded. 0 means underflow bin, 1 first bin, ...
  //        binMap[0] : destination of underflow bin
  //        binMap[1] : destination of first bin
  //          ...
  // return value: average global correlation

  Int_t nbin=rhoi->GetNbinsX();  
  // count number of bins mapped into one bin of the output histogram
  Int_t *nz=new Int_t[nbin+2];
  for(Int_t i=0;i<nbin+2;i++) nz[i]=0;
  Int_t binMapSize = fHistToX.GetSize();
  for(Int_t i=0;i<binMapSize;i++) {
    Int_t destBinI=binMap ? binMap[i] : i;
    Int_t srcBinI=fHistToX[i];
    if((destBinI>=0)&&(destBinI<nbin+2)&&(srcBinI>=0)) {
      nz[destBinI]++;
    }
  }
  // count bins which do receive some input
  // and provide lookup-table
  Int_t n=0;
  Int_t *destBin=new Int_t[nbin+2];
  Int_t *matrixBin=new Int_t[nbin+2];
  for(Int_t i=0;i<nbin+2;i++) {
    if(nz[i]>0) {
      matrixBin[i]=n;
      destBin[n]=i;
      n++;
    } else {
      matrixBin[i]=-1;
    }
  }
  // set up reduced error matrix
  TMatrixD e(n,n);
  for(Int_t i=0;i<binMapSize;i++) {
    Int_t destBinI=binMap ? binMap[i] : i;
    Int_t srcBinI=fHistToX[i];
    if((destBinI>=0)&&(destBinI<nbin+2)&&(srcBinI>=0)) {
      Int_t matrixBinI=matrixBin[destBinI];
      for(Int_t j=0;j<binMapSize;j++) {
        Int_t destBinJ=binMap ? binMap[j] : j;
        Int_t srcBinJ=fHistToX[j];
        if((destBinJ>=0)&&(destBinJ<nbin+2)&&(srcBinJ>=0)) {
          Int_t matrixBinJ=matrixBin[destBinJ];
          e(matrixBinI,matrixBinJ) += (*fE)(srcBinI,srcBinJ);
        }
      }
    }
  }
  TMatrixD einv(TMatrixD::kInverted,e);
  Double_t rhoMax=0.0;
  for(Int_t i=0;i<n;i++) {
    Int_t destBinI=destBin[i];
    Double_t rho=1.-1./(einv(i,i)*e(i,i));
    if(rho>=0.0) rho=TMath::Sqrt(rho);
    else rho=-TMath::Sqrt(-rho);
    if(rho>rhoMax) {
      rhoMax = rho;
    }
    rhoi->SetBinContent(destBinI,rho);
    if(ematrixinv) {
      for(Int_t j=0;j<n;j++) {
        ematrixinv->SetBinContent(destBinI,destBin[j],einv(i,j));
      }
    }
  }
  delete[] nz;
  delete[] destBin;
  delete[] matrixBin;
  return rhoMax;
}

void TUnfold::GetRhoIJ(TH2 *rhoij,Int_t const *binMap) const {
  // get correlation coefficient matrix, cumulated over several bins
  //   rhoij:  correlation coefficient matrix histogram
  //   binMap: for each bin of the original output distribution
  //           specify the destination bin. A value of -1 means that the bin
  //           is discarded. 0 means underflow bin, 1 first bin, ...
  //        binMap[0] : destination of underflow bin
  //        binMap[1] : destination of first bin
  //          ...
  GetEmatrix(rhoij,binMap);
  Int_t nbin=rhoij->GetNbinsX();  
  Double_t *e=new Double_t[nbin+2];
  for(Int_t i=0;i<nbin+2;i++) {
    e[i]=TMath::Sqrt(rhoij->GetBinContent(i,i));
  }
  for(Int_t i=0;i<nbin+2;i++) {
    for(Int_t j=0;j<nbin+2;j++) {
      if((e[i]>0.0)&&(e[j]>0.0)) {
        rhoij->SetBinContent(i,j,rhoij->GetBinContent(i,j)/e[i]/e[j]);
      } else {
        rhoij->SetBinContent(i,j,0.0);
      }
    }
  }
  delete[] e;
}
