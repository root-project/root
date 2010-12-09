// @(#)root/hist:$Id$
// Author: Stefan Schmitt
// DESY, 13/10/08

// Version 16, fix calculation of global correlations, improved error messages
//
//  History:
//    Version 15, simplified L-curve scan, new tau definition, new error calc., area preservation
//    Version 14, with changes in TUnfoldSys.cxx
//    Version 13, new methods for derived classes and small bug fix
//    Version 12, report singular matrices
//    Version 11, reduce the amount of printout
//    Version 10, more correct definition of the L curve, update references
//    Version 9, faster matrix inversion and skip edge points for L-curve scan
//    Version 8, replace all TMatrixSparse matrix operations by private code
//    Version 7, fix problem with TMatrixDSparse,TMatrixD multiplication
//    Version 6, replace class XY by std::pair
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
//   chi**2 = (y-Ax)# Vyy^-1 (y-Ax) + tau^2 (L(x-x0))# L(x-x0) + lambda sum_i(y_i -(Ax)_i)
//
// where # means that the matrix is transposed
//
// Monte Carlo input
// -----------------
//   y: vector of measured quantities  (dimension ny)
//   Vyy: covariance matrix for y (dimension ny x ny)
//        in many cases V is diagonal and calculated from the errors of y
//   A: migration matrix               (dimension ny x nx)
//   x: unknown underlying distribution (dimension nx)
//
// Regularisation
// --------------
//   tau: parameter, defining the regularisation strength
//   L: matrix of regularisation conditions (dimension nl x nx)
//   x0: bias distribution
//
// Preservation of the area
// ------------------------
//   lambda: lagrangian multiplier
//   y_i: one component of the vector y
//   (Ax)_i: one component of the vector Ax
//                                                                     
// and chi**2 is minimized
//   (a) not constrained: minimisation is performed a function of x for fixed lambda=0
//  or
//   (b) constrained: minimisation is performed a function of x and lambda
//                                                                      
// This applies to a very large number of problems, where the measured
// distribution y is a linear superposition of several Monte Carlo shapes
// and the sum of these shapes gives the output distribution x
//
// The constraint can be useful to reduce biases on the result x
// in cases where the vector y follows non-Gaussian probability densities
// (example: Poisson statistics at counting experiments in particle physics)
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
// Documentation
// =============
// Some technical documentation is available here:
//   http://www.desy.de/~sschmitt
//
// References:
// ===========
// A nice overview of the method is given in:
//  The L-curve and Its Use in the Numerical Treatment of Inverse Problems
//  (2000) by P. C. Hansen, in Computational Inverse Problems in
//  Electrocardiology, ed. P. Johnston,
//  Advances in Computational Bioengineering
//  http://www.imm.dtu.dk/~pch/TR/Lcurve.ps 
// The relevant equations are (1), (2) for the unfolding
// and (14) for the L-curve curvature definition
//
// Related literature on unfolding:
//  The program package RUN and the web-page by V.Blobel
//    http://www.desy.de/~blobel/unfold.html
//  Talk by V. Blobel, Terascale Statistics school
//    https://indico.desy.de/contributionDisplay.py?contribId=23&confId=1149
//  References quoted in Blobel's talk:
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
//    Lsquared = L#L            regularisation conditions squared
//
//    epsilon_j = sum_i A_ij    vector of efficiencies
//
//    E^-1  = ((A# Vyy^-1 A)+tau^2 Lsquared) 
//
//    x = E (A# Vyy^-1 y + tau^2 Lsquared x0 +lambda/2 * epsilon) is the result
//
// The derivatives
//    dx_k/dy_i
//    dx_k/dA_ij
//    dx_k/d(tau^2)
// are calculated for further usage.
//
// The covariance matrix V_xx is calculated as:
//    Vxx_ij = sum_kl dx_i/dy_k Vyy_kl dx_j/dy_l
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
// Note3: make sure all bins have sufficient statistics and their error is
//         non-zero. By default, bins with zero error are simply skipped;
//         however, this may cause problems if You try to unfold something
//         which depends on these input bins.
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
// if tau is very large, the output is biased to the generated distribution scaled by biasScale
// if tau is very small, the output will show oscillations
// and large entries in the correlation matrix
//
//
// Proper choice of tau
// ====================
// One of the difficult questions is about the choice of tau. The most
// common method is the L-curve method: a two-dimensional curve is plotted
//   x-axis: log10(chisquare)
//   y-axis: log10(regularisation condition)
// In many cases this curve has an L-shape. The best choice of tau is in the
// kink of the L
//
// Within TUnfold a simple version of the L-curve analysis is included.
// It tests a given number of points in a predefined tau-range and searches
// for the maximum of the curvature in the L-curve (kink position). 
// if no tau range is given, the range of teh scan is determied automatically
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
//      Int_t iBest;
//      TSpline *logTauX,*logTauY;
//      TGraph *lCurve;
//
//      iBest=unfold.ScanLcurve(nScan,0.0,0.0,&lCurve);
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
#include <TMatrixDSym.h>
#include <TMath.h>
#include <TDecompBK.h>

#include <TUnfold.h>

#include <map>
#include <vector>


// this option enables the use of Schur complement for matrix inversion
// with large diagonal sub-matrices
#define SCHUR_COMPLEMENT_MATRIX_INVERSION

// this option enables pre-conditioning of matrices prior to inversion
// This type of preconditioning is expected to improve the numerical
// accuracy for the symmetric and positive definite matrices connected
// to unfolding problems. Also, the determinant is more likely to be non-zero
// in case of non-singular matrices
#define INVERT_WITH_CONDITIONING

// this option saves the spline of the L curve curvature to a file 
// named splinec.ps for debugging

//#define DEBUG_LCURVE

#ifdef DEBUG_LCURVE
#include <TCanvas.h>
#endif

ClassImp(TUnfold)
//______________________________________________________________________________

const char *TUnfold::GetTUnfoldVersion(void) {
   return TUnfold_VERSION;
}


void TUnfold::InitTUnfold(void)
{
   // reset all data members
   fXToHist.Set(0);
   fHistToX.Set(0);
   fSumOverY.Set(0);
   fA = 0;
   fLsquared = 0;
   fVyy = 0;
   fY = 0;
   fX0 = 0;
   fTauSquared = 0.0;
   fBiasScale = 0.0;
   fNdf = 0;
   fConstraint = kEConstraintNone;
   fRegMode = kRegModeNone;
   // output
   fVxx = 0;
   fX = 0;
   fAx = 0;
   fChi2A = 0.0;
   fLXsquared = 0.0;
   fRhoMax = 999.0;
   fRhoAvg = -1.0;
   fDXDAM[0] = 0;
   fDXDAZ[0] = 0;
   fDXDAM[1] = 0;
   fDXDAZ[1] = 0;
   fDXDtauSquared = 0;
   fDXDY = 0;
   fEinv = 0;
   fE = 0;
   fVxxInv = 0;
}

void TUnfold::DeleteMatrix(TMatrixD **m) {
   if(*m) delete *m;
   *m=0;
}

void TUnfold::DeleteMatrix(TMatrixDSparse **m) {
   if(*m) delete *m;
   *m=0;
}

void TUnfold::ClearResults(void) {
   // delete old results (if any)
   // this function is virtual, so derived classes may flag their results
   // ad non-valid as well
   DeleteMatrix(&fVxx);
   DeleteMatrix(&fX);
   DeleteMatrix(&fAx);
   for(Int_t i=0;i<2;i++) {
      DeleteMatrix(fDXDAM+i);
      DeleteMatrix(fDXDAZ+i);
   }
   DeleteMatrix(&fDXDtauSquared);
   DeleteMatrix(&fDXDY);
   DeleteMatrix(&fEinv);
   DeleteMatrix(&fE);
   DeleteMatrix(&fVxxInv);
   fChi2A = 0.0;
   fLXsquared = 0.0;
   fRhoMax = 999.0;
   fRhoAvg = -1.0;
}

TUnfold::TUnfold(void)
{
   // set all matrix pointers to zero
   InitTUnfold();
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
   //     fVyy:  covariance matrix for y
   //     fLsquared: regularisation conditions
   //     fTauSquared: regularisation strength
   //     fConstraint: whether the constraint is applied
   // Data members modified:
   //     fEinv: inverse of the covariance matrix of x
   //     fE:    covariance matrix of x
   //     fX:    unfolded data points
   //     fDXDY: derivative of x wrt y (for error propagation)
   //     fVxx:  error matrix (covariance matrix) on x
   //     fAx:   estimate of distribution y from unfolded data
   //     fChi2A:  contribution to chi**2 from y-Ax
   //     fChi2L:  contribution to chi**2 from L*(x-x0)
   //     fDXDtauSquared: derivative of x wrt tau
   //     fDXDAM[0,1]: matrix parts of derivative x wrt A
   //     fDXDAZ[0,1]: vector parts of derivative x wrt A
   //     fRhoMax: maximum global correlation coefficient
   //     fRhoAvg: average global correlation coefficient
   // return code:
   //     fRhoMax   if(fRhoMax>=1.0) then the unfolding has failed!

   ClearResults();

   // get inverse matrix Vyyinv
   //   (1) create mapping for data bins, excluding bins with zero error
   Int_t ny=fVyy->GetNrows();
   const Int_t *vyy_rows=fVyy->GetRowIndexArray();
   const Int_t *vyy_cols=fVyy->GetColIndexArray();
   const Double_t *vyy_data=fVyy->GetMatrixArray();
   Int_t *usedBin=new Int_t[ny];
   for(Int_t i=0;i<ny;i++) {
      usedBin[i]=0;
   }
   for(Int_t i=0;i<ny;i++) {
      for(Int_t k=vyy_rows[i];k<vyy_rows[i+1];k++) {
         if(vyy_data[k]>0.0) {
            usedBin[i]++;
            usedBin[vyy_cols[k]]++;
         }
      }
   }
   Int_t n=0;
   Int_t *yToI=new Int_t[ny];
   Int_t *iToY=new Int_t[ny];
   for(Int_t i=0;i<ny;i++) {
      if(usedBin[i]) {
         yToI[i]=n;
         iToY[n]=i;
         n++;
      } else {
         yToI[i]=-1;
      }
   }
   delete[] usedBin;
   // here:
   //   n: dimension of non-degenerate error matrix
   //   yToI: return non-degenerate bin number (or -1)
   //   iToY: convert non-degenerate bin number to original bin number

   //  (2) create copy of Vyy, with unused bins removed
   Int_t nn=vyy_rows[ny];
   Int_t *vyyred_rows=new Int_t[nn];
   Int_t *vyyred_cols=new Int_t[nn];
   Double_t *vyyred_data=new Double_t[nn];
   nn=0;
   for(Int_t i=0;i<ny;i++) {
      for(Int_t k=vyy_rows[i];k<vyy_rows[i+1];k++) {
         if(vyy_data[k]>0.0) {
            vyyred_rows[nn]=yToI[i];
            vyyred_cols[nn]=yToI[vyy_cols[k]];
            vyyred_data[nn]=vyy_data[k];
            nn++;
         }
      }
   }
   TMatrixDSparse *vyyred=CreateSparseMatrix
      (n,n,nn,vyyred_rows,vyyred_cols,vyyred_data);
   delete[] vyyred_rows;
   delete[] vyyred_cols;
   delete[] vyyred_data;
   //   (3) invert this copy of Vyy
   TMatrixD *vyyredinv=InvertMSparse(vyyred);
   DeleteMatrix(&vyyred);
   //   (4) create sparse inverted matrix
   nn=ny*ny;
   Int_t *vyyinv_rows=new Int_t[nn];
   Int_t *vyyinv_cols=new Int_t[nn];
   Double_t *vyyinv_data=new Double_t[nn];
   nn=0;
   for(Int_t ix=0;ix<n;ix++) {
      for(Int_t iy=0;iy<n;iy++) {
         Double_t d=(*vyyredinv)(ix,iy);
         if(d != 0.0) {
            vyyinv_rows[nn]=iToY[ix];
            vyyinv_cols[nn]=iToY[iy];
            vyyinv_data[nn]=d;
            nn++;
         }
      }
   }
   DeleteMatrix(&vyyredinv);
   TMatrixDSparse *Vyyinv=CreateSparseMatrix
      (ny,ny,nn,vyyinv_rows,vyyinv_cols,vyyinv_data);

   delete[] vyyinv_rows;
   delete[] vyyinv_cols;
   delete[] vyyinv_data;

   delete[] yToI;
   delete[] iToY;

   //
   // get matrix
   //              T
   //            fA fV  = mAt_V
   //
   TMatrixDSparse *AtVyyinv=MultiplyMSparseTranspMSparse(fA,Vyyinv);
   //
   // get
   //       T
   //     fA fVyyinv fY + fTauSquared fBiasScale Lsquared fX0 = rhs
   //
   TMatrixDSparse *rhs=MultiplyMSparseM(AtVyyinv,fY);
   if (fBiasScale != 0.0) {
     TMatrixDSparse *rhs2=MultiplyMSparseM(fLsquared,fX0);
      AddMSparse(rhs, fTauSquared * fBiasScale ,rhs2);
      DeleteMatrix(&rhs2);
   }

   //
   // get matrix
   //              T
   //           (fA fV)fA + fTauSquared*fLsquared  = fEinv
   fEinv=MultiplyMSparseMSparse(AtVyyinv,fA);
   AddMSparse(fEinv,fTauSquared,fLsquared);

   //
   // get matrix
   //             -1
   //        fEinv    = fE
   //
   TMatrixD *EE = InvertMSparse(fEinv);
   fE=new TMatrixDSparse(*EE);

   //
   // get result
   //        fE rhs  = x
   //
   fX = new TMatrixD(*EE, TMatrixD::kMult, *rhs);
   DeleteMatrix(&rhs);

   // additional correction for constraint
   Double_t lambda_half=0.0;
   Double_t one_over_epsEeps=0.0;
   TMatrixDSparse *epsilon=0;
   TMatrixDSparse *Eepsilon=0;
   if(fConstraint != kEConstraintNone) {
      // calculate epsilon: verctor of efficiencies
      const Int_t *A_rows=fA->GetRowIndexArray();
      const Int_t *A_cols=fA->GetColIndexArray();
      const Double_t *A_data=fA->GetMatrixArray();
      TMatrixD epsilonNosparse(fA->GetNcols(),1);
      for(Int_t i=0;i<A_rows[fA->GetNrows()];i++) {
         epsilonNosparse(A_cols[i],0) += A_data[i];
      }
      epsilon=new TMatrixDSparse(epsilonNosparse);
      // calculate vector EE*epsilon
      Eepsilon=MultiplyMSparseMSparse(fE,epsilon);
      // calculate scalar product epsilon#*Eepsilon
      TMatrixDSparse *epsilonEepsilon=MultiplyMSparseTranspMSparse(epsilon,
                                                                   Eepsilon);
      // if epsilonEepsilon is zero, nothing works...
      if(epsilonEepsilon->GetRowIndexArray()[1]==1) {
         one_over_epsEeps=1./epsilonEepsilon->GetMatrixArray()[0];
      } else {
         Fatal("TUnfold::Unfold","epsilon#Eepsilon has dimension %d != 1",
               epsilonEepsilon->GetRowIndexArray()[1]);
      }
      DeleteMatrix(&epsilonEepsilon);
      // calculate sum(Y)
      Double_t y_minus_epsx=0.0;
      for(Int_t iy=0;iy<fY->GetNrows();iy++) {
         y_minus_epsx += (*fY)(iy,0);
      }
      // calculate sum(Y)-epsilon#*X
      for(Int_t ix=0;ix<epsilonNosparse.GetNrows();ix++) {
         y_minus_epsx -=  epsilonNosparse(ix,0) * (*fX)(ix,0);
      }
      // calculate lambda_half
      lambda_half=y_minus_epsx*one_over_epsEeps;
      // calculate final vector X
      const Int_t *EEpsilon_rows=Eepsilon->GetRowIndexArray();
      const Double_t *EEpsilon_data=Eepsilon->GetMatrixArray();
      for(Int_t ix=0;ix<Eepsilon->GetNrows();ix++) {
         if(EEpsilon_rows[ix]<EEpsilon_rows[ix+1]) {
            (*fX)(ix,0) += lambda_half * EEpsilon_data[EEpsilon_rows[ix]];
         }
      }
   }
   //
   // get derivative dx/dy
   // for error propagation
   //     dx/dy = E A# Vyy^-1  ( = B )
   fDXDY = MultiplyMSparseMSparse(fE,AtVyyinv);

   // additional correction for constraint
   if(fConstraint != kEConstraintNone) {
      // transposed vector of dimension GetNy() all elements equal 1/epseEeps
      Int_t *rows=new Int_t[GetNy()];
      Int_t *cols=new Int_t[GetNy()];
      Double_t *data=new Double_t[GetNy()];
      for(Int_t i=0;i<GetNy();i++) {
         rows[i]=0;
         cols[i]=i;
         data[i]=one_over_epsEeps;
      } 
      TMatrixDSparse *temp=CreateSparseMatrix
         (1,GetNy(),GetNy(),rows,cols,data);
      delete[] data;
      delete[] rows;
      delete[] cols;
      // B# * epsilon
      TMatrixDSparse *epsilonB=MultiplyMSparseTranspMSparse(epsilon,fDXDY);
      // temp- one_over_epsEeps*Bepsilon
      AddMSparse(temp, -one_over_epsEeps, epsilonB);
      DeleteMatrix(&epsilonB);
      // correction matrix
      TMatrixDSparse *corr=MultiplyMSparseMSparse(Eepsilon,temp);
      DeleteMatrix(&temp);
      // determine new derivative
      AddMSparse(fDXDY,1.0,corr);
      DeleteMatrix(&corr);
   }

   DeleteMatrix(&AtVyyinv);

   //
   // get error matrix on x
   //   fDXDY * Vyy * fDXDY# = fDXDY * A * E    (Note: E# = E)
   TMatrixDSparse *AE = MultiplyMSparseMSparse(fA,fE);
   fVxx = MultiplyMSparseMSparse(fDXDY,AE);

   DeleteMatrix(&AE);

   //
   // get result
   //        fA x  =  fAx
   //
   fAx = MultiplyMSparseM(fA,fX);

   //
   // calculate chi**2 etc

   // chi**2 contribution from (y-Ax)V(y-Ax)
   TMatrixD dy(*fY, TMatrixD::kMinus, *fAx);
   TMatrixDSparse *VyyinvDy=MultiplyMSparseM(Vyyinv,&dy);
   DeleteMatrix(&Vyyinv);

   const Int_t *VyyinvDy_rows=VyyinvDy->GetRowIndexArray();
   const Double_t *VyyinvDy_data=VyyinvDy->GetMatrixArray();
   fChi2A=0.0;
   for(Int_t iy=0;iy<VyyinvDy->GetNrows();iy++) {
      if(VyyinvDy_rows[iy]<VyyinvDy_rows[iy+1]) {
         fChi2A += VyyinvDy_data[VyyinvDy_rows[iy]]*dy(iy,0);
      }
   }
   TMatrixD dx( fBiasScale * (*fX0), TMatrixD::kMinus,(*fX));
   TMatrixDSparse *LsquaredDx=MultiplyMSparseM(fLsquared,&dx);
   const Int_t *LsquaredDx_rows=LsquaredDx->GetRowIndexArray();
   const Double_t *LsquaredDx_data=LsquaredDx->GetMatrixArray();
   fLXsquared = 0.0;
   for(Int_t ix=0;ix<LsquaredDx->GetNrows();ix++) {
      if(LsquaredDx_rows[ix]<LsquaredDx_rows[ix+1]) {
         fLXsquared += LsquaredDx_data[LsquaredDx_rows[ix]]*dx(ix,0);
      }
   }

   //
   // get derivative dx/dtau
   fDXDtauSquared=MultiplyMSparseMSparse(fE,LsquaredDx);

   if(fConstraint != kEConstraintNone) {
      TMatrixDSparse *temp=MultiplyMSparseTranspMSparse(epsilon,fDXDtauSquared);
      Double_t f=0.0;
      if(temp->GetRowIndexArray()[1]==1) {
         f=temp->GetMatrixArray()[0]*one_over_epsEeps;
      } else if(temp->GetRowIndexArray()[1]>1) {
         Fatal("TUnfold::Unfold",
               "epsilon#fDXDtauSquared has dimension %d != 1",
               temp->GetRowIndexArray()[1]);
      }
      if(f!=0.0) {
         AddMSparse(fDXDtauSquared, -f,Eepsilon);
      }
      DeleteMatrix(&temp);
   }
   DeleteMatrix(&epsilon);

   DeleteMatrix(&LsquaredDx);

   // calculate/store matrices defining the derivatives dx/dA
   fDXDAM[0]=new TMatrixDSparse(*fE);
   fDXDAM[1]=new TMatrixDSparse(*fDXDY); // create a copy
   fDXDAZ[0]=VyyinvDy; // instead of deleting VyyinvDy
   VyyinvDy=0;
   fDXDAZ[1]=new TMatrixDSparse(*fX); // create a copy

   if(fConstraint != kEConstraintNone) {
      // add correction to fDXDAM[0]
      TMatrixDSparse *temp1=MultiplyMSparseMSparseTranspVector
         (Eepsilon,Eepsilon,0);
      AddMSparse(fDXDAM[0], -one_over_epsEeps,temp1);
      DeleteMatrix(&temp1);
      // add correction to fDXDAZ[0]
      Int_t *rows=new Int_t[GetNy()];
      Int_t *cols=new Int_t[GetNy()];
      Double_t *data=new Double_t[GetNy()];
      for(Int_t i=0;i<GetNy();i++) {
         rows[i]=i;
         cols[i]=0;
         data[i]=lambda_half;
      }
      TMatrixDSparse *temp2=CreateSparseMatrix
         (GetNy(),1,GetNy(),rows,cols,data);
      delete[] data;
      delete[] rows;
      delete[] cols;
      AddMSparse(fDXDAZ[0],1.0,temp2);
      DeleteMatrix(&temp2);
   }

   DeleteMatrix(&Eepsilon);
   DeleteMatrix(&EE);

   TMatrixD *VxxINV = InvertMSparse(fVxx);
   fVxxInv=new TMatrixDSparse(*VxxINV);

   // maximum global correlation coefficient
   const Int_t *Vxx_rows=fVxx->GetRowIndexArray();
   const Int_t *Vxx_cols=fVxx->GetColIndexArray();
   const Double_t *Vxx_data=fVxx->GetMatrixArray();

   Double_t rho_squared_max = 0.0;
   Double_t rho_sum = 0.0;
   Int_t n_rho=0;
   for (int ix = 0; ix < fVxx->GetNrows(); ix++) {
      for(int ik=Vxx_rows[ix];ik<Vxx_rows[ix+1];ik++) {
         if(ix==Vxx_cols[ik]) {
            Double_t rho_squared =
               1. - 1. / ((*VxxINV) (ix, ix) * Vxx_data[ik]);
            if (rho_squared > rho_squared_max)
               rho_squared_max = rho_squared;
            if(rho_squared>0.0) {
               rho_sum += TMath::Sqrt(rho_squared);
               n_rho++;               
            }
            break;
         }
      }
   }
   fRhoMax = TMath::Sqrt(rho_squared_max);
   fRhoAvg = (n_rho>0) ? (rho_sum/n_rho) : -1.0;

   delete VxxINV;

   return fRhoMax;
}

TMatrixDSparse *TUnfold::CreateSparseMatrix
(Int_t nrow,Int_t ncol,Int_t nel,Int_t *row,Int_t *col,Double_t *data) const {
   TMatrixDSparse *A=new TMatrixDSparse(nrow,ncol);
   if(nel>0) {
      A->SetMatrixArray(nel,row,col,data);
   }
   return A;
}

TMatrixDSparse *TUnfold::MultiplyMSparseMSparse(const TMatrixDSparse *a,
                                                const TMatrixDSparse *b) const
{
   // calculate the product of two sparse matrices
   //    a,b: pointers to sparse matrices, where a->GetNcols()==b->GetNrows()
   // this is a replacement for the call
   //    new TMatrixDSparse(*a,TMatrixDSparse::kMult,*b);
   if(a->GetNcols()!=b->GetNrows()) {
      Fatal("MultiplyMSparseMSparse",
            "inconsistent matrix col/ matrix row %d !=%d",
            a->GetNcols(),b->GetNrows());
   }

   TMatrixDSparse *r=new TMatrixDSparse(a->GetNrows(),b->GetNcols());
   const Int_t *a_rows=a->GetRowIndexArray();
   const Int_t *a_cols=a->GetColIndexArray();
   const Double_t *a_data=a->GetMatrixArray();
   const Int_t *b_rows=b->GetRowIndexArray();
   const Int_t *b_cols=b->GetColIndexArray();
   const Double_t *b_data=b->GetMatrixArray();
   // maximum size of the output matrix
   int nMax=0;
   for (Int_t irow = 0; irow < a->GetNrows(); irow++) {
      if(a_rows[irow+1]>a_rows[irow]) nMax += b->GetNcols();
   }
   if((nMax>0)&&(a_cols)&&(b_cols)) {
      Int_t *r_rows=new Int_t[nMax];
      Int_t *r_cols=new Int_t[nMax];
      Double_t *r_data=new Double_t[nMax];
      Double_t *row_data=new Double_t[b->GetNcols()];
      Int_t n=0;
      for (Int_t irow = 0; irow < a->GetNrows(); irow++) {
         if(a_rows[irow+1]<=a_rows[irow]) continue;
         // clear row data
         for(Int_t icol=0;icol<b->GetNcols();icol++) {
            row_data[icol]=0.0;
         }
         // loop over a-columns in this a-row
         for(Int_t ia=a_rows[irow];ia<a_rows[irow+1];ia++) {
            Int_t k=a_cols[ia];
            // loop over b-columns in b-row k
            for(Int_t ib=b_rows[k];ib<b_rows[k+1];ib++) {
               row_data[b_cols[ib]] += a_data[ia]*b_data[ib];
            }
         }
         // store nonzero elements
         for(Int_t icol=0;icol<b->GetNcols();icol++) {
            if(row_data[icol] != 0.0) {
               r_rows[n]=irow;
               r_cols[n]=icol;
               r_data[n]=row_data[icol];
               n++;
            }
         }
      }
      if(n>0) {
         r->SetMatrixArray(n,r_rows,r_cols,r_data);
      }
      delete[] r_rows;
      delete[] r_cols;
      delete[] r_data;
      delete[] row_data;
   }

   return r;
}


TMatrixDSparse *TUnfold::MultiplyMSparseTranspMSparse(const TMatrixDSparse *a,
                                                      const TMatrixDSparse *b) const
{
   // multiply a transposed Sparse matrix with another Sparse matrix
   //    a:  pointer to sparse matrix (to be transposed)
   //    b:  pointer to sparse matrix
   // this is a replacement for the call
   //    new TMatrixDSparse(TMatrixDSparse(TMatrixDSparse::kTransposed,*a),
   //                       TMatrixDSparse::kMult,*b)
   if(a->GetNrows() != b->GetNrows()) {
      Fatal("MultiplyMSparseTranspMSparse",
            "inconsistent matrix row numbers %d!=%d",
            a->GetNrows(),b->GetNrows());
   }

   TMatrixDSparse *r=new TMatrixDSparse(a->GetNcols(),b->GetNcols());
   const Int_t *a_rows=a->GetRowIndexArray();
   const Int_t *a_cols=a->GetColIndexArray();
   const Double_t *a_data=a->GetMatrixArray();
   const Int_t *b_rows=b->GetRowIndexArray();
   const Int_t *b_cols=b->GetColIndexArray();
   const Double_t *b_data=b->GetMatrixArray();
   // maximum size of the output matrix

   // matrix multiplication
   typedef std::map<Int_t,Double_t> MMatrixRow_t;
   typedef std::map<Int_t, MMatrixRow_t > MMatrix_t;
   MMatrix_t matrix;

   for(Int_t iRowAB=0;iRowAB<a->GetNrows();iRowAB++) {
      for(Int_t ia=a_rows[iRowAB];ia<a_rows[iRowAB+1];ia++) {
         for(Int_t ib=b_rows[iRowAB];ib<b_rows[iRowAB+1];ib++) {
            // this creates a new row if necessary
            MMatrixRow_t &row=matrix[a_cols[ia]];
            MMatrixRow_t::iterator icol=row.find(b_cols[ib]);
            if(icol!=row.end()) {
               // update existing row
               (*icol).second += a_data[ia]*b_data[ib];
            } else {
               // create new row
               row[b_cols[ib]] = a_data[ia]*b_data[ib];
            }
         }
      }
   }

   Int_t n=0;
   for(MMatrix_t::const_iterator irow=matrix.begin();
       irow!=matrix.end();irow++) {
      n += (*irow).second.size();
   }
   if(n>0) {
      // pack matrix into arrays
      Int_t *r_rows=new Int_t[n];
      Int_t *r_cols=new Int_t[n];
      Double_t *r_data=new Double_t[n];
      n=0;
      for(MMatrix_t::const_iterator irow=matrix.begin();
          irow!=matrix.end();irow++) {
         for(MMatrixRow_t::const_iterator icol=(*irow).second.begin();
             icol!=(*irow).second.end();icol++) {
            r_rows[n]=(*irow).first;
            r_cols[n]=(*icol).first;
            r_data[n]=(*icol).second;
            n++;
         }
      }
      // pack arrays into TMatrixDSparse
      if(n>0) {
         r->SetMatrixArray(n,r_rows,r_cols,r_data);
      }
      delete[] r_rows;
      delete[] r_cols;
      delete[] r_data;
   }

   return r;
}

TMatrixDSparse *TUnfold::MultiplyMSparseM(const TMatrixDSparse *a,
                                          const TMatrixD *b) const
{
   // multiply a Sparse matrix with a non-sparse matrix
   //    a:  pointer to sparse matrix
   //    b:  pointer to non-sparse matrix
   // this is a replacement for the call
   //    new TMatrixDSparse(*a,TMatrixDSparse::kMult,*b);
   if(a->GetNcols()!=b->GetNrows()) {
      Fatal("MultiplyMSparseM","inconsistent matrix col /matrix row %d!=%d",
            a->GetNcols(),b->GetNrows());
   }

   TMatrixDSparse *r=new TMatrixDSparse(a->GetNrows(),b->GetNcols());
   const Int_t *a_rows=a->GetRowIndexArray();
   const Int_t *a_cols=a->GetColIndexArray();
   const Double_t *a_data=a->GetMatrixArray();
   // maximum size of the output matrix
   int nMax=0;
   for (Int_t irow = 0; irow < a->GetNrows(); irow++) {
      if(a_rows[irow+1]-a_rows[irow]>0) nMax += b->GetNcols();
   }
   if(nMax>0) {
      Int_t *r_rows=new Int_t[nMax];
      Int_t *r_cols=new Int_t[nMax];
      Double_t *r_data=new Double_t[nMax];

      Int_t n=0;
      // fill matrix r
      for (Int_t irow = 0; irow < a->GetNrows(); irow++) {
         if(a_rows[irow+1]-a_rows[irow]<=0) continue;
         for(Int_t icol=0;icol<b->GetNcols();icol++) {
            r_rows[n]=irow;
            r_cols[n]=icol;
            r_data[n]=0.0;
            for(Int_t i=a_rows[irow];i<a_rows[irow+1];i++) {
               Int_t j=a_cols[i];
               r_data[n] += a_data[i]*(*b)(j,icol);
            }
            if(r_data[n]!=0.0) n++;
         }
      }
      if(n>0) {
         r->SetMatrixArray(n,r_rows,r_cols,r_data);
      }
      delete[] r_rows;
      delete[] r_cols;
      delete[] r_data;
   }
   return r;
}

TMatrixDSparse *TUnfold::MultiplyMSparseMSparseTranspVector
(const TMatrixDSparse *m1,const TMatrixDSparse *m2,
 const TMatrixTBase<Double_t> *v) const {
   // calculate M_ij = sum_k [m1_ik*m2_jk*v[k] ].
   //    m1: pointer to sparse matrix with dimension I*K
   //    m2: pointer to sparse matrix with dimension J*K
   //    v: pointer to vector (matrix) with dimension K*1
   if((m1->GetNcols() != m2->GetNcols())||
      (v && ((m1->GetNcols()!=v->GetNrows())||(v->GetNcols()!=1)))) {
      if(v) {
         Fatal("MultiplyMSparseMSparseTranspVector",
               "matrix cols/vector rows %d!=%d!=%d or vector rows %d!=1\n",
               m1->GetNcols(),m2->GetNcols(),v->GetNrows(),v->GetNcols());
      } else {
         Fatal("MultiplyMSparseMSparseTranspVector",
               "matrix cols %d!=%d\n",m1->GetNcols(),m2->GetNcols());
      }
   }
   const Int_t *rows_m1=m1->GetRowIndexArray();
   const Int_t *cols_m1=m1->GetColIndexArray();
   const Double_t *data_m1=m1->GetMatrixArray();
   Int_t num_m1=0;
   for(Int_t i=0;i<m1->GetNrows();i++) {
      if(rows_m1[i]<rows_m1[i+1]) num_m1++;
   }
   const Int_t *rows_m2=m2->GetRowIndexArray();
   const Int_t *cols_m2=m2->GetColIndexArray();
   const Double_t *data_m2=m2->GetMatrixArray();
   Int_t num_m2=0;
   for(Int_t j=0;j<m2->GetNrows();j++) {
      if(rows_m2[j]<rows_m2[j+1]) num_m2++;
   }
   const TMatrixDSparse *v_sparse=dynamic_cast<const TMatrixDSparse *>(v);
   const Int_t *v_rows=0;
   const Double_t *v_data=0;
   if(v_sparse) {
      v_rows=v_sparse->GetRowIndexArray();
      v_data=v_sparse->GetMatrixArray();
   }
   Int_t num_r=num_m1*num_m2+1;
   Int_t *row_r=new Int_t[num_r];
   Int_t *col_r=new Int_t[num_r];
   Double_t *data_r=new Double_t[num_r];
   num_r=0;
   for(Int_t i=0;i<m1->GetNrows();i++) {
      for(Int_t j=0;j<m2->GetNrows();j++) {
         data_r[num_r]=0.0;
         Int_t index_m1=rows_m1[i];
         Int_t index_m2=rows_m2[j];
         while((index_m1<rows_m1[i+1])&&(index_m2<rows_m2[j+1])) {
            Int_t k1=cols_m1[index_m1];
            Int_t k2=cols_m2[index_m2];
            if(k1<k2) {
               index_m1++;
            } else if(k1>k2) {
               index_m2++;
            } else {
               if(v_sparse) {
                  Int_t v_index=v_rows[k1];
                  if(v_index<v_rows[k1+1]) {
                     data_r[num_r] += data_m1[index_m1] * data_m2[index_m2]
                        * v_data[v_index];
                  } else {
                     data_r[num_r] =0.0;
                  }
               } else if(v) {
                  data_r[num_r] += data_m1[index_m1] * data_m2[index_m2]
                     * (*v)(k1,0);
               } else {
                  data_r[num_r] += data_m1[index_m1] * data_m2[index_m2];
               }
               index_m1++;
               index_m2++;
            }
         }
         if(data_r[num_r] !=0.0) {
            row_r[num_r]=i;
            col_r[num_r]=j;
            num_r++;
         }
      }
   }
   TMatrixDSparse *r=CreateSparseMatrix(m1->GetNrows(),m2->GetNrows(),
                                        num_r,row_r,col_r,data_r);
   delete[] row_r;
   delete[] col_r;
   delete[] data_r;
   return r;
}

void TUnfold::AddMSparse(TMatrixDSparse *dest,Double_t f,
                         const TMatrixDSparse *src) {
   // a replacement for
   //     (*dest) += f*(*src)
   const Int_t *dest_rows=dest->GetRowIndexArray();
   const Int_t *dest_cols=dest->GetColIndexArray();
   const Double_t *dest_data=dest->GetMatrixArray();
   const Int_t *src_rows=src->GetRowIndexArray();
   const Int_t *src_cols=src->GetColIndexArray();
   const Double_t *src_data=src->GetMatrixArray();

   if((dest->GetNrows()!=src->GetNrows())||
      (dest->GetNcols()!=src->GetNcols())) {
      Fatal("AddMSparse","inconsistent matrix rows %d!=%d OR cols %d!=%d",
            src->GetNrows(),dest->GetNrows(),src->GetNcols(),dest->GetNcols());
   }
   Int_t nmax=dest->GetNrows()*dest->GetNcols();
   Double_t *result_data=new Double_t[nmax];
   Int_t *result_rows=new Int_t[nmax];
   Int_t *result_cols=new Int_t[nmax];
   Int_t n=0;
   for(Int_t row=0;row<dest->GetNrows();row++) {
      Int_t i_dest=dest_rows[row];
      Int_t i_src=src_rows[row];
      while((i_dest<dest_rows[row+1])||(i_src<src_rows[row+1])) {
         Int_t col_dest=(i_dest<dest_rows[row+1]) ? 
            dest_cols[i_dest] : dest->GetNcols();
         Int_t col_src =(i_src <src_rows[row+1] ) ?
            src_cols [i_src] :  src->GetNcols();
         result_rows[n]=row;
         if(col_dest<col_src) {
            result_cols[n]=col_dest;
            result_data[n]=dest_data[i_dest++];
         } else if(col_dest>col_src) {
            result_cols[n]=col_src;
            result_data[n]=f*src_data[i_src++];
         } else {
            result_cols[n]=col_dest;
            result_data[n]=dest_data[i_dest++]+f*src_data[i_src++];
         }
         if(result_data[n] !=0.0) n++;
      }
   }
   if(n<=0) {
      n=1;
      result_rows[0]=0;
      result_cols[0]=0;
      result_data[0]=0.0;
   }
   dest->SetMatrixArray(n,result_rows,result_cols,result_data);
   delete[] result_data;
   delete[] result_rows;
   delete[] result_cols;
}

TMatrixD *TUnfold::InvertMSparse(const TMatrixDSparse *A) const {
   // get the inverse of a sparse matrix
   //    A: the original matrix
   // this is a replacement of the call
   //    new TMatrixD(TMatrixD::kInverted, a);
   // the matrix inversion is optimized for the case
   // where a large submatrix of A is diagonal

   if(A->GetNcols()!=A->GetNrows()) {
      Fatal("InvertMSparse","inconsistent matrix row/col %d!=%d",
            A->GetNcols(),A->GetNrows());
   }

#ifdef SCHUR_COMPLEMENT_MATRIX_INVERSION

   const Int_t *a_rows=A->GetRowIndexArray();
   const Int_t *a_cols=A->GetColIndexArray();
   const Double_t *a_data=A->GetMatrixArray();

   Int_t nmin=0,nmax=0;
   // find largest diagonal submatrix
   for(Int_t imin=0;imin<A->GetNrows();imin++) {
      Int_t imax=A->GetNrows();
      for(Int_t i2=imin;i2<imax;i2++) {
         for(Int_t i=a_rows[i2];i<a_rows[i2+1];i++) {
            if(a_data[i]==0.0) continue;
            Int_t icol=a_cols[i];
            if(icol<imin) continue; // ignore first part of the matrix
            if(icol==i2) continue; // ignore diagonals
            if(icol<i2) {
               // this row spoils the diagonal matrix, so do not use
               imax=i2;
               break;
            } else {
               // this entry limits imax
               imax=icol;
               break;
            }
         }
      }
      if(imax-imin>nmax-nmin) {
         nmin=imin;
         nmax=imax;
      }
   }
   // if the diagonal part has size zero or one, use standard matrix inversion
   if(nmin>=nmax-1) {
      TMatrixD *r=new TMatrixD(*A);
      if(!InvertMConditioned(r)) {
         Fatal("InvertMSparse","InvertMConditioned(full matrix) failed");
      }
      return r;
   } else if((nmin==0)&&(nmax==A->GetNrows())) {
      // if the diagonal part spans the whole matrix,
      //   just set the diagomal elements
      TMatrixD *r=new TMatrixD(A->GetNrows(),A->GetNcols());
      Int_t error=0;
      for(Int_t irow=nmin;irow<nmax;irow++) {
         for(Int_t i=a_rows[irow];i<a_rows[irow+1];i++) {
            if(a_cols[i]==irow) {
               if(a_data[i] !=0.0) (*r)(irow,irow)=1./a_data[i];
               else error++;
            }
         }
      }
      if(error) {
         Error("InvertMSparse",
               "inversion failed (diagonal matrix) nerror=%d",error);
      }
      return r;
   }

   //  A  B
   // (    )
   //  C  D
   //
   // get inverse of diagonal part
   std::vector<Double_t> Dinv;
   Dinv.resize(nmax-nmin);
   Int_t error=0;
   for(Int_t irow=nmin;irow<nmax;irow++) {
      for(Int_t i=a_rows[irow];i<a_rows[irow+1];i++) {
         if(a_cols[i]==irow) {
            if(a_data[i]!=0.0) {
               Dinv[irow-nmin]=1./a_data[i];
            } else {
               Dinv[irow-nmin]=0.0;
               error++;
            }
            break;
         }
      }
   }
   if(error) {
      Error("InvertMSparse",
            "inversion failed (diagonal part) nerror=%d",error);      
   }
   // B*Dinv and C
   Int_t nBDinv=0,nC=0;
   for(Int_t irow_a=0;irow_a<A->GetNrows();irow_a++) {
      if((irow_a<nmin)||(irow_a>=nmax)) {
         for(Int_t i=a_rows[irow_a];i<a_rows[irow_a+1];i++) {
            Int_t icol=a_cols[i];
            if((icol>=nmin)&&(icol<nmax)) nBDinv++;
         }
      } else {
         for(Int_t i=a_rows[irow_a];i<a_rows[irow_a+1];i++) {
            Int_t icol = a_cols[i];
            if((icol<nmin)||(icol>=nmax)) nC++;
         }
      }
   }
   Int_t *row_BDinv=new Int_t[nBDinv+1];
   Int_t *col_BDinv=new Int_t[nBDinv+1];
   Double_t *data_BDinv=new Double_t[nBDinv+1];

   Int_t *row_C=new Int_t[nC+1];
   Int_t *col_C=new Int_t[nC+1];
   Double_t *data_C=new Double_t[nC+1];

   TMatrixD Aschur(A->GetNrows()-(nmax-nmin),A->GetNcols()-(nmax-nmin));

   nBDinv=0;
   nC=0;
   for(Int_t irow_a=0;irow_a<A->GetNrows();irow_a++) {
      if((irow_a<nmin)||(irow_a>=nmax)) {
         Int_t row=(irow_a<nmin) ? irow_a : (irow_a-(nmax-nmin));
         for(Int_t i=a_rows[irow_a];i<a_rows[irow_a+1];i++) {
            Int_t icol_a=a_cols[i];
            if(icol_a<nmin) {
               Aschur(row,icol_a)=a_data[i];
            } else if(icol_a>=nmax) {
               Aschur(row,icol_a-(nmax-nmin))=a_data[i];
            } else {
               row_BDinv[nBDinv]=row;
               col_BDinv[nBDinv]=icol_a-nmin;
               data_BDinv[nBDinv]=a_data[i]*Dinv[icol_a-nmin];
               nBDinv++;
            }
         }
      } else {
         for(Int_t i=a_rows[irow_a];i<a_rows[irow_a+1];i++) {
            Int_t icol_a=a_cols[i];
            if(icol_a<nmin) {
               row_C[nC]=irow_a-nmin;
               col_C[nC]=icol_a;
               data_C[nC]=a_data[i];
               nC++;
            } else if(icol_a>=nmax) {
               row_C[nC]=irow_a-nmin;
               col_C[nC]=icol_a-(nmax-nmin);
               data_C[nC]=a_data[i];
               nC++;
            }
         }
      }
   }
   TMatrixDSparse *BDinv=CreateSparseMatrix
      (A->GetNrows()-(nmax-nmin),nmax-nmin,
       nBDinv,row_BDinv,col_BDinv,data_BDinv);
   delete[] row_BDinv;
   delete[] col_BDinv;
   delete[] data_BDinv;


   TMatrixDSparse *C=CreateSparseMatrix(nmax-nmin,A->GetNcols()-(nmax-nmin),
                                        nC,row_C,col_C,data_C);
   delete[] row_C;
   delete[] col_C;
   delete[] data_C;

   TMatrixDSparse *BDinvC=MultiplyMSparseMSparse(BDinv,C);

   Aschur -= *BDinvC;
   if(!InvertMConditioned(&Aschur)) {
      Fatal("InvertMSparse","InvertMConditioned failed (part of matrix)");
   }

   DeleteMatrix(&BDinvC);

   TMatrixD *r=new TMatrixD(A->GetNrows(),A->GetNcols());

   for(Int_t row_a=0;row_a<Aschur.GetNrows();row_a++) {
      for(Int_t col_a=0;col_a<Aschur.GetNcols();col_a++) {
         (*r)((row_a<nmin) ? row_a : (row_a+nmax-nmin),
              (col_a<nmin) ? col_a : (col_a+nmax-nmin))=Aschur(row_a,col_a);
      }
   }

   TMatrixDSparse *CAschur=MultiplyMSparseM(C,&Aschur);
   TMatrixDSparse *CAschurBDinv=MultiplyMSparseMSparse(CAschur,BDinv);

   DeleteMatrix(&C);

   const Int_t *CAschurBDinv_row=CAschurBDinv->GetRowIndexArray();
   const Int_t *CAschurBDinv_col=CAschurBDinv->GetColIndexArray();
   const Double_t *CAschurBDinv_data=CAschurBDinv->GetMatrixArray();
   for(Int_t row=0;row<CAschurBDinv->GetNrows();row++) {
      for(Int_t i=CAschurBDinv_row[row];i<CAschurBDinv_row[row+1];i++) {
         Int_t col=CAschurBDinv_col[i];
         (*r)(row+nmin,col+nmin)=CAschurBDinv_data[i]*Dinv[row];
      }
      (*r)(row+nmin,row+nmin) += Dinv[row];
   }

   DeleteMatrix(&CAschurBDinv);

   const Int_t *CAschur_row=CAschur->GetRowIndexArray();
   const Int_t *CAschur_col=CAschur->GetColIndexArray();
   const Double_t *CAschur_data=CAschur->GetMatrixArray();
   for(Int_t row=0;row<CAschur->GetNrows();row++) {
      for(Int_t i=CAschur_row[row];i<CAschur_row[row+1];i++) {
         Int_t col=CAschur_col[i];
         (*r)(row+nmin,
              (col<nmin) ? col : (col+nmax-nmin))= -CAschur_data[i]*Dinv[row];
      }
   }
   DeleteMatrix(&CAschur);

   const Int_t *BDinv_row=BDinv->GetRowIndexArray();
   const Int_t *BDinv_col=BDinv->GetColIndexArray();
   const Double_t *BDinv_data=BDinv->GetMatrixArray();  
   for(Int_t row_aschur=0;row_aschur<Aschur.GetNrows();row_aschur++) {
      Int_t row=(row_aschur<nmin) ? row_aschur : (row_aschur+nmax-nmin);
      for(Int_t row_bdinv=0;row_bdinv<BDinv->GetNrows();row_bdinv++) {
         for(Int_t i=BDinv_row[row_bdinv];i<BDinv_row[row_bdinv+1];i++) {
            (*r)(row,BDinv_col[i]+nmin) -= Aschur(row_aschur,row_bdinv)*
               BDinv_data[i];
         }
      }
   }
   DeleteMatrix(&BDinv);

   return r;
#else
   TMatrixD *r=new TMatrixD(A);
   if(!InvertMConditioned(*r)) {
      Fatal("InvertMSparse","InvertMConditioned failed (full matrix)";
      print_backtrace();
   }
   return r;
#endif
}

Bool_t TUnfold::InvertMConditioned(TMatrixD *A) {
   // invert the matrix A
   // the inversion is done with pre-conditioning
   // all rows and columns are normalized to sqrt(abs(a_ii*a_jj))
   // such that the diagonals are equal to 1.0
   // This type of preconditioning improves the numerival results
   // for the symmetric, positive definite matrices which are
   // treated here in the context of unfolding
#ifdef INVERT_WITH_CONDITIONING
   // divide matrix by the square-root of its diagonals
   Double_t *A_diagonals=new Double_t[A->GetNrows()];
   for(Int_t i=0;i<A->GetNrows();i++) {
      A_diagonals[i]=TMath::Sqrt(TMath::Abs((*A)(i,i)));
      if(A_diagonals[i]>0.0) A_diagonals[i]=1./A_diagonals[i];
      else A_diagonals[i]=1.0;
   }
   // condition the matrix prior to inversion
   for(Int_t i=0;i<A->GetNrows();i++) {
      for(Int_t j=0;j<A->GetNcols();j++) {
         (*A)(i,j) *= A_diagonals[i]*A_diagonals[j];
      }
   }
#endif
   Double_t det=0.0;
   A->Invert(&det);
#ifdef INVERT_WITH_CONDITIONING
   // revert conditioning on the inverted matrix
   for(Int_t i=0;i<A->GetNrows();i++) {
      for(Int_t j=0;j<A->GetNcols();j++) {
         (*A)(i,j) *= A_diagonals[i]*A_diagonals[j];
      }
   }
   delete[] A_diagonals;
#endif
   return (det !=0.0);
}

TUnfold::TUnfold(const TH2 *hist_A, EHistMap histmap, ERegMode regmode,
                 TUnfold::EConstraint constraint)
{
   // set up unfolding matrix and initial regularisation scheme
   //    hist_A:  matrix that describes the migrations
   //    histmap: mapping of the histogram axes to the unfolding output 
   //    regmode: global regularisation mode
   //    constraint: type of constraint to use
   // data members initialized to something different from zero:
   //    fA: filled from hist_A
   //    fDA: filled from hist_A
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

   InitTUnfold();
   SetConstraint(constraint);
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
   fSumOverY.Set(nx0);
   fXToHist.Set(nx0);
   fHistToX.Set(nx0);
   Int_t nonzeroA=0;
   // loop
   //  - calculate bias distribution
   //      sum_over_y
   //  - count those generator bins which can be unfolded
   //      fNx
   //  - histogram bins are added to the lookup-tables
   //      fHistToX[] and fXToHist[]
   //    these convert histogram bins to matrix indices and vice versa
   //  - the number of non-zero elements is counted
   //      nonzeroA
   Int_t nskipped=0;
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
            nonzeroA++;
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
         fSumOverY[nx] = sum;
         if (histmap == kHistMapOutputHoriz) {
            fSumOverY[nx] +=
               hist_A->GetBinContent(ix, 0) +
               hist_A->GetBinContent(ix, ny + 1);
         } else {
            fSumOverY[nx] +=
               hist_A->GetBinContent(0, ix) +
               hist_A->GetBinContent(ny + 1, ix);
         }
         nx++;
      } else {
         nskipped++;
         // histogram bin pointing to -1 (non-existing matrix entry)
         fHistToX[ix] = -1;
      }
   }
   Int_t underflowBin=0,overflowBin=0;
   for (Int_t ix = 0; ix < nx; ix++) {
      Int_t ibinx = fXToHist[ix];
      if(ibinx<1) underflowBin=1;
      if (histmap == kHistMapOutputHoriz) {
         if(ibinx>hist_A->GetNbinsX()) overflowBin=1;
      } else {
         if(ibinx>hist_A->GetNbinsY()) overflowBin=1;
      }
   }
   if(nskipped) {
      Int_t nprint=0;
      Int_t ixfirst=-1,ixlast=-1;
      TString binlist;
      for (Int_t ix = 0; ix < nx0; ix++) {
         if(fHistToX[ix]<0) {
            nprint++;
            if(ixlast<0) {
               binlist +=" ";
               binlist +=ix;
               ixfirst=ix;
            }
            ixlast=ix;
         }
         if(((fHistToX[ix]>=0)&&(ixlast>=0))||
            (nprint==nskipped)) {
            if(ixlast>ixfirst) {
               binlist += "-";
               binlist += ixlast;
            }
            ixfirst=-1;
            ixlast=-1;
         }
         if(nprint==nskipped) {
            break;
         }
      }
      if(nskipped==(2-underflowBin-overflowBin)) {
         Info("TUnfold","the following output bins "
              "are not connected to the input side %s",
              static_cast<const char *>(binlist));
      } else {
         Warning("TUnfold","the following output bins "
                 "are not connected to the input side %s",
                 static_cast<const char *>(binlist));
      }
   }
   // store bias as matrix
   fX0 = new TMatrixD(nx, 1, fSumOverY.GetArray());
   // store non-zero elements in sparse matrix fA
   // construct sparse matrix fA
   Int_t *rowA = new Int_t[nonzeroA];
   Int_t *colA = new Int_t[nonzeroA];
   Double_t *dataA = new Double_t[nonzeroA];
   Int_t index=0;
   for (Int_t iy = 0; iy < ny; iy++) {
      for (Int_t ix = 0; ix < nx; ix++) {
         Int_t ibinx = fXToHist[ix];
         Double_t z;
         if (histmap == kHistMapOutputHoriz) {
            z = hist_A->GetBinContent(ibinx, iy + 1);
         } else {
            z = hist_A->GetBinContent(iy + 1, ibinx);
         }
         if (z > 0.0) {
            rowA[index]=iy;
            colA[index] = ix;
            dataA[index] = z / fSumOverY[ix];
            index++;
         }
      }
   }
   if(underflowBin && overflowBin) {
      Info("TUnfold","%d input bins and %d output bins (includes 2 underflow/overflow bins)",ny,nx);
   } else if(underflowBin) {
      Info("TUnfold","%d input bins and %d output bins (includes 1 underflow bin)",ny,nx);
   } else if(overflowBin) {
      Info("TUnfold","%d input bins and %d output bins (includes 1 overflow bin)",ny,nx);
   } else {
      Info("TUnfold","%d input bins and %d output bins",ny,nx);
   }
   fA = CreateSparseMatrix(ny,nx,index,rowA,colA,dataA);
   if(ny<nx) {
      Error("TUnfold","too few (ny=%d) input bins for nx=%d output bins",ny,nx);
   } else if(ny==nx) {
      Warning("TUnfold","too few (ny=%d) input bins for nx=%d output bins",ny,nx);
   }
   delete[] rowA;
   delete[] colA;
   delete[] dataA;
   // regularisation conditions squared
   fLsquared = new TMatrixDSparse(GetNx(), GetNx());
   if (regmode != kRegModeNone) {
      Int_t nError=RegularizeBins(0, 1, nx0, regmode);
      if(nError>0) {
         if(nError>1) {
            Info("TUnfold",
                 "%d regularisation conditions have been skipped",nError);
         } else {
            Info("TUnfold",
                 "One regularisation condition has been skipped");
         }
      }
   }
}

TUnfold::~TUnfold(void)
{
   // delete all data members

   DeleteMatrix(&fA);
   DeleteMatrix(&fLsquared);
   DeleteMatrix(&fVyy);
   DeleteMatrix(&fY);
   DeleteMatrix(&fX0);

   ClearResults();
}

void TUnfold::SetBias(const TH1 *bias)
{
   // initialize alternative bias from histogram
   // modifies data member fX0
   DeleteMatrix(&fX0);
   fX0 = new TMatrixD(GetNx(), 1);
   for (Int_t i = 0; i < GetNx(); i++) {
      (*fX0) (i, 0) = bias->GetBinContent(fXToHist[i]);
   }
}

Int_t TUnfold::RegularizeSize(int bin, Double_t scale)
{
   // add regularisation on the size of bin i
   //    bin: bin number
   //    scale: size of the regularisation
   // return value: number of conditions which have been skipped
   // modifies data member fLsquared

   if(fRegMode==kRegModeNone) fRegMode=kRegModeSize;
   if(fRegMode!=kRegModeSize) fRegMode=kRegModeMixed;

   Int_t i = fHistToX[bin];
   if (i < 0) {
      return 1;
   }
   (*fLsquared) (i, i) = scale * scale;
   return 0;
}

Int_t TUnfold::RegularizeDerivative(int left_bin, int right_bin,
                                   Double_t scale)
{
   // add regularisation on the difference of two bins
   //   left_bin: 1st bin
   //   right_bin: 2nd bin
   //   scale: size of the regularisation
   // return value: number of conditions which have been skipped
   // modifies data member fLsquared

   if(fRegMode==kRegModeNone) fRegMode=kRegModeDerivative;
   if(fRegMode!=kRegModeDerivative) fRegMode=kRegModeMixed;

   Int_t il = fHistToX[left_bin];
   Int_t ir = fHistToX[right_bin];
   if ((il < 0) || (ir < 0)) {
      return 1;
   }
   Double_t scale_squared = scale * scale;
   (*fLsquared) (il, il) += scale_squared;
   (*fLsquared) (il, ir) -= scale_squared;
   (*fLsquared) (ir, il) -= scale_squared;
   (*fLsquared) (ir, ir) += scale_squared;
   return 0;
}

Int_t TUnfold::RegularizeCurvature(int left_bin, int center_bin,
                                  int right_bin,
                                  Double_t scale_left,
                                  Double_t scale_right)
{
   // add regularisation on the curvature through 3 bins (2nd derivative)
   //   left_bin: 1st bin
   //   center_bin: 2nd bin
   //   right_bin: 3rd bin
   //   scale_left: scale factor on center-left difference
   //   scale_right: scale factor on right-center difference
   // return value: number of conditions which have been skipped
   // modifies data member fLsquared

   if(fRegMode==kRegModeNone) fRegMode=kRegModeCurvature;
   if(fRegMode!=kRegModeCurvature) fRegMode=kRegModeMixed;

   Int_t il, ic, ir;
   il = fHistToX[left_bin];
   ic = fHistToX[center_bin];
   ir = fHistToX[right_bin];
   if ((il < 0) || (ic < 0) || (ir < 0)) {
      return 1;
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
   return 0;
}

Int_t TUnfold::RegularizeBins(int start, int step, int nbin,
                             ERegMode regmode)
{
   // set regulatisation on a 1-dimensional curve
   //   start: first bin
   //   step:  distance between neighbouring bins
   //   nbin:  total number of bins
   //   regmode:  regularisation mode
   // return value:
   //   number of errors (i.e. conditions which have been skipped)
   // modifies data member fLsquared

   Int_t i0, i1, i2;
   i0 = start;
   i1 = i0 + step;
   i2 = i1 + step;
   Int_t nSkip = 0;
   Int_t nError= 0;
   if (regmode == kRegModeDerivative) {
      nSkip = 1;
   } else if (regmode == kRegModeCurvature) {
      nSkip = 2;
   } else if(regmode != kRegModeSize) {
      Error("TUnfold::RegularizeBins","regmode = %d is not valid",regmode);
   }
   for (Int_t i = nSkip; i < nbin; i++) {
      if (regmode == kRegModeSize) {
         nError += RegularizeSize(i0);
      } else if (regmode == kRegModeDerivative) {
         nError += RegularizeDerivative(i0, i1);
      } else if (regmode == kRegModeCurvature) {
         nError += RegularizeCurvature(i0, i1, i2);
      }
      i0 = i1;
      i1 = i2;
      i2 += step;
   }
   return nError;
}

Int_t TUnfold::RegularizeBins2D(int start_bin, int step1, int nbin1,
                               int step2, int nbin2, ERegMode regmode)
{
   // set regularisation on a 2-dimensional grid of bins
   //     start: first bin
   //     step1: distance between bins in 1st direction
   //     nbin1: number of bins in 1st direction
   //     step2: distance between bins in 2nd direction
   //     nbin2: number of bins in 2nd direction
    // return value:
   //   number of errors (i.e. conditions which have been skipped)
  // modifies data member fLsquared

   Int_t nError = 0;
   for (Int_t i1 = 0; i1 < nbin1; i1++) {
      nError += RegularizeBins(start_bin + step1 * i1, step2, nbin2, regmode);
   }
   for (Int_t i2 = 0; i2 < nbin2; i2++) {
      nError += RegularizeBins(start_bin + step2 * i2, step1, nbin1, regmode);
   }
   return nError;
}

Double_t TUnfold::DoUnfold(Double_t tau_reg,const TH1 *input,
                           Double_t scaleBias)
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

Int_t TUnfold::SetInput(const TH1 *input, Double_t scaleBias,
                        Double_t oneOverZeroError) {
  // Define the input data for subsequent calls to DoUnfold(Double_t)
  //   input:   input distribution with errors
  //   scaleBias:  scale factor applied to the bias
  //   oneOverZeroError: for bins with zero error, this number defines 1/error.
  // Return value: number of bins with bad error
  //                 +10000*number of unconstrained output bins
  //         Note: return values>=10000 are fatal errors, 
  //               for the given input, the unfolding can not be done!
  // Data members modified:
  //   fY, fVyy, fVyyinv, fBiasScale, fNdf
  // Data members cleared
  //   see ClearResults
  fBiasScale = scaleBias;

   // delete old results (if any)
  ClearResults();

  // construct error matrix and inverted error matrix of measured quantities
  // from errors of input histogram
  // and count number of degrees of freedom
  fNdf = -GetNpar();
  Int_t *rowColVyy=new Int_t[GetNy()];
  Int_t *col1Vyy=new Int_t[GetNy()];
  Double_t *dataVyy=new Double_t[GetNy()];
  Int_t nError=0;
  for (Int_t iy = 0; iy < GetNy(); iy++) {
     Double_t dy = input->GetBinError(iy + 1);
     rowColVyy[iy] = iy;
     col1Vyy[iy] = 0;
     if (dy <= 0.0) {
        nError++;
        if(oneOverZeroError>0.0) {
           dataVyy[iy] = 1./ ( oneOverZeroError*oneOverZeroError);
        } else {
           dataVyy[iy] = 0.0;
        }
     } else {
        dataVyy[iy] = dy * dy;
     }
     if( dataVyy[iy]>0.0) fNdf ++;
  }
  DeleteMatrix(&fVyy);
  fVyy = CreateSparseMatrix
     (GetNy(),GetNy(),GetNy(),rowColVyy,rowColVyy,dataVyy);

  TMatrixDSparse *vecV=CreateSparseMatrix
     (GetNy(),1,GetNy(),rowColVyy,col1Vyy, dataVyy);
  delete[] rowColVyy;
  delete[] col1Vyy;
  delete[] dataVyy;

  //
  // get input vector
  DeleteMatrix(&fY);
  fY = new TMatrixD(GetNy(), 1);
  for (Int_t i = 0; i < GetNy(); i++) {
     (*fY) (i, 0) = input->GetBinContent(i + 1);
  }
  // simple check whether unfolding is possible, given the matrices fA and  fV
  TMatrixDSparse *mAtV=MultiplyMSparseTranspMSparse(fA,vecV);
  DeleteMatrix(&vecV);
  Int_t nError2=0;
  for (Int_t i = 0; i <mAtV->GetNrows();i++) {
     if(mAtV->GetRowIndexArray()[i]==
        mAtV->GetRowIndexArray()[i+1]) {
        nError2 ++;
     }
  }
  if(nError>0) {
     if(oneOverZeroError !=0.0) {
        if(nError>1) {
           Warning("SetInput","%d input bins have zero error,"
                   " 1/error set to %lf.",nError,oneOverZeroError);
        } else {
           Warning("SetInput","One input bin has zero error,"
                   " 1/error set to %lf.",oneOverZeroError);
        }
     } else {
        if(nError>1) {
           Warning("SetInput","%d input bins have zero error,"
                   " and are ignored.",nError);
        } else {
           Warning("SetInput","One input bin has zero error,"
                   " and is ignored.");
        }
     }
  }
  if(nError2>0) {
     if(nError2>1) {
        Warning("SetInput","%d output bins are not constrained by any data.",
                nError2);
     } else {
        Warning("SetInput","One output bins is not constrained by any data.");
     }
     // check whether data points with zero error are responsible
     if(oneOverZeroError<=0.0) {
        const Int_t *a_rows=fA->GetRowIndexArray();
        const Int_t *a_cols=fA->GetColIndexArray();
        for (Int_t col = 0; col <mAtV->GetNrows();col++) {
           if(mAtV->GetRowIndexArray()[col]==
              mAtV->GetRowIndexArray()[col+1]) {
              TString binlist("output bin ");
              binlist += fXToHist[col];
              binlist +=" depends on ignored input bins ";
              for(Int_t row=0;row<fA->GetNrows();row++) {
                 if(input->GetBinError(row + 1)>0.0) continue;
                 for(Int_t i=a_rows[row];i<a_rows[row+1];i++) {
                    if(a_cols[i]!=col) continue;
                    binlist +=" ";
                    binlist +=row;
                 }
              }
              Warning("SetInput","%s",binlist.Data());
           }
        }
     }
  }
  DeleteMatrix(&mAtV);

  return nError+10000*nError2;
}

Double_t TUnfold::DoUnfold(Double_t tau) {
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
   //     fTauSquared and those documented in DoUnfold(void)
   fTauSquared=tau*tau;
   return DoUnfold();
}

Int_t TUnfold::ScanLcurve(Int_t nPoint,
                          Double_t tauMin,Double_t tauMax,
			  TGraph **lCurve,TSpline **logTauX,
			  TSpline **logTauY) {
  // scan the L curve
  //   nPoint: number of points on the resulting curve
  //   tauMin: smallest tau value to study
  //   tauMax: largest tau value to study
  //   lCurve: the L curve as graph
  //   logTauX: output spline of x-coordinates vs tau for the L curve
  //   logTauY: output spline of y-coordinates vs tau for the L curve
  // return value: the coordinate number (0..nPoint-1) with the "best" choice
  //     of tau
  typedef std::map<Double_t,std::pair<Double_t,Double_t> > XYtau_t;
  XYtau_t curve;

  //==========================================================
  // algorithm:
  //  (1) do the unfolding for nPoint-1 points
  //      and store the results in the map
  //        curve
  //    (1a) store minimum and maximum tau to curve
  //    (1b) insert additional points, until nPoint-1 values
  //          have been calculated
  //
  //  (2) determine the best choice of tau
  //      do the unfolding for this point
  //      and store the result in
  //        curve
  //  (3) return the result in
  //       lCurve logTauX logTauY

  //==========================================================
  //  (1) do the unfolding for nPoint-1 points
  //      and store the results in
  //        curve
  //    (1a) store minimum and maximum tau to curve

  if((tauMin<=0)||(tauMax<=0.0)||(tauMin>=tauMax)) {
     // if the number of degrees of freedom is too small, create an error
     if(GetNdf()<=0) {
        Error("ScanLcurve","too few input bins, NDF<=0");  
     }
     // here no range is given, has to be determined automatically
     // the maximum tau is determined from the chi**2 values 
     // observed from unfolding without regulatisation

     // first unfolding, without regularisation
     DoUnfold(0.0);
     Double_t x0=GetLcurveX();
     Double_t y0=GetLcurveY();
     Info("ScanLcurve","logtau=-Infinity X=%lf Y=%lf",x0,y0);
     {
        // unfolding guess maximum tau and store it
        Double_t logTau=
           0.5*(TMath::Log10(fChi2A+3.*TMath::Sqrt(GetNdf()+1.0))
                -GetLcurveY());
        DoUnfold(TMath::Power(10.,logTau));
        curve[logTau]=std::make_pair(GetLcurveX(),GetLcurveY());
        Info("ScanLcurve","logtau=%lf X=%lf Y=%lf",
             logTau,GetLcurveX(),GetLcurveY());
     }
     if((*curve.begin()).second.first<x0) {
        // if the point at tau==0 seems numerically unstable,
        // try to find the minimum chi**2 as start value
        //
        // "unstable" means that there is a finite tau where the
        // unfolding chi**2 is smaller than for the case of no
        // regularisation. Ideally this shoukld never happen
        do {
           x0=GetLcurveX();
           Double_t logTau=(*curve.begin()).first-0.5;
           DoUnfold(TMath::Power(10.,logTau));
           curve[logTau]=std::make_pair(GetLcurveX(),GetLcurveY());
           Info("ScanLcurve","logtau=%lf X=%lf Y=%lf",
                logTau,GetLcurveX(),GetLcurveY());
        }
        while(((int)curve.size()<(nPoint-1)/2)&&
              ((*curve.begin()).second.first<x0));
     } else {
        // minimum tau is chosen such that it is less than
        // 1% different from the case of no regularusation
        // log10(1.01) = 0.00432

        // here, more than one point are insetred in necessary
        while(((int)curve.size()<nPoint-1)&&
              (((*curve.begin()).second.first-x0>0.00432)||
               ((*curve.begin()).second.second-y0>0.00432)||
               (curve.size()<2))) {
           Double_t logTau=(*curve.begin()).first-0.5;
           DoUnfold(TMath::Power(10.,logTau));
           curve[logTau]=std::make_pair(GetLcurveX(),GetLcurveY());
           Info("ScanLcurve","logtau=%lf X=%lf Y=%lf",
                logTau,GetLcurveX(),GetLcurveY());
        }
     }
  } else {
     Double_t logTauMin=TMath::Log10(tauMin);
     Double_t logTauMax=TMath::Log10(tauMax);
     if(nPoint>1) {
        // insert maximum tau
        DoUnfold(TMath::Power(10.,logTauMax));
        Info("ScanLcurve","logtau=%lf X=%lf Y=%lf",
             logTauMax,GetLcurveX(),GetLcurveY());
        curve[logTauMax]=std::make_pair(GetLcurveX(),GetLcurveY());
     }
     // insert minimum tau
     DoUnfold(TMath::Power(10.,logTauMin));
     Info("ScanLcurve","logtau=%lf X=%lf Y=%lf",
          logTauMin,GetLcurveX(),GetLcurveY());
     curve[logTauMin]=std::make_pair(GetLcurveX(),GetLcurveY());
  }


  //==========================================================
  //    (1b) insert additional points, until nPoint-1 values
  //          have been calculated

  while(int(curve.size())<nPoint-1) {
    // insert additional points, such that the sizes of the delta(XY) vectors
    // are getting smaller and smaller
    XYtau_t::const_iterator i0,i1;
    i0=curve.begin();
    i1=i0;
    Double_t logTau=(*i0).first;
    Double_t distMax=0.0;
    for(i1++;i1!=curve.end();i1++) {
      const std::pair<Double_t,Double_t> &xy0=(*i0).second;
      const std::pair<Double_t,Double_t> &xy1=(*i1).second;
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
    Info("ScanLcurve","logtau=%lf X=%lf Y=%lf",logTau,GetLcurveX(),GetLcurveY());
    curve[logTau]=std::make_pair(GetLcurveX(),GetLcurveY());
  }

  //==========================================================
  //  (2) determine the best choice of tau
  //      do the unfolding for this point
  //      and store the result in
  //        curve
  XYtau_t::const_iterator i0,i1;
  i0=curve.begin();
  i1=i0;
  i1++;
  Double_t logTauFin=(*i0).first;
  if( ((int)curve.size())<nPoint) {
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
    // find the maximum of the curvature
    // if the parameter iskip is non-zero, then iskip points are
    // ignored when looking for the largest curvature
    // (there are problems with the curvature determined from the first
    //  few points of splineX,splineY in the algorithm above)
    Int_t iskip=0;
    if(n>4) iskip=1;
    if(n>7) iskip=2;
    Double_t cCmax=cCi[iskip];
    Double_t cTmax=cTi[iskip];
    for(Int_t i=iskip;i<n-2-iskip;i++) {
      // find maximum on this spline section
      // check boundary conditions for x[i+1]
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
#ifdef DEBUG_LCURVE
    {
      TCanvas lcc;
      lcc.Divide(1,1);
      lcc.cd(1);
      splineC->Draw();
      lcc.SaveAs("splinec.ps");
    }
#endif
    delete splineC;
    delete[] cTi;
    delete[] cCi;
    logTauFin=cTmax;
    DoUnfold(TMath::Power(10.,logTauFin));
    Info("ScanLcurve","Result logtau=%lf X=%lf Y=%lf",
         logTauFin,GetLcurveX(),GetLcurveY());
    curve[logTauFin]=std::make_pair(GetLcurveX(),GetLcurveY());
  }


  //==========================================================
  //  (3) return the result in
  //       lCurve logTauX logTauY

  Int_t bestChoice=-1;
  if(curve.size()>0) {
    Double_t *x=new Double_t[curve.size()];
    Double_t *y=new Double_t[curve.size()];
    Double_t *logT=new Double_t[curve.size()];
    int n=0;
    for( XYtau_t::const_iterator i=curve.begin();i!=curve.end();i++) {
      if(logTauFin==(*i).first) {
        bestChoice=n;
      }
      x[n]=(*i).second.first;
      y[n]=(*i).second.second;
      logT[n]=(*i).first;
      n++;
    }
    if(lCurve) {
       (*lCurve)=new TGraph(n,x,y); 
       (*lCurve)->SetTitle("L curve");
   }
    if(logTauX) (*logTauX)=new TSpline3("log(chi**2)%log(tau)",logT,x,n);
    if(logTauY) (*logTauY)=new TSpline3("log(reg.cond)%log(tau)",logT,y,n);
    delete[] x;
    delete[] y;
    delete[] logT;
  }

  return bestChoice;
}


TH1D *TUnfold::GetOutput(const char *name,const char *title,
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
   GetOutput(out);

   return out;
}

TH1D *TUnfold::GetBias(const char *name,const char *title,
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

TH1D *TUnfold::GetFoldedOutput(const char *name,const char *title,
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

   TMatrixDSparse *AVxx=MultiplyMSparseMSparse(fA,fVxx);


   const Int_t *rows_A=fA->GetRowIndexArray();
   const Int_t *cols_A=fA->GetColIndexArray();
   const Double_t *data_A=fA->GetMatrixArray();
   const Int_t *rows_AVxx=AVxx->GetRowIndexArray();
   const Int_t *cols_AVxx=AVxx->GetColIndexArray();
   const Double_t *data_AVxx=AVxx->GetMatrixArray();
   
   for (Int_t i = 0; i < GetNy(); i++) {
      out->SetBinContent(i + 1, (*fAx) (i, 0));
      Double_t e2=0.0;
      Int_t index_a=rows_A[i];
      Int_t index_av=rows_AVxx[i];
      while((index_a<rows_A[i+1])&&(index_av<rows_AVxx[i])) {
         Int_t j_a=cols_A[index_a];
         Int_t j_av=cols_AVxx[index_av];
         if(j_a<j_av) {
            index_a++;
         } else if(j_a>j_av) {
            index_av++;
         } else {
            e2 += data_AVxx[index_av] * data_A[index_a];
            index_a++;
            index_av++;
         }
      }
      out->SetBinError(i + 1,TMath::Sqrt(e2));
   }
   DeleteMatrix(&AVxx);
   return out;
}

TH1D *TUnfold::GetInput(const char *name,const char *title,
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

   const Int_t *rows_Vyy=fVyy->GetRowIndexArray();
   const Int_t *cols_Vyy=fVyy->GetColIndexArray();
   const Double_t *data_Vyy=fVyy->GetMatrixArray();


   for (Int_t i = 0; i < GetNy(); i++) {
      out->SetBinContent(i + 1, (*fY) (i, 0));
      Double_t e=0.0;
      for(int index=rows_Vyy[i];index<rows_Vyy[i+1];index++) {
         if(cols_Vyy[index]==i) {
            e=TMath::Sqrt(data_Vyy[index]);
         }
      }
      out->SetBinError(i + 1, e);
   }
   return out;
}

TH2D *TUnfold::GetRhoIJ(const char *name,const char *title,
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
   GetRhoIJ(out);
   return out;
}

TH2D *TUnfold::GetEmatrix(const char *name,const char *title,
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
   GetEmatrix(out);

   return out;
}

TH1D *TUnfold::GetRhoI(const char *name,const char *title,
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
   GetRhoI(out);

   return out;
}

TH2D *TUnfold::GetLsquared(const char *name,const char *title,
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
   const Int_t *rows=fLsquared->GetRowIndexArray();
   const Int_t *cols=fLsquared->GetColIndexArray();
   const Double_t *data=fLsquared->GetMatrixArray();
   for (Int_t i = 0; i < GetNx(); i++) {
      for (Int_t cindex = rows[i]; cindex < rows[i+1]; cindex++) {
        Int_t j=cols[cindex];
        out->SetBinContent(fXToHist[i], fXToHist[j], fTauSquared * data[cindex]);
      }
   }

  return out;
}

void TUnfold::SetConstraint(TUnfold::EConstraint constraint) {
   // set type of constraint for the next unfolding
   if(fConstraint !=constraint) ClearResults();
   fConstraint=constraint;
   Info("TUnfold::SetConstraint","fConstraint=%d",fConstraint);
}

Double_t TUnfold::GetTau(void) const
{
   // return regularisation parameter
   return TMath::Sqrt(fTauSquared);
}

Double_t TUnfold::GetChi2L(void) const
{
   // return chi**2 contribution from regularisation conditions
   return fLXsquared*fTauSquared;
}

Int_t TUnfold::GetNpar(void) const
{
   // return number of parameters
   return GetNx();
}

Double_t TUnfold::GetLcurveX(void) const {
  // return value on x axis of L curve
  return TMath::Log10(fChi2A);
}

Double_t TUnfold::GetLcurveY(void) const {
  // return value on y axis of L curve
  return TMath::Log10(fLXsquared);
}

void TUnfold::GetOutput(TH1 *output,const Int_t *binMap) const {
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

   const Int_t *rows_Vxx=fVxx->GetRowIndexArray();
   const Int_t *cols_Vxx=fVxx->GetColIndexArray();
   const Double_t *data_Vxx=fVxx->GetMatrixArray();

   Int_t binMapSize = fHistToX.GetSize();
   for(Int_t i=0;i<binMapSize;i++) {
      Int_t destBinI=binMap ? binMap[i] : i;
      Int_t srcBinI=fHistToX[i];
      if((destBinI>=0)&&(destBinI<nbin+2)&&(srcBinI>=0)) {
         c[destBinI]+=(*fX)(srcBinI,0);
         // here we loop over the columns of the error matrix
         //   j: counts histogram bins
         //   index: counts sparse matrix index
         // the algorithm makes use of the fact that fHistToX is ordered
         Int_t j=0;
         Int_t index_vxx=rows_Vxx[srcBinI];
         while((j<binMapSize)&&(index_vxx<rows_Vxx[srcBinI+1])) {
            Int_t destBinJ=binMap ? binMap[j] : j;
            if(destBinI!=destBinJ) {
               // only diagonal elements are calculated
               j++;
            } else {
               Int_t srcBinJ=fHistToX[j];
               if(srcBinJ<0) {
                  // bin is not used, check next bin
                  j++;
               } else {
                  if(cols_Vxx[index_vxx]<srcBinJ) {
                     // index is too small
                     index_vxx++;
                  } else if(cols_Vxx[index_vxx]>srcBinJ) {
                     // index is too large, skip bin
                     j++;
                  } else {
                     // add this bin
                     e2[destBinI]+= data_Vxx[index_vxx];
                     j++;
                     index_vxx++;
                  }
               }
            }
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

void TUnfold::ErrorMatrixToHist(TH2 *ematrix,const TMatrixDSparse *emat,
                                const Int_t *binMap,Bool_t doClear) const {

   // get an error matrix, cumulated over several bins
   //   ematrix: output error matrix histogram
   //   emat: error matrix
   //   binMap: for each bin of the original output distribution
   //           specify the destination bin. A value of -1 means that the bin
   //           is discarded. 0 means underflow bin, 1 first bin, ...
   //        binMap[0] : destination of underflow bin
   //        binMap[1] : destination of first bin
   //          ...
   Int_t nbin=ematrix->GetNbinsX();
   if(doClear) {
      for(Int_t i=0;i<nbin+2;i++) {
         for(Int_t j=0;j<nbin+2;j++) {
            ematrix->SetBinContent(i,j,0.0);
            ematrix->SetBinError(i,j,0.0);
         }
      }
   }

   if(emat) {
      const Int_t *rows_emat=emat->GetRowIndexArray();
      const Int_t *cols_emat=emat->GetColIndexArray();
      const Double_t *data_emat=emat->GetMatrixArray();

      Int_t binMapSize = fHistToX.GetSize();
      for(Int_t i=0;i<binMapSize;i++) {
         Int_t destBinI=binMap ? binMap[i] : i;
         Int_t srcBinI=fHistToX[i];
         if((destBinI>=0)&&(destBinI<nbin+2)&&(srcBinI>=0)) {
            // here we loop over the columns of the source matrix
            //   j: counts histogram bins
            //   index: counts sparse matrix index
            // the algorithm makes use of the fact that fHistToX is ordered
            Int_t j=0;
            Int_t index_vxx=rows_emat[srcBinI];
            while((j<binMapSize)&&(index_vxx<rows_emat[srcBinI+1])) {
               Int_t destBinJ=binMap ? binMap[j] : j;
               Int_t srcBinJ=fHistToX[j];
               if((destBinJ<0)||(destBinJ>=nbin+2)||(srcBinJ<0)) {
                  // check next bin
                  j++;
               } else {
                  if(cols_emat[index_vxx]<srcBinJ) {
                     // index is too small
                     index_vxx++;
                  } else if(cols_emat[index_vxx]>srcBinJ) {
                     // index is too large, skip bin
                     j++;
                  } else {
                     // add this bin
                     Double_t e2= ematrix->GetBinContent(destBinI,destBinJ)
                        + data_emat[index_vxx];
                     ematrix->SetBinContent(destBinI,destBinJ,e2);
                     j++;
                     index_vxx++;
                  }
               }
            }
         }
      }
   }
}

void TUnfold::GetEmatrix(TH2 *ematrix,const Int_t *binMap) const {
   // get output error matrix, cumulated over several bins
   //   ematrix: output error matrix histogram
   //   binMap: for each bin of the original output distribution
   //           specify the destination bin. A value of -1 means that the bin
   //           is discarded. 0 means underflow bin, 1 first bin, ...
   //        binMap[0] : destination of underflow bin
   //        binMap[1] : destination of first bin
   //          ...
   ErrorMatrixToHist(ematrix,fVxx,binMap,kTRUE);
}

Double_t TUnfold::GetRhoI(TH1 *rhoi,TH2 *ematrixinv,const Int_t *binMap) const {
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

   // The unfolding is based on a root TH2D histogram.
   // Internally, these histogram bins are mapped to Matrix cols:
   //    fHistToX[i]   is the matrix-row corresponding to matrix-bin i
   //    fXToHist[i]   is the matrix-bin corresponding to matrix-col i
   //
   // for the output, another level of mapping is added, such that
   // a matrix-bin is mapped to one or more output-bins
   //    binMap[i]     is the output-bin corresponding to matrix-bin i

   // below, the matrix-rows are summed to emat-rows, then the emat-rows
   // are stored to output-bins

   // n counts the number of active bins in the output histogram
   Int_t n=0;
   Int_t *outputToEmat=new Int_t[nbin+2];
   Int_t *ematToOutput=new Int_t[nbin+2];
   Int_t *vxxToEmat=new Int_t[GetNx()];
   for(Int_t i=0;i<nbin+2;i++) {
      outputToEmat[i]=-1;
   }
   for(Int_t i=0;i<GetNx();i++) {
      Int_t matrix_bin=fXToHist[i];
      Int_t output_bin=binMap ? binMap[matrix_bin] : matrix_bin;
      if((output_bin>=0)&&(output_bin<nbin+2)) {
         // matrix-col i will be stored to output_bin
         if(outputToEmat[output_bin]<0) {
            // new bin n
            outputToEmat[output_bin]=n;
            ematToOutput[n]=output_bin;
            n++;
         }
         vxxToEmat[i]=outputToEmat[output_bin];
      } else {
         vxxToEmat[i]=-1;
      }
   }
   delete[] outputToEmat;

   // now:
   //   create new error matrix emat(n,n)
   //   sum all bins of Vxx into the new matrix, using the mapping
   //    vxxToEmat[]
   // later:
   //   pack emat(n,n) to the histogram, using the mapping
   //    ematToOutput

   // set up reduced error matrix
   TMatrixD e(n,n);
   const Int_t *rows_Vxx=fVxx->GetRowIndexArray();
   const Int_t *cols_Vxx=fVxx->GetColIndexArray();
   const Double_t *data_Vxx=fVxx->GetMatrixArray();
   for(Int_t i=0;i<GetNx();i++) {
      Int_t ie=vxxToEmat[i];
      if(ie<0) continue;
      for(int index_vxx=rows_Vxx[i];index_vxx<rows_Vxx[i+1];index_vxx++) {
         Int_t je=vxxToEmat[cols_Vxx[index_vxx]];
         if(je<0) continue;
         e(ie,je) += data_Vxx[index_vxx];
      }
   }
   delete[] vxxToEmat;

   // invert error matrix
   TMatrixD einv(e);
   InvertMConditioned(&einv);

   Double_t rhoMax=0.0;
   for(Int_t i=0;i<n;i++) {
      Int_t i_out=ematToOutput[i];
      Double_t rho=1.-1./(einv(i,i)*e(i,i));
      if(rho>=0.0) rho=TMath::Sqrt(rho);
      else rho=-TMath::Sqrt(-rho);
      if(rho>rhoMax) {
         rhoMax = rho;
      }
      rhoi->SetBinContent(i_out,rho);
      if(ematrixinv) {
         for(Int_t j=0;j<n;j++) {
            ematrixinv->SetBinContent(i_out,ematToOutput[j],einv(i,j));
         }
      }
   }
   delete[] ematToOutput;

   return rhoMax;
}

void TUnfold::GetRhoIJ(TH2 *rhoij,const Int_t *binMap) const {
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
