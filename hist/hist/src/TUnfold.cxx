// Author: Stefan Schmitt
// DESY, 13/10/08

//  Version 17.1, bug fixes in GetFoldedOutput, GetOutput
//
//  History:
//    Version 17.0, option to specify an error matrix with SetInput(), new ScanRho() method
//    Version 16.2, in parallel to bug-fix in TUnfoldSys
//    Version 16.1, fix bug with error matrix in case kEConstraintArea is used
//    Version 16.0, fix calculation of global correlations, improved error messages
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
//  TUnfold is used to decompose a measurement y into several sources x
//  given the measurement uncertainties and a matrix of migrations A
//
//  *************************************************************
//  * For most applications, it is better to use TUnfoldDensity *
//  * instead of using TUnfoldSys or TUnfold                    *
//  *************************************************************
//
//  If you use this software, please consider the following citation
//       S.Schmitt, JINST 7 (2012) T10003 [arXiv:1205.6201]
//
//  More documentation and updates are available on
//      http://www.desy.de/~sschmitt
//
//
//  A short summary of the algorithm is given in the following:
//
//  the "best" x matching the measurement y within errors
//  is determined by minimizing the function
//    L1+L2+L3
//
//  where
//   L1 = (y-Ax)# Vyy^-1 (y-Ax)
//   L2 = tau^2 (L(x-x0))# L(x-x0)
//   L3 = lambda sum_i(y_i -(Ax)_i)
//
//  [the notation # means that the matrix is transposed ]
//
//  The term L1 is familiar from a least-square minimisation
//  The term L2 defines the regularisation (smootheness condition on x),
//    where the parameter tau^2 gives the strength of teh regularisation
//  The term L3 is an optional area constraint with Lagrangian parameter
//    lambda, ensuring that that the normalisation of x is consistent with the
//    normalisation of y
//
//  The method can be applied to a very large number of problems,
//  where the measured distribution y is a linear superposition
//  of several Monte Carlo shapes
//
// Input from measurement:
// -----------------------
//   y: vector of measured quantities  (dimension ny)
//   Vyy: covariance matrix for y (dimension ny x ny)
//        in many cases V is diagonal and calculated from the errors of y
//
// From simulation:
// ----------------
//   A: migration matrix               (dimension ny x nx)
//
// Result
// ------
//   x: unknown underlying distribution (dimension nx)
//      The error matrix of x, V_xx, is also determined
//
// Regularisation
// --------------
//   tau: parameter, defining the regularisation strength
//   L: matrix of regularisation conditions (dimension nl x nx)
//        depends on the structure of the input data
//   x0: bias distribution, from simulation
//
// Preservation of the area
// ------------------------
//   lambda: lagrangian multiplier
//   y_i: one component of the vector y
//   (Ax)_i: one component of the vector Ax
//
//
// Determination of the unfolding result x:
//
//   (a) not constrained: minimisation is performed as a function of x
//         for fixed lambda=0
//  or
//   (b) constrained: stationary point is found as a function of x and lambda
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
// Note:
//   For most applications it is better to use the derived class
//     TUnfoldSys or even better TUnfoldDensity
//
//  TUnfoldSys extends the functionality of TUnfold
//             such that systematic errors are propagated to teh result
//             and that the unfolding can be done with proper background
//             subtraction
//
//  TUnfoldDensity extends further the functionality of TUnfoldSys
//                 complex binning schemes are supported
//                 The binning of input histograms is handeld consistently:
//                  (1) the regularisation may be done by density,
//                      i.e respecting the bin widths
//                  (2) methods are provided which preserve the proper binning
//                      of the result histograms
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
// Proper choice of tau
// ====================
// One of the difficult questions is about the choice of tau.
// The method implemented in TUnfold is the L-curve method:
// a two-dimensional curve is plotted
//   x-axis: log10(chisquare)
//   y-axis: log10(regularisation condition)
// In many cases this curve has an L-shape. The best choice of tau is in the
// kink of the L
//
// Within TUnfold a simple version of the L-curve analysis is available.
// It tests a given number of points in a predefined tau-range and searches
// for the maximum of the curvature in the L-curve (kink position).
// if no tau range is given, the range of the scan is determined automatically
//
//  A nice overview of the L-curve method is given in:
//   The L-curve and Its Use in the Numerical Treatment of Inverse Problems
//   (2000) by P. C. Hansen, in Computational Inverse Problems in
//   Electrocardiology, ed. P. Johnston,
//   Advances in Computational Bioengineering
//   http://www.imm.dtu.dk/~pch/TR/Lcurve.ps
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
// kRegModeSize is the regularisation scheme which often is found in
// literature. The second derivative is often named curvature.
// Sometimes the bias is not discussed,  equivalent to a bias scale factor
// of zero.
//
// The regularisation schemes kRegModeDerivative and
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
// Note, the class TUnfoldDensity provides an automatic setup of complex
// regularisation schemes
//
// For ultimate flexibility, the regularisation condition can be set on each
// bin individually
//  -> give  kRegModeNone  in the constructor and use
//      RegularizeSize()        regularize one bin
//      RegularizeDerivative()  regularize the slope given by two bins
//      RegularizeCurvature()   regularize the curvature given by three bins
//      AddRegularisationCondition()
//                              define an arbitrary regulatisation condition
//
///////////////////////////////////////////////////////////////////////////

/*
 This file is part of TUnfold.

 TUnfold is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 TUnfold is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with TUnfold.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <iostream>
#include <TMatrixD.h>
#include <TMatrixDSparse.h>
#include <TMatrixDSym.h>
#include <TMatrixDSymEigen.h>
#include <TMath.h>
#include "TUnfold.h"

#include <map>
#include <vector>

// this option saves the spline of the L curve curvature to a file
// named splinec.ps for debugging

//#define DEBUG
//#define DEBUG_DETAIL
//#define DEBUG_LCURVE
//#define FORCE_EIGENVALUE_DECOMPOSITION

#ifdef DEBUG_LCURVE
#include <TCanvas.h>
#endif

ClassImp(TUnfold)
//______________________________________________________________________________

const char *TUnfold::GetTUnfoldVersion(void)
{
   return TUnfold_VERSION;
}


void TUnfold::InitTUnfold(void)
{
   // reset all data members
   fXToHist.Set(0);
   fHistToX.Set(0);
   fSumOverY.Set(0);
   fA = 0;
   fL = 0;
   fVyy = 0;
   fY = 0;
   fX0 = 0;
   fTauSquared = 0.0;
   fBiasScale = 0.0;
   fNdf = 0;
   fConstraint = kEConstraintNone;
   fRegMode = kRegModeNone;
   // output
   fVyyInv = 0;
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
   fEpsMatrix=1.E-13;
   fIgnoredBins=0;
}

void TUnfold::DeleteMatrix(TMatrixD **m)
{
   if(*m) delete *m;
   *m=0;
}

void TUnfold::DeleteMatrix(TMatrixDSparse **m)
{
   if(*m) delete *m;
   *m=0;
}

void TUnfold::ClearResults(void)
{
   // delete old results (if any)
   // this function is virtual, so derived classes may implement their own
   // method to flag results as non-valid

   // note: the inverse of the input covariance is not cleared
   //       because it does not change until the input is changed

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
   //     fL: regularisation conditions
   //     fTauSquared: regularisation strength
   //     fConstraint: whether the constraint is applied
   // Data members modified:
   //     fVyyInv: inverse of input data covariance matrix
   //     fNdf: number of dgerres of freedom
   //     fEinv: inverse of the matrix needed for unfolding calculations
   //     fE:    the matrix needed for unfolding calculations
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

   // get pseudo-inverse matrix Vyyinv and NDF
   if(!fVyyInv) {
      GetInputInverseEmatrix(0);
      if(fConstraint != kEConstraintNone) {
         fNdf--;
      }
   }
   //
   // get matrix
   //              T
   //            fA fV  = mAt_V
   //
   TMatrixDSparse *AtVyyinv=MultiplyMSparseTranspMSparse(fA,fVyyInv);
   //
   // get
   //       T
   //     fA fVyyinv fY + fTauSquared fBiasScale Lsquared fX0 = rhs
   //
   TMatrixDSparse *rhs=MultiplyMSparseM(AtVyyinv,fY);
   TMatrixDSparse *lSquared=MultiplyMSparseTranspMSparse(fL,fL);
   if (fBiasScale != 0.0) {
      TMatrixDSparse *rhs2=MultiplyMSparseM(lSquared,fX0);
      AddMSparse(rhs, fTauSquared * fBiasScale ,rhs2);
      DeleteMatrix(&rhs2);
   }

   //
   // get matrix
   //              T
   //           (fA fV)fA + fTauSquared*fLsquared  = fEinv
   fEinv=MultiplyMSparseMSparse(AtVyyinv,fA);
   AddMSparse(fEinv,fTauSquared,lSquared);

   //
   // get matrix
   //             -1
   //        fEinv    = fE
   //
   Int_t rank=0;
   fE = InvertMSparseSymmPos(fEinv,&rank);
   if(rank != GetNx()) {
      Warning("DoUnfold","rank of matrix E %d expect %d",rank,GetNx());
   }

   //
   // get result
   //        fE rhs  = x
   //
   TMatrixDSparse *xSparse=MultiplyMSparseMSparse(fE,rhs);
   fX = new TMatrixD(*xSparse);
   DeleteMatrix(&rhs);
   DeleteMatrix(&xSparse);

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
         Fatal("Unfold","epsilon#Eepsilon has dimension %d != 1",
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
   //   fDXDY * Vyy * fDXDY#
   TMatrixDSparse *fDXDYVyy = MultiplyMSparseMSparse(fDXDY,fVyy);
   fVxx = MultiplyMSparseMSparseTranspVector(fDXDYVyy,fDXDY,0);

   DeleteMatrix(&fDXDYVyy);

   //
   // get result
   //        fA x  =  fAx
   //
   fAx = MultiplyMSparseM(fA,fX);

   //
   // calculate chi**2 etc

   // chi**2 contribution from (y-Ax)V(y-Ax)
   TMatrixD dy(*fY, TMatrixD::kMinus, *fAx);
   TMatrixDSparse *VyyinvDy=MultiplyMSparseM(fVyyInv,&dy);

   const Int_t *VyyinvDy_rows=VyyinvDy->GetRowIndexArray();
   const Double_t *VyyinvDy_data=VyyinvDy->GetMatrixArray();
   fChi2A=0.0;
   for(Int_t iy=0;iy<VyyinvDy->GetNrows();iy++) {
      if(VyyinvDy_rows[iy]<VyyinvDy_rows[iy+1]) {
         fChi2A += VyyinvDy_data[VyyinvDy_rows[iy]]*dy(iy,0);
      }
   }
   TMatrixD dx( fBiasScale * (*fX0), TMatrixD::kMinus,(*fX));
   TMatrixDSparse *LsquaredDx=MultiplyMSparseM(lSquared,&dx);
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
         Fatal("Unfold",
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
   DeleteMatrix(&lSquared);

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

   rank=0;
   fVxxInv = InvertMSparseSymmPos(fVxx,&rank);
   if(rank != GetNx()) {
      Warning("DoUnfold","rank of output covariance is %d expect %d",
              rank,GetNx());
   }

   TVectorD VxxInvDiag(fVxxInv->GetNrows());
   const Int_t *VxxInv_rows=fVxxInv->GetRowIndexArray();
   const Int_t *VxxInv_cols=fVxxInv->GetColIndexArray();
   const Double_t *VxxInv_data=fVxxInv->GetMatrixArray();
   for (int ix = 0; ix < fVxxInv->GetNrows(); ix++) {
      for(int ik=VxxInv_rows[ix];ik<VxxInv_rows[ix+1];ik++) {
         if(ix==VxxInv_cols[ik]) {
            VxxInvDiag(ix)=VxxInv_data[ik];
         }
      }
   }

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
            1. - 1. / (VxxInvDiag(ix) * Vxx_data[ik]);
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

   return fRhoMax;
}

TMatrixDSparse *TUnfold::CreateSparseMatrix
(Int_t nrow,Int_t ncol,Int_t nel,Int_t *row,Int_t *col,Double_t *data) const
{
   // create a sparse matri
   //   nrow,ncol : dimension of the matrix
   //   nel: number of non-zero elements
   //   row[nel],col[nel],data[nel] : indices and data of the non-zero elements
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


TMatrixDSparse *TUnfold::MultiplyMSparseTranspMSparse
(const TMatrixDSparse *a,const TMatrixDSparse *b) const
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
 const TMatrixTBase<Double_t> *v) const
{
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
                         const TMatrixDSparse *src) const
{
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
         if(result_data[n] !=0.0) {
            if(!TMath::Finite(result_data[n])) {
               Fatal("AddMSparse","Nan detected %d %d %d",
                     row,i_dest,i_src);
            }
            n++;
         }
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

TMatrixDSparse *TUnfold::InvertMSparseSymmPos
(const TMatrixDSparse *A,Int_t *rankPtr) const
{
   // get the inverse or pseudo-inverse of a sparse matrix
   //    A: the original matrix
   //    rank:
   //      if(rankPtr==0)
   //          rerturn the inverse if it exists, or return NULL
   //      else
   //          return the pseudo-invers
   //          and return the rank of the matrix in *rankPtr
   // the matrix inversion is optimized for the case
   // where a large submatrix of A is diagonal

   if(A->GetNcols()!=A->GetNrows()) {
      Fatal("InvertMSparseSymmPos","inconsistent matrix row/col %d!=%d",
            A->GetNcols(),A->GetNrows());
   }

   Bool_t *isZero=new Bool_t[A->GetNrows()];
   const Int_t *a_rows=A->GetRowIndexArray();
   const Int_t *a_cols=A->GetColIndexArray();
   const Double_t *a_data=A->GetMatrixArray();

   // determine diagonal elements
   //  Aii: diagonals of A
   Int_t iDiagonal=0;
   Int_t iBlock=A->GetNrows();
   Bool_t isDiagonal=kTRUE;
   TVectorD aII(A->GetNrows());
   Int_t nError=0;
   for(Int_t iA=0;iA<A->GetNrows();iA++) {
      for(Int_t indexA=a_rows[iA];indexA<a_rows[iA+1];indexA++) {
         Int_t jA=a_cols[indexA];
         if(iA==jA) {
            if(!(a_data[indexA]>=0.0)) nError++;
            aII(iA)=a_data[indexA];
         }
      }
   }
   if(nError>0) {
      delete [] isZero;
      Fatal("InvertMSparseSymmPos",
            "Matrix has %d negative elements on the diagonal", nError);
      return 0;
   }

   // reorder matrix such that the largest block of zeros is swapped
   // to the lowest indices
   // the result of this ordering:
   //  swap[i] : for index i point to the row in A
   //  swapBack[iA] : for index iA in A point to the swapped index i
   // the indices are grouped into three groups
   //   0..iDiagonal-1 : these rows only have diagonal elements
   //   iDiagonal..iBlock : these rows contain a diagonal block matrix
   //
   Int_t *swap=new Int_t[A->GetNrows()];
   for(Int_t j=0;j<A->GetNrows();j++) swap[j]=j;
   for(Int_t iStart=0;iStart<iBlock;iStart++) {
      Int_t nZero=0;
      for(Int_t i=iStart;i<iBlock;i++) {
         Int_t iA=swap[i];
         Int_t n=A->GetNrows()-(a_rows[iA+1]-a_rows[iA]);
         if(n>nZero) {
            Int_t tmp=swap[iStart];
            swap[iStart]=swap[i];
            swap[i]=tmp;
            nZero=n;
         }
      }
      for(Int_t i=0;i<A->GetNrows();i++) isZero[i]=kTRUE;
      Int_t iA=swap[iStart];
      for(Int_t indexA=a_rows[iA];indexA<a_rows[iA+1];indexA++) {
         Int_t jA=a_cols[indexA];
         isZero[jA]=kFALSE;
         if(iA!=jA) {
            isDiagonal=kFALSE;
         }
      }
      if(isDiagonal) {
         iDiagonal=iStart+1;
      } else {
         for(Int_t i=iStart+1;i<iBlock;) {
            if(isZero[swap[i]]) {
               i++;
            } else {
               iBlock--;
               Int_t tmp=swap[iBlock];
               swap[iBlock]=swap[i];
               swap[i]=tmp;
            }
         }
      }
   }

   // for tests uncomment this:
   //  iBlock=iDiagonal;


   // conditioning of the iBlock part
#ifdef CONDITION_BLOCK_PART
   Int_t nn=A->GetNrows()-iBlock;
   for(int inc=(nn+1)/2;inc;inc /= 2) {
      for(int i=inc;i<nn;i++) {
         int itmp=swap[i+iBlock];
         int j;
         for(j=i;(j>=inc)&&(aII(swap[j-inc+iBlock])<aII(itmp));j -=inc) {
            swap[j+iBlock]=swap[j-inc+iBlock];
         }
         swap[j+iBlock]=itmp;
      }
   }
#endif
   // construct array for swapping back
   Int_t *swapBack=new Int_t[A->GetNrows()];
   for(Int_t i=0;i<A->GetNrows();i++) {
      swapBack[swap[i]]=i;
   }
#ifdef DEBUG_DETAIL
   for(Int_t i=0;i<A->GetNrows();i++) {
      std::cout<<"    "<<i<<" "<<swap[i]<<" "<<swapBack[i]<<"\n";
   }
   std::cout<<"after sorting\n";
   for(Int_t i=0;i<A->GetNrows();i++) {
      if(i==iDiagonal) std::cout<<"iDiagonal="<<i<<"\n";
      if(i==iBlock) std::cout<<"iBlock="<<i<<"\n";
      std::cout<<" "<<swap[i]<<" "<<aII(swap[i])<<"\n";
   }
   {
      // sanity check:
      // (1) sub-matrix swapped[0]..swapped[iDiagonal]
      //       must not contain off-diagonals
      // (2) sub-matrix swapped[0]..swapped[iBlock-1] must be diagonal
      Int_t nDiag=0;
      Int_t nError1=0;
      Int_t nError2=0;
      Int_t nBlock=0;
      Int_t nNonzero=0;
      for(Int_t i=0;i<A->GetNrows();i++) {
         Int_t row=swap[i];
         for(Int_t indexA=a_rows[row];indexA<a_rows[row+1];indexA++) {
            Int_t j=swapBack[a_cols[indexA]];
            if(i==j) nDiag++;
            else if((i<iDiagonal)||(j<iDiagonal)) nError1++;
            else if((i<iBlock)&&(j<iBlock)) nError2++;
            else if((i<iBlock)||(j<iBlock)) nBlock++;
            else nNonzero++;
         }
      }
      if(nError1+nError2>0) {
         Fatal("InvertMSparseSymmPos","sparse matrix analysis failed %d %d %d %d %d",
               iDiagonal,iBlock,A->GetNrows(),nError1,nError2);
      }
   }
#endif
#ifdef DEBUG
   Info("InvertMSparseSymmPos","iDiagonal=%d iBlock=%d nRow=%d",
        iDiagonal,iBlock,A->GetNrows());
#endif

   //=============================================
   // matrix inversion starts here
   //
   //  the matrix is split into several parts
   //         D1  0   0
   //  A = (  0   D2  B# )
   //         0   B   C
   //
   //  D1,D2 are diagonal matrices
   //
   //  first, the D1 part is inverted by calculating 1/D1
   //   dim(D1)= iDiagonal
   //  next, the other parts are inverted using Schur complement
   //
   //           1/D1   0    0
   //  Ainv = (  0     E    G#   )
   //            0     G    F^-1
   //
   //  where F = C + (-B/D2) B#
   //        G = (F^-1) (-B/D2)
   //        E = 1/D2 + (-B#/D2) G)
   //
   //  the inverse of F is calculated using a Cholesky decomposition
   //
   // Error handling:
   // (a) if rankPtr==0: user requests inverse
   //
   //    if D1 is not strictly positive, return NULL
   //    if D2 is not strictly positive, return NULL
   //    if F is not strictly positive (Cholesky decomposition failed)
   //           return NULL
   //    else
   //      return Ainv as defined above
   //
   // (b) if rankPtr !=0: user requests pseudo-inverse
   //    if D2 is not strictly positive or F is not strictly positive
   //      calculate singular-value decomposition of
   //          D2   B#
   //         (       ) = O D O^-1
   //           B   C
   //      if some eigenvalues are negative, return NULL
   //      else calculate pseudo-inverse
   //        *rankPtr = rank(D1)+rank(D)
   //    else
   //      calculate pseudo-inverse of D1  (D1_ii==0) ? 0 : 1/D1_ii
   //      *rankPtr=rank(D1)+nrow(D2)+nrow(C)
   //      return Ainv as defined above

   Double_t *rEl_data=new Double_t[A->GetNrows()*A->GetNrows()];
   Int_t *rEl_col=new Int_t[A->GetNrows()*A->GetNrows()];
   Int_t *rEl_row=new Int_t[A->GetNrows()*A->GetNrows()];
   Int_t rNumEl=0;

   //====================================================
   // invert D1
   Int_t rankD1=0;
   for(Int_t i=0;i<iDiagonal;++i) {
      Int_t iA=swap[i];
      if(aII(iA)>0.0) {
         rEl_col[rNumEl]=iA;
         rEl_row[rNumEl]=iA;
         rEl_data[rNumEl]=1./aII(iA);
         ++rankD1;
         ++rNumEl;
      }
   }
   if((!rankPtr)&&(rankD1!=iDiagonal)) {
      Fatal("InvertMSparseSymmPos",
            "diagonal part 1 has rank %d != %d, matrix can not be inverted",
            rankD1,iDiagonal);
      rNumEl=-1;
   }


   //====================================================
   // invert D2
   Int_t nD2=iBlock-iDiagonal;
   TMatrixDSparse *D2inv=0;
   if((rNumEl>=0)&&(nD2>0)) {
      Double_t *D2inv_data=new Double_t[nD2];
      Int_t *D2inv_col=new Int_t[nD2];
      Int_t *D2inv_row=new Int_t[nD2];
      Int_t D2invNumEl=0;
      for(Int_t i=0;i<nD2;++i) {
         Int_t iA=swap[i+iDiagonal];
         if(aII(iA)>0.0) {
            D2inv_col[D2invNumEl]=i;
            D2inv_row[D2invNumEl]=i;
            D2inv_data[D2invNumEl]=1./aII(iA);
            ++D2invNumEl;
         }
      }
      if(D2invNumEl==nD2) {
         D2inv=CreateSparseMatrix(nD2,nD2,D2invNumEl,D2inv_row,D2inv_col,
                                  D2inv_data);
      } else if(!rankPtr) {
         Fatal("InvertMSparseSymmPos",
               "diagonal part 2 has rank %d != %d, matrix can not be inverted",
               D2invNumEl,nD2);
         rNumEl=-2;
      }
      delete [] D2inv_data;
      delete [] D2inv_col;
      delete [] D2inv_row;
   }

   //====================================================
   // invert F

   Int_t nF=A->GetNrows()-iBlock;
   TMatrixDSparse *Finv=0;
   TMatrixDSparse *B=0;
   TMatrixDSparse *minusBD2inv=0;
   if((rNumEl>=0)&&(nF>0)&&((nD2==0)||D2inv)) {
      // construct matrices F and B
      Int_t nFmax=nF*nF;
      Double_t epsilonF2=fEpsMatrix;
      Double_t *F_data=new Double_t[nFmax];
      Int_t *F_col=new Int_t[nFmax];
      Int_t *F_row=new Int_t[nFmax];
      Int_t FnumEl=0;

      Int_t nBmax=nF*(nD2+1);
      Double_t *B_data=new Double_t[nBmax];
      Int_t *B_col=new Int_t[nBmax];
      Int_t *B_row=new Int_t[nBmax];
      Int_t BnumEl=0;

      for(Int_t i=0;i<nF;i++) {
         Int_t iA=swap[i+iBlock];
         for(Int_t indexA=a_rows[iA];indexA<a_rows[iA+1];indexA++) {
            Int_t jA=a_cols[indexA];
            Int_t j0=swapBack[jA];
            if(j0>=iBlock) {
               Int_t j=j0-iBlock;
               F_row[FnumEl]=i;
               F_col[FnumEl]=j;
               F_data[FnumEl]=a_data[indexA];
               FnumEl++;
            } else if(j0>=iDiagonal) {
               Int_t j=j0-iDiagonal;
               B_row[BnumEl]=i;
               B_col[BnumEl]=j;
               B_data[BnumEl]=a_data[indexA];
               BnumEl++;
            }
         }
      }
      TMatrixDSparse *F=0;
      if(FnumEl) {
#ifndef FORCE_EIGENVALUE_DECOMPOSITION
         F=CreateSparseMatrix(nF,nF,FnumEl,F_row,F_col,F_data);
#endif
      }
      delete [] F_data;
      delete [] F_col;
      delete [] F_row;
      if(BnumEl) {
         B=CreateSparseMatrix(nF,nD2,BnumEl,B_row,B_col,B_data);
      }
      delete [] B_data;
      delete [] B_col;
      delete [] B_row;

      if(B && D2inv) {
         minusBD2inv=MultiplyMSparseMSparse(B,D2inv);
         if(minusBD2inv) {
            Int_t mbd2_nMax=minusBD2inv->GetRowIndexArray()
            [minusBD2inv->GetNrows()];
            Double_t *mbd2_data=minusBD2inv->GetMatrixArray();
            for(Int_t i=0;i<mbd2_nMax;i++) {
               mbd2_data[i] = -  mbd2_data[i];
            }
         }
      }
      if(minusBD2inv && F) {
         TMatrixDSparse *minusBD2invBt=
         MultiplyMSparseMSparseTranspVector(minusBD2inv,B,0);
         AddMSparse(F,1.,minusBD2invBt);
         DeleteMatrix(&minusBD2invBt);
      }

      if(F) {
         // cholesky decomposition with preconditioning
         const Int_t *f_rows=F->GetRowIndexArray();
         const Int_t *f_cols=F->GetColIndexArray();
         const Double_t *f_data=F->GetMatrixArray();
         // cholesky-type decomposition of F
         TMatrixD c(nF,nF);
         Int_t nErrorF=0;
         for(Int_t i=0;i<nF;i++) {
            for(Int_t indexF=f_rows[i];indexF<f_rows[i+1];indexF++) {
               if(f_cols[indexF]>=i) c(f_cols[indexF],i)=f_data[indexF];
            }
            // calculate diagonal element
            Double_t c_ii=c(i,i);
            for(Int_t j=0;j<i;j++) {
               Double_t c_ij=c(i,j);
               c_ii -= c_ij*c_ij;
            }
            if(c_ii<=0.0) {
               nErrorF++;
               break;
            }
            c_ii=TMath::Sqrt(c_ii);
            c(i,i)=c_ii;
            // off-diagonal elements
            for(Int_t j=i+1;j<nF;j++) {
               Double_t c_ji=c(j,i);
               for(Int_t k=0;k<i;k++) {
                  c_ji -= c(i,k)*c(j,k);
               }
               c(j,i) = c_ji/c_ii;
            }
         }
         // check condition of dInv
         if(!nErrorF) {
            Double_t dmin=c(0,0);
            Double_t dmax=dmin;
            for(Int_t i=1;i<nF;i++) {
               dmin=TMath::Min(dmin,c(i,i));
               dmax=TMath::Max(dmax,c(i,i));
            }
#ifdef DEBUG
            std::cout<<"dmin,dmax: "<<dmin<<" "<<dmax<<"\n";
#endif
            if(dmin/dmax<epsilonF2*nF) nErrorF=2;
         }
         if(!nErrorF) {
            // here: F = c c#
            // construct inverse of c
            TMatrixD cinv(nF,nF);
            for(Int_t i=0;i<nF;i++) {
               cinv(i,i)=1./c(i,i);
            }
            for(Int_t i=0;i<nF;i++) {
               for(Int_t j=i+1;j<nF;j++) {
                  Double_t tmp=-c(j,i)*cinv(i,i);
                  for(Int_t k=i+1;k<j;k++) {
                     tmp -= cinv(k,i)*c(j,k);
                  }
                  cinv(j,i)=tmp*cinv(j,j);
               }
            }
            TMatrixDSparse cInvSparse(cinv);
            Finv=MultiplyMSparseTranspMSparse
            (&cInvSparse,&cInvSparse);
         }
         DeleteMatrix(&F);
      }
   }

   // here:
   //   rNumEl>=0: diagonal part has been inverted
   //   (nD2==0)||D2inv : D2 part has been inverted or is empty
   //   (nF==0)||Finv : F part has been inverted or is empty

   Int_t rankD2Block=0;
   if(rNumEl>=0) {
      if( ((nD2==0)||D2inv) && ((nF==0)||Finv) ) {
         // all matrix parts have been inverted, compose full matrix
         //           1/D1   0    0
         //  Ainv = (  0     E    G#   )
         //            0     G    F^-1
         //
         //        G = (F^-1) (-B/D2)
         //        E = 1/D2 + (-B#/D2) G)

         TMatrixDSparse *G=0;
         if(Finv && minusBD2inv) {
            G=MultiplyMSparseMSparse(Finv,minusBD2inv);
         }
         TMatrixDSparse *E=0;
         if(D2inv) E=new TMatrixDSparse(*D2inv);
         if(G && minusBD2inv) {
            TMatrixDSparse *minusBD2invTransG=
            MultiplyMSparseTranspMSparse(minusBD2inv,G);
            if(E) {
               AddMSparse(E,1.,minusBD2invTransG);
               DeleteMatrix(&minusBD2invTransG);
            } else {
               E=minusBD2invTransG;
            }
         }
         // pack matrix E to r
         if(E) {
            const Int_t *e_rows=E->GetRowIndexArray();
            const Int_t *e_cols=E->GetColIndexArray();
            const Double_t *e_data=E->GetMatrixArray();
            for(Int_t iE=0;iE<E->GetNrows();iE++) {
               Int_t iA=swap[iE+iDiagonal];
               for(Int_t indexE=e_rows[iE];indexE<e_rows[iE+1];++indexE) {
                  Int_t jE=e_cols[indexE];
                  Int_t jA=swap[jE+iDiagonal];
                  rEl_col[rNumEl]=iA;
                  rEl_row[rNumEl]=jA;
                  rEl_data[rNumEl]=e_data[indexE];
                  ++rNumEl;
               }
            }
            DeleteMatrix(&E);
         }
         // pack matrix G to r
         if(G) {
            const Int_t *g_rows=G->GetRowIndexArray();
            const Int_t *g_cols=G->GetColIndexArray();
            const Double_t *g_data=G->GetMatrixArray();
            for(Int_t iG=0;iG<G->GetNrows();iG++) {
               Int_t iA=swap[iG+iBlock];
               for(Int_t indexG=g_rows[iG];indexG<g_rows[iG+1];++indexG) {
                  Int_t jG=g_cols[indexG];
                  Int_t jA=swap[jG+iDiagonal];
                  // G
                  rEl_col[rNumEl]=iA;
                  rEl_row[rNumEl]=jA;
                  rEl_data[rNumEl]=g_data[indexG];
                  ++rNumEl;
                  // G#
                  rEl_col[rNumEl]=jA;
                  rEl_row[rNumEl]=iA;
                  rEl_data[rNumEl]=g_data[indexG];
                  ++rNumEl;
               }
            }
            DeleteMatrix(&G);
         }
         if(Finv) {
            // pack matrix Finv to r
            const Int_t *finv_rows=Finv->GetRowIndexArray();
            const Int_t *finv_cols=Finv->GetColIndexArray();
            const Double_t *finv_data=Finv->GetMatrixArray();
            for(Int_t iF=0;iF<Finv->GetNrows();iF++) {
               Int_t iA=swap[iF+iBlock];
               for(Int_t indexF=finv_rows[iF];indexF<finv_rows[iF+1];++indexF) {
                  Int_t jF=finv_cols[indexF];
                  Int_t jA=swap[jF+iBlock];
                  rEl_col[rNumEl]=iA;
                  rEl_row[rNumEl]=jA;
                  rEl_data[rNumEl]=finv_data[indexF];
                  ++rNumEl;
               }
            }
         }
         rankD2Block=nD2+nF;
      } else if(!rankPtr) {
         rNumEl=-3;
         Fatal("InvertMSparseSymmPos",
               "non-trivial part has rank < %d, matrix can not be inverted",
               nF);
      } else {
         // part of the matrix could not be inverted, get eingenvalue
         // decomposition
         Int_t nEV=nD2+nF;
         Double_t epsilonEV2=fEpsMatrix /* *nEV*nEV */;
         Info("InvertMSparseSymmPos",
              "cholesky-decomposition failed, try eigenvalue analysis");
#ifdef DEBUG
         std::cout<<"nEV="<<nEV<<" iDiagonal="<<iDiagonal<<"\n";
#endif
         TMatrixDSym EV(nEV);
         for(Int_t i=0;i<nEV;i++) {
            Int_t iA=swap[i+iDiagonal];
            for(Int_t indexA=a_rows[iA];indexA<a_rows[iA+1];indexA++) {
               Int_t jA=a_cols[indexA];
               Int_t j0=swapBack[jA];
               if(j0>=iDiagonal) {
                  Int_t j=j0-iDiagonal;
#ifdef DEBUG
                  if((i<0)||(j<0)||(i>=nEV)||(j>=nEV)) {
                     std::cout<<" error "<<nEV<<" "<<i<<" "<<j<<"\n";
                  }
#endif
                  if(!TMath::Finite(a_data[indexA])) {
                     Fatal("InvertMSparseSymmPos",
                           "non-finite number detected element %d %d\n",
                           iA,jA);
                  }
                  EV(i,j)=a_data[indexA];
               }
            }
         }
         // EV.Print();
         TMatrixDSymEigen Eigen(EV);
#ifdef DEBUG
         std::cout<<"Eigenvalues\n";
         // Eigen.GetEigenValues().Print();
#endif
         Int_t errorEV=0;
         Double_t condition=1.0;
         if(Eigen.GetEigenValues()(0)<0.0) {
            errorEV=1;
         } else if(Eigen.GetEigenValues()(0)>0.0) {
            condition=Eigen.GetEigenValues()(nEV-1)/Eigen.GetEigenValues()(0);
         }
         if(condition<-epsilonEV2) {
            errorEV=2;
         }
         if(errorEV) {
            if(errorEV==1) {
               Error("InvertMSparseSymmPos",
                     "Largest Eigenvalue is negative %f",
                     Eigen.GetEigenValues()(0));
            } else {
               Error("InvertMSparseSymmPos",
                     "Some Eigenvalues are negative (EV%d/EV0=%g epsilon=%g)",
                     nEV-1,
                     Eigen.GetEigenValues()(nEV-1)/Eigen.GetEigenValues()(0),
                     epsilonEV2);
            }
         }
         // compose inverse matrix
         rankD2Block=0;
         Double_t setToZero=epsilonEV2*Eigen.GetEigenValues()(0);
         TMatrixD inverseEV(nEV,1);
         for(Int_t i=0;i<nEV;i++) {
            Double_t x=Eigen.GetEigenValues()(i);
            if(x>setToZero) {
               inverseEV(i,0)=1./x;
               ++rankD2Block;
            }
         }
         TMatrixDSparse V(Eigen.GetEigenVectors());
         TMatrixDSparse *VDVt=MultiplyMSparseMSparseTranspVector
         (&V,&V,&inverseEV);

         // pack matrix VDVt to r
         const Int_t *vdvt_rows=VDVt->GetRowIndexArray();
         const Int_t *vdvt_cols=VDVt->GetColIndexArray();
         const Double_t *vdvt_data=VDVt->GetMatrixArray();
         for(Int_t iVDVt=0;iVDVt<VDVt->GetNrows();iVDVt++) {
            Int_t iA=swap[iVDVt+iDiagonal];
            for(Int_t indexVDVt=vdvt_rows[iVDVt];
                indexVDVt<vdvt_rows[iVDVt+1];++indexVDVt) {
               Int_t jVDVt=vdvt_cols[indexVDVt];
               Int_t jA=swap[jVDVt+iDiagonal];
               rEl_col[rNumEl]=iA;
               rEl_row[rNumEl]=jA;
               rEl_data[rNumEl]=vdvt_data[indexVDVt];
               ++rNumEl;
            }
         }
         DeleteMatrix(&VDVt);
      }
   }
   if(rankPtr) {
      *rankPtr=rankD1+rankD2Block;
   }


   DeleteMatrix(&D2inv);
   DeleteMatrix(&Finv);
   DeleteMatrix(&B);
   DeleteMatrix(&minusBD2inv);

   delete [] swap;
   delete [] swapBack;
   delete [] isZero;

   TMatrixDSparse *r=(rNumEl>=0) ?
   CreateSparseMatrix(A->GetNrows(),A->GetNrows(),rNumEl,
                      rEl_row,rEl_col,rEl_data) : 0;
   delete [] rEl_data;
   delete [] rEl_col;
   delete [] rEl_row;

#ifdef DEBUG_DETAIL
   // sanity test
   if(r) {
      TMatrixDSparse *Ar=MultiplyMSparseMSparse(A,r);
      TMatrixDSparse *ArA=MultiplyMSparseMSparse(Ar,A);
      TMatrixDSparse *rAr=MultiplyMSparseMSparse(r,Ar);

      TMatrixD ar(*Ar);
      TMatrixD a(*A);
      TMatrixD ara(*ArA);
      TMatrixD R(*r);
      TMatrixD rar(*rAr);

      DeleteMatrix(&Ar);
      DeleteMatrix(&ArA);
      DeleteMatrix(&rAr);

      Double_t epsilonA2=fEpsMatrix /* *a.GetNrows()*a.GetNcols() */;
      for(Int_t i=0;i<a.GetNrows();i++) {
         for(Int_t j=0;j<a.GetNcols();j++) {
            // ar should be symmetric
            if(TMath::Abs(ar(i,j)-ar(j,i))>
               epsilonA2*(TMath::Abs(ar(i,j))+TMath::Abs(ar(j,i)))) {
               std::cout<<"Ar is not symmetric Ar("<<i<<","<<j<<")="<<ar(i,j)
               <<" Ar("<<j<<","<<i<<")="<<ar(j,i)<<"\n";
            }
            // ara should be equal a
            if(TMath::Abs(ara(i,j)-a(i,j))>
               epsilonA2*(TMath::Abs(ara(i,j))+TMath::Abs(a(i,j)))) {
               std::cout<<"ArA is not equal A ArA("<<i<<","<<j<<")="<<ara(i,j)
               <<" A("<<i<<","<<j<<")="<<a(i,j)<<"\n";
            }
            // ara should be equal a
            if(TMath::Abs(rar(i,j)-R(i,j))>
               epsilonA2*(TMath::Abs(rar(i,j))+TMath::Abs(R(i,j)))) {
               std::cout<<"rAr is not equal r rAr("<<i<<","<<j<<")="<<rar(i,j)
               <<" r("<<i<<","<<j<<")="<<R(i,j)<<"\n";
            }
         }
      }
      if(rankPtr) std::cout<<"rank="<<*rankPtr<<"\n";
   } else {
      std::cout<<"Matrix is not positive\n";
   }
#endif
   return r;

}

TString TUnfold::GetOutputBinName(Int_t iBinX) const
{
   // given a bin number, return the name of the output bin
   // this method makes more sense for the class TUnfoldDnesity
   // where it gets overwritten
   return TString::Format("#%d",iBinX);
}

TUnfold::TUnfold(const TH2 *hist_A, EHistMap histmap, ERegMode regmode,
                 EConstraint constraint)
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
   //    fL: filled depending on the regularisation scheme
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
      Int_t nonZeroY = 0;
      for (Int_t iy = 0; iy < ny; iy++) {
         Double_t z;
         if (histmap == kHistMapOutputHoriz) {
            z = hist_A->GetBinContent(ix, iy + 1);
         } else {
            z = hist_A->GetBinContent(iy + 1, ix);
         }
         if (z != 0.0) {
            nonzeroA++;
            nonZeroY++;
            sum += z;
         }
      }
      // check whether there is any sensitivity to this generator bin

      if (nonZeroY) {
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
      int nDisconnected=0;
      for (Int_t ix = 0; ix < nx0; ix++) {
         if(fHistToX[ix]<0) {
            nprint++;
            if(ixlast<0) {
               binlist +=" ";
               binlist +=ix;
               ixfirst=ix;
            }
            ixlast=ix;
            nDisconnected++;
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
         Info("TUnfold","underflow and overflow bin "
              "do not depend on the input data");
      } else {
         Warning("TUnfold","%d output bins "
                 "do not depend on the input data %s",nDisconnected,
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
         if (z != 0.0) {
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
   fL = new TMatrixDSparse(1, GetNx());
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
   DeleteMatrix(&fL);
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

Bool_t TUnfold::AddRegularisationCondition
(Int_t i0,Double_t f0,Int_t i1,Double_t f1,Int_t i2,Double_t f2)
{
   // add regularisation condition for a triplet of bins and scale factors
   // the arguments are used to form one row (k) of the matrix L
   //   L(k,i0)=f0
   //   L(k,i1)=f1
   //   L(k,i2)=f2
   // the indices i0,i1,i2 are transformed of bin numbers
   // if i2<0, do not fill  L(k,i2)
   // if i1<0, do not fill  L(k,i1)

   Int_t indices[3];
   Double_t data[3];
   Int_t nEle=0;

   if(i2>=0) {
      data[nEle]=f2;
      indices[nEle]=i2;
      nEle++;
   }
   if(i1>=0) {
      data[nEle]=f1;
      indices[nEle]=i1;
      nEle++;
   }
   if(i0>=0) {
      data[nEle]=f0;
      indices[nEle]=i0;
      nEle++;
   }
   return AddRegularisationCondition(nEle,indices,data);
}

Bool_t TUnfold::AddRegularisationCondition
(Int_t nEle,const Int_t *indices,const Double_t *rowData)
{
   // add a regularisation condition
   // the arguments are used to form one row (k) of the matrix L
   //   L(k,indices[0])=rowData[0]
   //    ...
   //   L(k,indices[nEle-1])=rowData[nEle-1]
   //
   // the indices are taken as bin numbers,
   // transformed to internal bin numbers using the array fHistToX


   Bool_t r=kTRUE;
   const Int_t *l0_rows=fL->GetRowIndexArray();
   const Int_t *l0_cols=fL->GetColIndexArray();
   const Double_t *l0_data=fL->GetMatrixArray();

   Int_t nF=l0_rows[fL->GetNrows()]+nEle;
   Int_t *l_row=new Int_t[nF];
   Int_t *l_col=new Int_t[nF];
   Double_t *l_data=new Double_t[nF];
   // decode original matrix
   nF=0;
   for(Int_t row=0;row<fL->GetNrows();row++) {
      for(Int_t k=l0_rows[row];k<l0_rows[row+1];k++) {
         l_row[nF]=row;
         l_col[nF]=l0_cols[k];
         l_data[nF]=l0_data[k];
         nF++;
      }
   }

   // if the original matrix does not have any data, reset its row count
   Int_t rowMax=0;
   if(nF>0) {
      rowMax=fL->GetNrows();
   }

   // add the new regularisation conditions
   for(Int_t i=0;i<nEle;i++) {
      Int_t col=fHistToX[indices[i]];
      if(col<0) {
         r=kFALSE;
         break;
      }
      l_row[nF]=rowMax;
      l_col[nF]=col;
      l_data[nF]=rowData[i];
      nF++;
   }

   // replace the old matrix fL
   if(r) {
      DeleteMatrix(&fL);
      fL=CreateSparseMatrix(rowMax+1,GetNx(),nF,l_row,l_col,l_data);
   }
   delete [] l_row;
   delete [] l_col;
   delete [] l_data;
   return r;
}

Int_t TUnfold::RegularizeSize(int bin, Double_t scale)
{
   // add regularisation on the size of bin i
   //    bin: bin number
   //    scale: size of the regularisation
   // return value: number of conditions which have been skipped
   // modifies data member fL

   if(fRegMode==kRegModeNone) fRegMode=kRegModeSize;
   if(fRegMode!=kRegModeSize) fRegMode=kRegModeMixed;

   return AddRegularisationCondition(bin,scale) ? 0 : 1;
}

Int_t TUnfold::RegularizeDerivative(int left_bin, int right_bin,
                                    Double_t scale)
{
   // add regularisation on the difference of two bins
   //   left_bin: 1st bin
   //   right_bin: 2nd bin
   //   scale: size of the regularisation
   // return value: number of conditions which have been skipped
   // modifies data member fL

   if(fRegMode==kRegModeNone) fRegMode=kRegModeDerivative;
   if(fRegMode!=kRegModeDerivative) fRegMode=kRegModeMixed;

   return AddRegularisationCondition(left_bin,-scale,right_bin,scale) ? 0 : 1;
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
   // modifies data member fL

   if(fRegMode==kRegModeNone) fRegMode=kRegModeCurvature;
   if(fRegMode!=kRegModeCurvature) fRegMode=kRegModeMixed;

   return AddRegularisationCondition
   (left_bin,-scale_left,
    center_bin,scale_left+scale_right,
    right_bin,-scale_right)
   ? 0 : 1;
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
   // modifies data member fL

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
      Error("RegularizeBins","regmode = %d is not valid",regmode);
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
   // modifies data member fL

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
   //   fA, fX0, fL
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
                        Double_t oneOverZeroError,const TH2 *hist_vyy,
                        const TH2 *hist_vyy_inv)
{
   // Define the input data for subsequent calls to DoUnfold(Double_t)
   //   input:   input distribution with errors
   //   scaleBias:  scale factor applied to the bias
   //   oneOverZeroError: for bins with zero error, this number defines 1/error.
   //   hist_vyy: if non-zero, defines the data covariance matrix
   //             otherwise it is calculated from the data errors
   //   hist_vyy_inv: if non-zero and if hist_vyy  is set, defines the inverse of the data covariance matrix
   // Return value: number of bins with bad error
   //                 +10000*number of unconstrained output bins
   //         Note: return values>=10000 are fatal errors,
   //               for the given input, the unfolding can not be done!
   // Data members modified:
   //   fY, fVyy, , fBiasScale
   // Data members cleared
   //   fVyyInv, fNdf
   //   + see ClearResults

   DeleteMatrix(&fVyyInv);
   fNdf=0;

   fBiasScale = scaleBias;

   // delete old results (if any)
   ClearResults();

   // construct error matrix and inverted error matrix of measured quantities
   // from errors of input histogram or use error matrix

   Int_t *rowVyyN=new Int_t[GetNy()*GetNy()+1];
   Int_t *colVyyN=new Int_t[GetNy()*GetNy()+1];
   Double_t *dataVyyN=new Double_t[GetNy()*GetNy()+1];

   Int_t *rowVyy1=new Int_t[GetNy()];
   Int_t *colVyy1=new Int_t[GetNy()];
   Double_t *dataVyy1=new Double_t[GetNy()];
   Double_t *dataVyyDiag=new Double_t[GetNy()];

   Int_t nError=0;
   Int_t nVyyN=0;
   Int_t nVyy1=0;
   for (Int_t iy = 0; iy < GetNy(); iy++) {
      // diagonals
      Double_t dy2;
      if(!hist_vyy) {
         Double_t dy = input->GetBinError(iy + 1);
         dy2=dy*dy;
         if (dy2 <= 0.0) {
            nError++;
            if(oneOverZeroError>0.0) {
               dy2 = 1./ ( oneOverZeroError*oneOverZeroError);
            }
         }
      } else {
         dy2 = hist_vyy->GetBinContent(iy+1,iy+1);
      }
      rowVyyN[nVyyN] = iy;
      colVyyN[nVyyN] = iy;
      rowVyy1[nVyy1] = iy;
      colVyy1[nVyy1] = 0;
      dataVyyDiag[iy] = dy2;
      if(dy2>0.0) {
         dataVyyN[nVyyN++] = dy2;
         dataVyy1[nVyy1++] = dy2;
      }
   }
   if(hist_vyy) {
      // non-diagonal elements
      for (Int_t iy = 0; iy < GetNy(); iy++) {
         // ignore rows where the diagonal is zero
         if(dataVyyDiag[iy]<=0.0) continue;
         for (Int_t jy = 0; jy < GetNy(); jy++) {
            // skip diagonal elements
            if(iy==jy) continue;
            // ignore columns where the diagonal is zero
            if(dataVyyDiag[jy]<=0.0) continue;

            rowVyyN[nVyyN] = iy;
            colVyyN[nVyyN] = jy;
            dataVyyN[nVyyN]= hist_vyy->GetBinContent(iy+1,jy+1);
            if(dataVyyN[nVyyN] == 0.0) continue;
            nVyyN ++;
         }
      }
      if(hist_vyy_inv) {
         Warning("SetInput",
                 "inverse of input covariance is taken from user input");
         Int_t *rowVyyInv=new Int_t[GetNy()*GetNy()+1];
         Int_t *colVyyInv=new Int_t[GetNy()*GetNy()+1];
         Double_t *dataVyyInv=new Double_t[GetNy()*GetNy()+1];
         Int_t nVyyInv=0;
         for (Int_t iy = 0; iy < GetNy(); iy++) {
            for (Int_t jy = 0; jy < GetNy(); jy++) {
               rowVyyInv[nVyyInv] = iy;
               colVyyInv[nVyyInv] = jy;
               dataVyyInv[nVyyInv]= hist_vyy_inv->GetBinContent(iy+1,jy+1);
               if(dataVyyInv[nVyyInv] == 0.0) continue;
               nVyyInv ++;
            }
         }
         fVyyInv=CreateSparseMatrix
         (GetNy(),GetNy(),nVyyInv,rowVyyInv,colVyyInv,dataVyyInv);
         delete [] rowVyyInv;
         delete [] colVyyInv;
         delete [] dataVyyInv;
      }
   }
   DeleteMatrix(&fVyy);
   fVyy = CreateSparseMatrix
   (GetNy(),GetNy(),nVyyN,rowVyyN,colVyyN,dataVyyN);

   delete[] rowVyyN;
   delete[] colVyyN;
   delete[] dataVyyN;

   TMatrixDSparse *vecV=CreateSparseMatrix
   (GetNy(),1,nVyy1,rowVyy1,colVyy1, dataVyy1);

   delete[] rowVyy1;
   delete[] colVyy1;
   delete[] dataVyy1;

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
            Warning("SetInput","%d/%d input bins have zero error,"
                    " 1/error set to %lf.",nError,GetNy(),oneOverZeroError);
         } else {
            Warning("SetInput","One input bin has zero error,"
                    " 1/error set to %lf.",oneOverZeroError);
         }
      } else {
         if(nError>1) {
            Warning("SetInput","%d/%d input bins have zero error,"
                    " and are ignored.",nError,GetNy());
         } else {
            Warning("SetInput","One input bin has zero error,"
                    " and is ignored.");
         }
      }
      fIgnoredBins=nError;
   }
   if(nError2>0) {
      // check whether data points with zero error are responsible
      if(oneOverZeroError<=0.0) {
         //const Int_t *a_rows=fA->GetRowIndexArray();
         //const Int_t *a_cols=fA->GetColIndexArray();
         for (Int_t col = 0; col <mAtV->GetNrows();col++) {
            if(mAtV->GetRowIndexArray()[col]==
               mAtV->GetRowIndexArray()[col+1]) {
               TString binlist("no data to constrain output bin ");
               binlist += GetOutputBinName(fXToHist[col]);
               /* binlist +=" depends on ignored input bins ";
                for(Int_t row=0;row<fA->GetNrows();row++) {
                if(dataVyyDiag[row]>0.0) continue;
                for(Int_t i=a_rows[row];i<a_rows[row+1];i++) {
                if(a_cols[i]!=col) continue;
                binlist +=" ";
                binlist +=row;
                }
                } */
               Warning("SetInput","%s",binlist.Data());
            }
         }
      }
      if(nError2>1) {
         Error("SetInput","%d/%d output bins are not constrained by any data.",
               nError2,mAtV->GetNrows());
      } else {
         Error("SetInput","One output bins is not constrained by any data.");
      }
   }
   DeleteMatrix(&mAtV);

   delete[] dataVyyDiag;

   return nError+10000*nError2;
}

Double_t TUnfold::DoUnfold(Double_t tau)
{
   // Unfold with given value of regularisation parameter tau
   //     tau: new tau parameter
   // required data members:
   //     fA:  matrix to relate x and y
   //     fY:  measured data points
   //     fX0: bias on x
   //     fBiasScale: scale factor for fX0
   //     fV:  inverse of covariance matrix for y
   //     fL: regularisation conditions
   // modified data members:
   //     fTauSquared and those documented in DoUnfold(void)
   fTauSquared=tau*tau;
   return DoUnfold();
}

Int_t TUnfold::ScanLcurve(Int_t nPoint,
                          Double_t tauMin,Double_t tauMax,
                          TGraph **lCurve,TSpline **logTauX,
                          TSpline **logTauY)
{
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
      // here no range is given, has to be determined automatically
      // the maximum tau is determined from the chi**2 values
      // observed from unfolding without regulatisation

      // first unfolding, without regularisation
      DoUnfold(0.0);

      // if the number of degrees of freedom is too small, create an error
      if(GetNdf()<=0) {
         Error("ScanLcurve","too few input bins, NDF<=0 %d",GetNdf());
      }

      Double_t x0=GetLcurveX();
      Double_t y0=GetLcurveY();
      Info("ScanLcurve","logtau=-Infinity X=%lf Y=%lf",x0,y0);
      if(!TMath::Finite(x0)) {
         Fatal("ScanLcurve","problem (too few input bins?) X=%f",x0);
      }
      if(!TMath::Finite(y0)) {
         Fatal("ScanLcurve","problem (missing regularisation?) Y=%f",y0);
      }
      {
         // unfolding guess maximum tau and store it
         Double_t logTau=
         0.5*(TMath::Log10(fChi2A+3.*TMath::Sqrt(GetNdf()+1.0))
              -GetLcurveY());
         DoUnfold(TMath::Power(10.,logTau));
         if((!TMath::Finite(GetLcurveX())) ||(!TMath::Finite(GetLcurveY()))) {
            Fatal("ScanLcurve","problem (missing regularisation?) X=%f Y=%f",
                  GetLcurveX(),GetLcurveY());
         }
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
         // regularisation. Ideally this should never happen
         do {
            x0=GetLcurveX();
            Double_t logTau=(*curve.begin()).first-0.5;
            DoUnfold(TMath::Power(10.,logTau));
            if((!TMath::Finite(GetLcurveX())) ||(!TMath::Finite(GetLcurveY()))) {
               Fatal("ScanLcurve","problem (missing regularisation?) X=%f Y=%f",
                     GetLcurveX(),GetLcurveY());
            }
            curve[logTau]=std::make_pair(GetLcurveX(),GetLcurveY());
            Info("ScanLcurve","logtau=%lf X=%lf Y=%lf",
                 logTau,GetLcurveX(),GetLcurveY());
         }
         while(((int)curve.size()<(nPoint-1)/2)&&
               ((*curve.begin()).second.first<x0));
      } else {
         // minimum tau is chosen such that is less than
         // 1% different from the case of no regularusation
         // log10(1.01) = 0.00432

         // here, more than one point are inserted if necessary
         while(((int)curve.size()<nPoint-1)&&
               (((*curve.begin()).second.first-x0>0.00432)||
                ((*curve.begin()).second.second-y0>0.00432)||
                (curve.size()<2))) {
                  Double_t logTau=(*curve.begin()).first-0.5;
                  DoUnfold(TMath::Power(10.,logTau));
                  if((!TMath::Finite(GetLcurveX())) ||(!TMath::Finite(GetLcurveY()))) {
                     Fatal("ScanLcurve","problem (missing regularisation?) X=%f Y=%f",
                           GetLcurveX(),GetLcurveY());
                  }
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
         if((!TMath::Finite(GetLcurveX())) ||(!TMath::Finite(GetLcurveY()))) {
            Fatal("ScanLcurve","problem (missing regularisation?) X=%f Y=%f",
                  GetLcurveX(),GetLcurveY());
         }
         Info("ScanLcurve","logtau=%lf X=%lf Y=%lf",
              logTauMax,GetLcurveX(),GetLcurveY());
         curve[logTauMax]=std::make_pair(GetLcurveX(),GetLcurveY());
      }
      // insert minimum tau
      DoUnfold(TMath::Power(10.,logTauMin));
      if((!TMath::Finite(GetLcurveX())) ||(!TMath::Finite(GetLcurveY()))) {
         Fatal("ScanLcurve","problem (missing regularisation?) X=%f Y=%f",
               GetLcurveX(),GetLcurveY());
      }
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
      if((!TMath::Finite(GetLcurveX())) ||(!TMath::Finite(GetLcurveY()))) {
         Fatal("ScanLcurve","problem (missing regularisation?) X=%f Y=%f",
               GetLcurveX(),GetLcurveY());
      }
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
      if((!TMath::Finite(GetLcurveX())) ||(!TMath::Finite(GetLcurveY()))) {
         Fatal("ScanLcurve","problem (missing regularisation?) X=%f Y=%f",
               GetLcurveX(),GetLcurveY());
      }
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

void TUnfold::GetNormalisationVector(TH1 *out,const Int_t *binMap) const
{
   // get vector of normalisation factors
   //   out: output histogram
   //   binMap: for each bin of the original output distribution
   //           specify the destination bin. A value of -1 means that the bin
   //           is discarded. 0 means underflow bin, 1 first bin, ...
   //        binMap[0] : destination of underflow bin
   //        binMap[1] : destination of first bin
   //          ...

   ClearHistogram(out);
   for (Int_t i = 0; i < GetNx(); i++) {
      int dest=binMap ? binMap[fXToHist[i]] : fXToHist[i];
      if(dest>=0) {
         out->SetBinContent(dest, fSumOverY[i] + out->GetBinContent(dest));
      }
   }
}

void TUnfold::GetBias(TH1 *out,const Int_t *binMap) const
{
   // get bias distribution, possibly with bin remapping
   //   out: output histogram
   //   binMap: for each bin of the original output distribution
   //           specify the destination bin. A value of -1 means that the bin
   //           is discarded. 0 means underflow bin, 1 first bin, ...
   //        binMap[0] : destination of underflow bin
   //        binMap[1] : destination of first bin
   //          ...

   ClearHistogram(out);
   for (Int_t i = 0; i < GetNx(); i++) {
      int dest=binMap ? binMap[fXToHist[i]] : fXToHist[i];
      if(dest>=0) {
         out->SetBinContent(dest, fBiasScale*(*fX0) (i, 0) +
                            out->GetBinContent(dest));
      }
   }
}

void TUnfold::GetFoldedOutput(TH1 *out,const Int_t *binMap) const
{
   // get unfolding result, folded back trough the matrix
   //   out: output histogram
   //   binMap: for each bin of the original output distribution
   //           specify the destination bin. A value of -1 means that the bin
   //           is discarded. 0 means underflow bin, 1 first bin, ...
   //        binMap[0] : destination of underflow bin
   //        binMap[1] : destination of first bin
   //          ...

   ClearHistogram(out);

   TMatrixDSparse *AVxx=MultiplyMSparseMSparse(fA,fVxx);

   const Int_t *rows_A=fA->GetRowIndexArray();
   const Int_t *cols_A=fA->GetColIndexArray();
   const Double_t *data_A=fA->GetMatrixArray();
   const Int_t *rows_AVxx=AVxx->GetRowIndexArray();
   const Int_t *cols_AVxx=AVxx->GetColIndexArray();
   const Double_t *data_AVxx=AVxx->GetMatrixArray();

   for (Int_t i = 0; i < GetNy(); i++) {
      Int_t destI = binMap ? binMap[i+1] : i+1;
      if(destI<0) continue;

      out->SetBinContent(destI, (*fAx) (i, 0)+ out->GetBinContent(destI));
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
      out->SetBinError(destI,TMath::Sqrt(e2));
   }
   DeleteMatrix(&AVxx);
}

void TUnfold::GetProbabilityMatrix(TH2 *A,EHistMap histmap) const
{
   // retreive matrix of probabilities
   //    histmap: on which axis to arrange the input/output vector
   //    A: histogram to store the probability matrix
   const Int_t *rows_A=fA->GetRowIndexArray();
   const Int_t *cols_A=fA->GetColIndexArray();
   const Double_t *data_A=fA->GetMatrixArray();
   for (Int_t iy = 0; iy <fA->GetNrows(); iy++) {
      for(Int_t indexA=rows_A[iy];indexA<rows_A[iy+1];indexA++) {
         Int_t ix = cols_A[indexA];
         Int_t ih=fXToHist[ix];
         if (histmap == kHistMapOutputHoriz) {
            A->SetBinContent(ih, iy,data_A[indexA]);
         } else {
            A->SetBinContent(iy, ih,data_A[indexA]);
         }
      }
   }
}

void TUnfold::GetInput(TH1 *out,const Int_t *binMap) const
{
   // retreive input distribution
   //   out: output histogram
   //   binMap: for each bin of the original output distribution
   //           specify the destination bin. A value of -1 means that the bin
   //           is discarded. 0 means underflow bin, 1 first bin, ...
   //        binMap[0] : destination of underflow bin
   //        binMap[1] : destination of first bin
   //          ...

   ClearHistogram(out);

   const Int_t *rows_Vyy=fVyy->GetRowIndexArray();
   const Int_t *cols_Vyy=fVyy->GetColIndexArray();
   const Double_t *data_Vyy=fVyy->GetMatrixArray();

   for (Int_t i = 0; i < GetNy(); i++) {
      Int_t destI=binMap ? binMap[i] : i;
      if(destI<0) continue;

      out->SetBinContent(destI, (*fY) (i, 0)+out->GetBinContent(destI));

      Double_t e=0.0;
      for(int index=rows_Vyy[i];index<rows_Vyy[i+1];index++) {
         if(cols_Vyy[index]==i) {
            e=TMath::Sqrt(data_Vyy[index]);
         }
      }
      out->SetBinError(destI, e);
   }
}

void TUnfold::GetInputInverseEmatrix(TH2 *out)
{
   // calculate the inverse of the contribution to the error matrix
   // corresponding to the input data
   if(!fVyyInv) {
      Int_t rank=0;
      fVyyInv=InvertMSparseSymmPos(fVyy,&rank);
      // and count number of degrees of freedom
      fNdf = rank-GetNpar();

      if(rank<GetNy()-fIgnoredBins) {
         Warning("GetInputInverseEmatrix","input covariance matrix has rank %d expect %d",
                 rank,GetNy());
      }
      if(fNdf<0) {
         Error("GetInputInverseEmatrix","number of parameters %d > %d (rank of input covariance). Problem can not be solved",GetNpar(),rank);
      } else if(fNdf==0) {
         Warning("GetInputInverseEmatrix","number of parameters %d = input rank %d. Problem is ill posed",GetNpar(),rank);
      }
   }
   if(out) {
      // return matrix as histogram
      const Int_t *rows_Vyy=fVyy->GetRowIndexArray();
      const Int_t *cols_Vyy=fVyy->GetColIndexArray();
      const Double_t *data_Vyy=fVyy->GetMatrixArray();

      for(int i=0;i<=out->GetNbinsX()+1;i++) {
         for(int j=0;j<=out->GetNbinsY()+1;j++) {
            out->SetBinContent(i,j,0.);
         }
      }

      for (Int_t i = 0; i < fVyy->GetNrows(); i++) {
         for(int index=rows_Vyy[i];index<rows_Vyy[i+1];index++) {
            Int_t j=cols_Vyy[index];
            out->SetBinContent(i+1,j+1,data_Vyy[index]);
         }
      }
   }
}

void TUnfold::GetLsquared(TH2 *out) const
{
   // retreive matrix of regularisation conditions squared
   //   out: prebooked matrix

   TMatrixDSparse *lSquared=MultiplyMSparseTranspMSparse(fL,fL);
   // loop over sparse matrix
   const Int_t *rows=lSquared->GetRowIndexArray();
   const Int_t *cols=lSquared->GetColIndexArray();
   const Double_t *data=lSquared->GetMatrixArray();
   for (Int_t i = 0; i < GetNx(); i++) {
      for (Int_t cindex = rows[i]; cindex < rows[i+1]; cindex++) {
         Int_t j=cols[cindex];
         out->SetBinContent(fXToHist[i], fXToHist[j],data[cindex]);
      }
   }
   DeleteMatrix(&lSquared);
}

void TUnfold::GetL(TH2 *out) const
{
   // retreive matrix of regularisation conditions
   //   out: prebooked matrix

   // loop over sparse matrix
   const Int_t *rows=fL->GetRowIndexArray();
   const Int_t *cols=fL->GetColIndexArray();
   const Double_t *data=fL->GetMatrixArray();
   for (Int_t row = 0; row < GetNr(); row++) {
      for (Int_t cindex = rows[row]; cindex < rows[row+1]; cindex++) {
         Int_t col=cols[cindex];
         Int_t indexH=fXToHist[col];
         out->SetBinContent(indexH,row+1,data[cindex]);
      }
   }
}

void TUnfold::SetConstraint(EConstraint constraint)
{
   // set type of constraint for the next unfolding
   if(fConstraint !=constraint) ClearResults();
   fConstraint=constraint;
   Info("SetConstraint","fConstraint=%d",fConstraint);
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

Double_t TUnfold::GetLcurveX(void) const
{
   // return value on x axis of L curve
   return TMath::Log10(fChi2A);
}

Double_t TUnfold::GetLcurveY(void) const
{
   // return value on y axis of L curve
   return TMath::Log10(fLXsquared);
}

void TUnfold::GetOutput(TH1 *output,const Int_t *binMap) const
{
   // get output distribution, cumulated over several bins
   //   output: output histogram
   //   binMap: for each bin of the original output distribution
   //           specify the destination bin. A value of -1 means that the bin
   //           is discarded. 0 means underflow bin, 1 first bin, ...
   //        binMap[0] : destination of underflow bin
   //        binMap[1] : destination of first bin
   //          ...

   ClearHistogram(output);
   /* Int_t nbin=output->GetNbinsX();
    Double_t *c=new Double_t[nbin+2];
    Double_t *e2=new Double_t[nbin+2];
    for(Int_t i=0;i<nbin+2;i++) {
    c[i]=0.0;
    e2[i]=0.0;
    } */

   std::map<Int_t,Double_t> e2;

   const Int_t *rows_Vxx=fVxx->GetRowIndexArray();
   const Int_t *cols_Vxx=fVxx->GetColIndexArray();
   const Double_t *data_Vxx=fVxx->GetMatrixArray();

   Int_t binMapSize = fHistToX.GetSize();
   for(Int_t i=0;i<binMapSize;i++) {
      Int_t destBinI=binMap ? binMap[i] : i; // histogram bin
      Int_t srcBinI=fHistToX[i]; // matrix row index
      if((destBinI>=0)&&(srcBinI>=0)) {
         output->SetBinContent
         (destBinI, (*fX)(srcBinI,0)+ output->GetBinContent(destBinI));
         // here we loop over the columns of the error matrix
         //   j: counts histogram bins
         //   index: counts sparse matrix index
         // the algorithm makes use of the fact that fHistToX is ordered
         Int_t j=0; // histogram bin
         Int_t index_vxx=rows_Vxx[srcBinI];
         //Double_t e2=TMath::Power(output->GetBinError(destBinI),2.);
         while((j<binMapSize)&&(index_vxx<rows_Vxx[srcBinI+1])) {
            Int_t destBinJ=binMap ? binMap[j] : j; // histogram bin
            if(destBinI!=destBinJ) {
               // only diagonal elements are calculated
               j++;
            } else {
               Int_t srcBinJ=fHistToX[j]; // matrix column index
               if(srcBinJ<0) {
                  // bin was not unfolded, check next bin
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
                     e2[destBinI] += data_Vxx[index_vxx];
                     j++;
                     index_vxx++;
                  }
               }
            }
         }
         //output->SetBinError(destBinI,TMath::Sqrt(e2));
      }
   }
   for(std::map<Int_t,Double_t>::const_iterator i=e2.begin();
       i!=e2.end();i++) {
      //cout<<(*i).first<<" "<<(*i).second<<"\n";
      output->SetBinError((*i).first,TMath::Sqrt((*i).second));
   }
}

void TUnfold::ErrorMatrixToHist(TH2 *ematrix,const TMatrixDSparse *emat,
                                const Int_t *binMap,Bool_t doClear) const
{
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

void TUnfold::GetEmatrix(TH2 *ematrix,const Int_t *binMap) const
{
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

void TUnfold::GetRhoIJ(TH2 *rhoij,const Int_t *binMap) const
{
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

Double_t TUnfold::GetRhoI(TH1 *rhoi,const Int_t *binMap,TH2 *invEmat) const
{
   // get global correlation coefficients with arbitrary min map
   //   rhoi: global correlation histogram
   //   binMap: for each bin of the original output distribution
   //           specify the destination bin. A value of -1 means that the bin
   //           is discarded. 0 means underflow bin, 1 first bin, ...
   //        binMap[0] : destination of underflow bin
   //        binMap[1] : destination of first bin
   //          ...
   // return value: maximum global correlation

   ClearHistogram(rhoi,-1.);

   if(binMap) {
      // if there is a bin map, the matrix needs to be inverted
      // otherwise, use the existing inverse, such that
      // no matrix inversion is needed
      return GetRhoIFromMatrix(rhoi,fVxx,binMap,invEmat);
   } else {
      Double_t rhoMax=0.0;

      const Int_t *rows_Vxx=fVxx->GetRowIndexArray();
      const Int_t *cols_Vxx=fVxx->GetColIndexArray();
      const Double_t *data_Vxx=fVxx->GetMatrixArray();

      const Int_t *rows_VxxInv=fVxxInv->GetRowIndexArray();
      const Int_t *cols_VxxInv=fVxxInv->GetColIndexArray();
      const Double_t *data_VxxInv=fVxxInv->GetMatrixArray();

      for(Int_t i=0;i<GetNx();i++) {
         Int_t destI=fXToHist[i];
         Double_t e_ii=0.0,einv_ii=0.0;
         for(int index_vxx=rows_Vxx[i];index_vxx<rows_Vxx[i+1];index_vxx++) {
            if(cols_Vxx[index_vxx]==i) {
               e_ii=data_Vxx[index_vxx];
               break;
            }
         }
         for(int index_vxxinv=rows_VxxInv[i];index_vxxinv<rows_VxxInv[i+1];
             index_vxxinv++) {
            if(cols_VxxInv[index_vxxinv]==i) {
               einv_ii=data_VxxInv[index_vxxinv];
               break;
            }
         }
         Double_t rho=1.0;
         if((einv_ii>0.0)&&(e_ii>0.0)) rho=1.-1./(einv_ii*e_ii);
         if(rho>=0.0) rho=TMath::Sqrt(rho);
         else rho=-TMath::Sqrt(-rho);
         if(rho>rhoMax) {
            rhoMax = rho;
         }
         rhoi->SetBinContent(destI,rho);
      }
      return rhoMax;
   }
}

Double_t TUnfold::GetRhoIFromMatrix(TH1 *rhoi,const TMatrixDSparse *eOrig,
                                    const Int_t *binMap,TH2 *invEmat) const
{
   // get global correlation coefficients with arbitrary min map
   //   rhoi: global correlation histogram
   //   emat: error matrix
   //   binMap: for each bin of the original output distribution
   //           specify the destination bin. A value of -1 means that the bin
   //           is discarded. 0 means underflow bin, 1 first bin, ...
   //        binMap[0] : destination of underflow bin
   //        binMap[1] : destination of first bin
   //          ...
   // return value: maximum global correlation

   Double_t rhoMax=0.;
   // original number of bins:
   //    fHistToX.GetSize()
   // loop over binMap and find number of used bins

   Int_t binMapSize = fHistToX.GetSize();

   // histToLocalBin[iBin] points to a local index
   //    only bins iBin of the histogram rhoi whih are referenced
   //    in the bin map have a local index
   std::map<int,int> histToLocalBin;
   Int_t nBin=0;
   for(Int_t i=0;i<binMapSize;i++) {
      Int_t mapped_i=binMap ? binMap[i] : i;
      if(mapped_i>=0) {
         if(histToLocalBin.find(mapped_i)==histToLocalBin.end()) {
            histToLocalBin[mapped_i]=nBin;
            nBin++;
         }
      }
   }
   // number of local indices: nBin
   if(nBin>0) {
      // construct inverse mapping function
      //    local index -> histogram bin
      Int_t *localBinToHist=new Int_t[nBin];
      for(std::map<int,int>::const_iterator i=histToLocalBin.begin();
          i!=histToLocalBin.end();i++) {
         localBinToHist[(*i).second]=(*i).first;
      }

      const Int_t *rows_eOrig=eOrig->GetRowIndexArray();
      const Int_t *cols_eOrig=eOrig->GetColIndexArray();
      const Double_t *data_eOrig=eOrig->GetMatrixArray();

      // remap error matrix
      //   matrix row  i  -> origI  (fXToHist[i])
      //   origI  -> destI  (binMap)
      //   destI -> ematBinI  (histToLocalBin)
      TMatrixD eRemap(nBin,nBin);
      // i:  row of the matrix eOrig
      for(Int_t i=0;i<GetNx();i++) {
         // origI: pointer in output histogram with all bins
         Int_t origI=fXToHist[i];
         // destI: pointer in the histogram rhoi
         Int_t destI=binMap ? binMap[origI] : origI;
         if(destI<0) continue;
         Int_t ematBinI=histToLocalBin[destI];
         for(int index_eOrig=rows_eOrig[i];index_eOrig<rows_eOrig[i+1];
             index_eOrig++) {
            // j: column of the matrix fVxx
            Int_t j=cols_eOrig[index_eOrig];
            // origJ: pointer in output histogram with all bins
            Int_t origJ=fXToHist[j];
            // destJ: pointer in the histogram rhoi
            Int_t destJ=binMap ? binMap[origJ] : origJ;
            if(destJ<0) continue;
            Int_t ematBinJ=histToLocalBin[destJ];
            eRemap(ematBinI,ematBinJ) += data_eOrig[index_eOrig];
         }
      }
      // invert remapped error matrix
      TMatrixDSparse eSparse(eRemap);
      Int_t rank=0;
      TMatrixDSparse *einvSparse=InvertMSparseSymmPos(&eSparse,&rank);
      if(rank!=einvSparse->GetNrows()) {
         Warning("GetRhoIFormMatrix","Covariance matrix has rank %d expect %d",
                 rank,einvSparse->GetNrows());
      }
      // fill to histogram
      const Int_t *rows_eInv=einvSparse->GetRowIndexArray();
      const Int_t *cols_eInv=einvSparse->GetColIndexArray();
      const Double_t *data_eInv=einvSparse->GetMatrixArray();

      for(Int_t i=0;i<einvSparse->GetNrows();i++) {
         Double_t e_ii=eRemap(i,i);
         Double_t einv_ii=0.0;
         for(Int_t index_einv=rows_eInv[i];index_einv<rows_eInv[i+1];
             index_einv++) {
            Int_t j=cols_eInv[index_einv];
            if(j==i) {
               einv_ii=data_eInv[index_einv];
            }
            if(invEmat) {
               invEmat->SetBinContent(localBinToHist[i],localBinToHist[j],
                                      data_eInv[index_einv]);
            }
         }
         Double_t rho=1.0;
         if((einv_ii>0.0)&&(e_ii>0.0)) rho=1.-1./(einv_ii*e_ii);
         if(rho>=0.0) rho=TMath::Sqrt(rho);
         else rho=-TMath::Sqrt(-rho);
         if(rho>rhoMax) {
            rhoMax = rho;
         }
         //std::cout<<i<<" "<<localBinToHist[i]<<" "<<rho<<"\n";
         rhoi->SetBinContent(localBinToHist[i],rho);
      }
      
      DeleteMatrix(&einvSparse);
      delete [] localBinToHist;
   }
   return rhoMax;
}

void TUnfold::ClearHistogram(TH1 *h,Double_t x) const
{
   // clear histogram contents and error
   Int_t nxyz[3];
   nxyz[0]=h->GetNbinsX()+1;
   nxyz[1]=h->GetNbinsY()+1;
   nxyz[2]=h->GetNbinsZ()+1;
   for(int i=h->GetDimension();i<3;i++) nxyz[i]=0;
   Int_t ixyz[3];
   for(int i=0;i<3;i++) ixyz[i]=0;
   while((ixyz[0]<=nxyz[0])&&
         (ixyz[1]<=nxyz[1])&&
         (ixyz[2]<=nxyz[2])){
      Int_t ibin=h->GetBin(ixyz[0],ixyz[1],ixyz[2]);
      h->SetBinContent(ibin,x);
      h->SetBinError(ibin,0.0);
      for(Int_t i=0;i<3;i++) {
         ixyz[i] += 1;
         if(ixyz[i]<=nxyz[i]) break;
         if(i<2) ixyz[i]=0;
      }
   }
}

void TUnfold::SetEpsMatrix(Double_t eps) {
   // set accuracy for matrix inversion
   if((eps>0.0)&&(eps<1.0)) fEpsMatrix=eps;
}
