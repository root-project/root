// @(#)root/matrix:$Name:  $:$Id: TDecompBase.cxx,v 1.6 2004/02/04 17:12:44 brun Exp $
// Authors: Fons Rademakers, Eddy Offermann  Dec 2003

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

///////////////////////////////////////////////////////////////////////////
//                                                                       //
// Decomposition Base class                                              //
//                                                                       //
// This class forms the base for all the decompositions methods in the   //
// linear algebra package .                                              //
// It or its derived classes have installed the methods to solve         //
// equations,invert matrices and calculate determinants while monitoring //
// the accuracy.                                                         //
//                                                                       //
// When the constructor is called with a "const" matrix, the original    //
// matrix "survives" (of course) . Some classes (LU) have also a non-    //
// "const" constructor available . Here the original matrix is adopted   //
// by the decomposition class (See Adoption in TMatrixDBase) to store the//
// decomposed matrix, thereby avoiding new memory allocation.            //
//                                                                       //
// The decomposition (which is called by the constructor) fails when the //
// matrix is singular  or not positive-definite in case of Cholesky      //
// This can be checked before applying the decomposition by checking the //
// matrix through GetDecompMatrix()                                      //
//                                                                       //
// Each derived class has always the following methods available:        //
//                                                                       //
// Condition() :                                                         //
//   In an iterative scheme the condition number for matrix inversion is //
//   calculated . This number is of interest for estimating the accuracy //
//   of x in the equation Ax=b                                           //
//   For example:                                                        //
//     A is a (10x10) Hilbert matrix which looks deceivingly innocent    //
//     and simple, A(i,j) = 1/(i+j+1)                                    //
//     b(i) = Sum_j A(i,j), so a sum of a row in A                       //
//                                                                       //
//     the solution is x(i) = 1. i=0,.,9                                 //
//                                                                       //
//   However,                                                            //
//     TMatrixD m....; TVectorD b.....                                   //
//     TDecompLU lu(m); lu.SetTol(1.0e-12); lu.Solve(b); b.Print()       //
//   gives,                                                              //
//                                                                       //
//   {1.000,1.000,1.000,1.000,0.998,1.000,0.993,1.001,0.996,1.000}       //
//                                                                       //
//   Looking at the condition number, this is in line with expected the  //
//   accuracy . The condition number is 3.957e+12 . As a simple rule of  //
//   thumb, a condition number of 1.0e+n means that you lose up to n     //
//   digits of accuracy in a solution . Since doubles are stored with 15 //
//   digits, we can expect the accuracy to be as small as 3 digits .     //
//                                                                       //
// Det(Double_t &d1,Double_t &d2)                                        //
//   The determinant is d1*TMath::Power(2.,d2)                           //
//   Expressing the determinant this way makes under/over-flow very      //
//   unlikely .                                                          //
//                                                                       //
// Decompose(const TMatrixDBase &A)                                      //
//   Here the actually decomposition is performed . This method is       //
//   called by each constructor, one can changed the matrix A afterwards //
//   without effecting the decomposition                                 //
//                                                                       //
// Solve(TVectorD &b)                                                    //
//  Solve A x = b . x is supplied through the argument and replaced with //
//  the solution .                                                       //
//                                                                       //
// TransSolve(TVectorD &b)                                               //
//  Solve A^T x = b . x is supplied through the argument and replaced    //
//  with the solution .                                                  //
//                                                                       //
// MultiSolve(TMatrixDBase &B)                                           //
//  Solve A X = B . where X and are now matrices . X is supplied through //
//  the argument and replaced with the solution .                        //
//                                                                       //
// Invert(TMatrixDBase &inv)                                             //
//  This is of course just a call to MultiSolve with as input argument   //
//  the unit matrix . Note that for a matrix a(m,n) with m > n  a        //
//  pseudo-inverse is calculated .                                       //
//                                                                       //
// Tolerances and Scaling                                                //
// ----------------------                                                //
// The tolerance parameter (which is a member of this base class) plays  //
// a crucial role in all operations of the decomposition classes . It    //
// gives the user a powerful tool to monitor and steer the operations    //
// Its default value is sqrt(epsilon) where 1+epsilon = 1                //
//                                                                       //
// If you do not want to be bothered by the following considerations,    //
// like in most other linear algebra packages, just set the tolerance    //
// with SetTol to an arbitrary small number .                            //
//                                                                       //
// The tolerance number is used by each decomposition method to decide   //
// whether the matrix is near singular, except of course SVD which can   //
// handle singular matrices .                                            //
// For each decomposition this will be checked in a different way; in LU //
// the matrix is considered singular when, at some point in the          //
// decomposition, a diagonal element < fTol . Therefore, we had to set in//
// the example above of the (10x10) Hilbert, which is near singular, the //
// tolerance on 10e-12 . (The fact that we have to set the tolerance <   //
// sqrt(epsilon) is a clear indication that we are losing precision .)   //
//                                                                       //
// If the matrix is flagged as being singular, operations with the       //
// decomposition will fail and will return matrices/vectors that are     //
// invalid .                                                             //
//                                                                       //
// The observant reader will notice that by scaling the complete matrix  //
// by some small number the decomposition will detect a singular matrix .//
// In this case the user will have to reduce the tolerance number by this//
// factor . (For CPU time saving we decided not to make this an automatic//
// procedure) .                                                          //
//                                                                       //
// Code for this could look as follows:                                  //
// const TMatrixD ab = Abs(a);                                           //
// const Int_t imax = TMath::LocMax(ab.GetNoElements(),                  //
//                                    ab.GetMatrixArray());              //
// const Double_t max_abs = ab.GetMatrixArray()[imax];                   //
// const Double_t scale = TMath::Min(max_abs,1.);                        //
// a.SetTol(a.GetTol()*scale);                                           //
//                                                                       //
// For usage examples see $ROOTSYS/test/stressLinear.cxx                 //
///////////////////////////////////////////////////////////////////////////

#include "TDecompBase.h"

ClassImp(TDecompBase)

//______________________________________________________________________________
TDecompBase::TDecompBase()
{
  fStatus    = kInit;
//  fTol       = std::numerical_limits<double>::epsilon();
  fTol       = DBL_EPSILON;
  fDet1      = 0;
  fDet2      = 0;
  fCondition = 0;
}

//______________________________________________________________________________
TDecompBase::TDecompBase(const TDecompBase &another) : TObject(another)
{
  *this = another;
}

//______________________________________________________________________________
Int_t TDecompBase::Hager(Double_t &est,Int_t iter)
{

// Estimates lower bound for norm1 of inverse of A. Returns norm
// estimate in est.  iter sets the maximum number of iterations to be used.
// The return value indicates the number of iterations remaining on exit from
// loop, hence if this is non-zero the processed "converged".
// This routine uses Hager's Convex Optimisation Algorithm.
// See Applied Numerical Linear Algebra, p139 & SIAM J Sci Stat Comp 1984 pp 311-16

  const TMatrixD &m = GetDecompMatrix();
  Assert(m.IsValid());

  const Int_t n = m.GetNrows();

  TVectorD b(n); TVectorD y(n); TVectorD z(n);
  b = Double_t(1.0/n);
  est = -1.0;
  Double_t inv_norm1 = 0.0;
  Bool_t stop = kFALSE;
  do {
    y = b;
    if (!Solve(y))
      return iter;
    const Double_t ynorm1 = y.Norm1();
    if ( ynorm1 <= inv_norm1 ) {
      stop = kTRUE;
    } else {
      inv_norm1 = ynorm1;
      Int_t i;
      for (i = 0; i < n; i++)
        z(i) = ( y(i) >= 0.0 ? 1.0 : -1.0 );
      if (!TransSolve(z))
        return iter;
      Int_t imax = 0;
      Double_t maxz = TMath::Abs(z(0));
      for (i = 1; i < n; i++) {
        const Double_t absz = TMath::Abs(z(i));
        if ( absz > maxz ) {
          maxz = absz;
          imax = i;
        }
      }
      stop = (maxz <= b*z);
      if (!stop) {
        b = 0.0;
        b(imax) = 1.0;
      }
    }
    iter--;
  } while (!stop && iter);
  est = inv_norm1;

  return iter;
}

//______________________________________________________________________________
void TDecompBase::DiagProd(const TVectorD &diag,Double_t tol,Double_t &d1,Double_t &d2)
{

// Returns product of matrix diagonal elements in d1 and d2. d1 is a mantissa and d2
// an exponent for powers of 2. If matrix is in diagonal or triangular-matrix form this
// will be the determinant.
// Based on Bowler, Martin, Peters and Wilkinson in HACLA

  const Double_t zero      = 0.0;
  const Double_t one       = 1.0;
  const Double_t four      = 4.0;
  const Double_t sixteen   = 16.0;
  const Double_t sixteenth = 0.0625;

  const Int_t n = diag.GetNrows();

  Double_t t1 = 1.0;
  Double_t t2 = 0.0;
  for (Int_t i = 0; (i < n) && (t1 != zero); i++) {
    if (TMath::Abs(diag(i)) > tol) {
      t1 *= (Double_t) diag(i);
      while (TMath::Abs(t1) > one) {
        t1 *= sixteenth;
        t2 += four;
      }
      while (TMath::Abs(t1) < sixteenth) {
        t1 *= sixteen;
        t2 -= four;
      }
    } else {
      t1 = zero;
      t2 = zero;
    }
  }
  d1 = t1;
  d2 = t2;

  return;
}

//______________________________________________________________________________
Double_t TDecompBase::Condition()
{
  if ( !(fStatus & kCondition) ) {
    Double_t invNorm;
    if (Hager(invNorm))
      fCondition *= invNorm;
    else {// no convergence in Hager
      Error("Condition()","Hager procedure did NOT converge");
      fCondition = -1;
    }
    fStatus |= kCondition;
  }
  return fCondition;
}

//______________________________________________________________________________
Bool_t TDecompBase::MultiSolve(TMatrixD &B)
{
// Solve set of equations with RHS in columns of B

  const TMatrixD &m = GetDecompMatrix();
  Assert(m.IsValid() && B.IsValid());

  const Int_t colLwb = B.GetColLwb();
  const Int_t colUpb = B.GetColUpb();
  Bool_t status = kTRUE;
  for (Int_t icol = colLwb; icol <= colUpb && status; icol++) {
    TMatrixDColumn b(B,icol);
    status &= Solve(b);
  }

  if (!status)
    B.Invalidate();

  return status;
}

//______________________________________________________________________________
void TDecompBase::Invert(TMatrixD &inv)
{
  // For a matrix A(m,n), its inverse A_inv is defined as A * A_inv = A_inv * A = unit
  // The user should always supply a matrix of size (m x m) !
  // If m > n , only the (n x m) part of the returned (pseudo inverse) matrix
  // should be used .

  if (inv.GetNrows() != GetNrows() || inv.GetNcols() != GetNrows()) {
    Error("Invert(TMatrixDBase &","Input matrix has wrong shape");
    inv.Invalidate();
    return;
  }

  memset(inv.GetMatrixArray(),0,inv.GetNoElements()*sizeof(Double_t));
  TMatrixDDiag(inv,0) = 1.;
  MultiSolve(inv);
}

//______________________________________________________________________________
TMatrixD TDecompBase::Invert()
{   
  // For a matrix A(m,n), its inverse A_inv is defined as A * A_inv = A_inv * A = unit
  // (n x m) Ainv is returned . 

  TMatrixD inv(GetNrows(),GetNrows());
  inv.UnitMatrix();
  MultiSolve(inv);
  inv.ResizeTo(GetNcols(),GetNrows());

  return inv;
}

//______________________________________________________________________________
void TDecompBase::Det(Double_t &d1,Double_t &d2)
{
  if ( !(fStatus & kDetermined) ) {
    if ( fStatus & kSingular ) {
      fDet1 = 0.0;
      fDet2 = 0.0;
    } else {
      const TMatrixD &m = GetDecompMatrix();
      Assert(m.IsValid());
      const TVectorD diagv = TMatrixDDiag_const(m);
      DiagProd(diagv,fTol,fDet1,fDet2);
    }
    fStatus |= kDetermined;
  }
  d1 = fDet1;
  d2 = fDet2;
}

//______________________________________________________________________________
TDecompBase &TDecompBase::operator=(const TDecompBase &source)
{
  if (this != &source) {
    TObject::operator=(source);
    fStatus    = source.fStatus;
    fTol       = source.fTol;
    fDet1      = source.fDet1;
    fDet2      = source.fDet2;
    fCondition = source.fCondition;
  }
  return *this;
}


//______________________________________________________________________________
Bool_t DefHouseHolder(const TVectorD &vc,Int_t lp,Int_t l,Double_t &up,Double_t &beta,
                      Double_t tol)
{
// Define a Householder-transformation through the parameters up and b .

  const Int_t n = vc.GetNrows();
  const Double_t * const vp = vc.GetMatrixArray();

  Double_t c = TMath::Abs(vp[lp]);
  Int_t i;
  for (i = l; i < n; i++)
    c = TMath::Max(TMath::Abs(vp[i]),c);

  up   = 0.0;
  beta = 0.0;
  if (c <= tol) {
//    Warning("DefHouseHolder","max vector=%.4e < %.4e",c,tol);
    return kFALSE;
  }

  Double_t sd = vp[lp]/c; sd *= sd;
  for (i = l; i < n; i++) {
    const Double_t tmp = vp[i]/c;
    sd += tmp*tmp;
  }

  Double_t vpprim = c*TMath::Sqrt(sd);
  if (vp[lp] > 0.) vpprim = -vpprim;
  up = vp[lp]-vpprim;
  beta = 1./(vpprim*up);

  return kTRUE;
}

//______________________________________________________________________________
void ApplyHouseHolder(const TVectorD &vc,Double_t up,Double_t beta,
                      Int_t lp,Int_t l,TMatrixDRow &cr)
{
//  Apply Householder-transformation.

  const Int_t nv = vc.GetNrows();
  const Int_t nc = (cr.GetMatrix())->GetNcols();

  if (nv > nc) {
    Error("ApplyHouseHolder(const TVectorD &,..,TMatrixDRow &)","matrix row too short");
    return;
  }

  const Int_t inc_c = cr.GetInc();
  const Double_t * const vp = vc.GetMatrixArray();
        Double_t *       cp = cr.GetPtr();

  Double_t s = cp[lp*inc_c]*up;
  Int_t i;
  for (i = l; i < nv; i++)
    s += cp[i*inc_c]*vp[i];

  s = s*beta;
  cp[lp*inc_c] += s*up;
  for (i = l; i < nv; i++)
    cp[i*inc_c] += s*vp[i];
}

//______________________________________________________________________________
void ApplyHouseHolder(const TVectorD &vc,Double_t up,Double_t beta,
                      Int_t lp,Int_t l,TMatrixDColumn &cc)
{
//  Apply Householder-transformation.

  const Int_t nv = vc.GetNrows();
  const Int_t nc = (cc.GetMatrix())->GetNrows();

  if (nv > nc) {
    Error("ApplyHouseHolder(const TVectorD &,..,TMatrixDRow &)","matrix column too short");
    return;
  }

  const Int_t inc_c = cc.GetInc();
  const Double_t * const vp = vc.GetMatrixArray();
        Double_t *       cp = cc.GetPtr();

  Double_t s = cp[lp*inc_c]*up;
  Int_t i;
  for (i = l; i < nv; i++)
    s += cp[i*inc_c]*vp[i];

  s = s*beta;
  cp[lp*inc_c] += s*up;
  for (i = l; i < nv; i++)
    cp[i*inc_c] += s*vp[i];
}

//______________________________________________________________________________
void ApplyHouseHolder(const TVectorD &vc,Double_t up,Double_t beta,
                      Int_t lp,Int_t l,TVectorD &cv)
{
//  Apply Householder-transformation.

  const Int_t nv = vc.GetNrows();
  const Int_t nc = cv.GetNrows();

  if (nv > nc) {
    Error("ApplyHouseHolder(const TVectorD &,..,TVectorD &)","vector too short");
    return;
  }

  const Double_t * const vp = vc.GetMatrixArray();
        Double_t *       cp = cv.GetMatrixArray();

  Double_t s = cp[lp]*up;
  Int_t i;
  for (i = l; i < nv; i++)
    s += cp[i]*vp[i];

  s = s*beta;
  cp[lp] += s*up;
  for (i = l; i < nv; i++)
    cp[i] += s*vp[i];
}

//______________________________________________________________________________
void DefGivens(Double_t v1,Double_t v2,Double_t &c,Double_t &s)
{
// Defines a Givens-rotation by calculating 2 rotation parameters c and s.
// The rotation is defined with the vector components v1 and v2.

  const Double_t a1 = TMath::Abs(v1);
  const Double_t a2 = TMath::Abs(v2);
  if (a1 > a2) {
    const Double_t w = v2/v1;
    const Double_t q = TMath::Sqrt(1.+w*w);
    c = 1./q;
    if (v1 < 0.) c = -c;
    s = c*w;
  }
  else {
    if (v2 != 0) {
      const Double_t w = v1/v2;
      const Double_t q = TMath::Sqrt(1.+w*w);
      s = 1./q;
      if (v2 < 0.) s = -s;
      c = s*w;
    }
    else {
      c = 1.;
      s = 0.;
    }
  }
}

//______________________________________________________________________________
void DefAplGivens(Double_t &v1,Double_t &v2,Double_t &c,Double_t &s)
{
// Define and apply a Givens-rotation by calculating 2 rotation
// parameters c and s. The rotation is defined with and applied to the vector
// components v1 and v2.

  const Double_t a1 = TMath::Abs(v1);
  const Double_t a2 = TMath::Abs(v2);
  if (a1 > a2) {
    const Double_t w = v2/v1;
    const Double_t q = TMath::Sqrt(1.+w*w);
    c = 1./q;
    if (v1 < 0.) c = -c;
    s  = c*w;
    v1 = a1*q;
    v2 = 0.;
  }
  else {
    if (v2 != 0) {
      const Double_t w = v1/v2;
      const Double_t q = TMath::Sqrt(1.+w*w);
      s = 1./q;
      if (v2 < 0.) s = -s;
      c  = s*w;
      v1 = a2*q;
      v2 = 0.;
    }
    else {
      c = 1.;
      s = 0.;
    }
  }
}

//______________________________________________________________________________
void ApplyGivens(Double_t &z1,Double_t &z2,Double_t c,Double_t s)
{
// Apply a Givens transformation as defined by c and s to the vector compenents
// v1 and v2 .

  const Double_t w  = z1*c+z2*s;
  z2 = -z1*s+z2*c;
  z1 = w;
}
