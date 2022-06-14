// @(#)root/matrix:$Id$
// Authors: Fons Rademakers, Eddy Offermann  Dec 2003

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TDecompLU.h"
#include "TMath.h"

ClassImp(TDecompLU);

/** \class TDecompLU
    \ingroup Matrix

 LU Decomposition class

 Decompose  a general n x n matrix A into P A = L U

 where P is a permutation matrix, L is unit lower triangular and U
 is upper triangular.
 L is stored in the strict lower triangular part of the matrix fLU.
 The diagonal elements of L are unity and are not stored.
 U is stored in the diagonal and upper triangular part of the matrix
 fU.
 P is stored in the index array fIndex : j = fIndex[i] indicates that
 row j and row i should be swapped .

 fSign gives the sign of the permutation, (-1)^n, where n is the
 number of interchanges in the permutation.

 fLU has the same indexing range as matrix A .

 The decomposition fails if a diagonal element of abs(fLU) is == 0,
 The matrix fUL is made invalid .
*/

////////////////////////////////////////////////////////////////////////////////
/// Default constructor

TDecompLU::TDecompLU()
{
   fSign = 0.0;
   fNIndex = 0;
   fIndex = 0;
   fImplicitPivot = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor for (nrows x nrows) matrix

TDecompLU::TDecompLU(Int_t nrows)
{
   fSign = 1.0;
   fNIndex = nrows;
   fIndex = new Int_t[fNIndex];
   memset(fIndex,0,fNIndex*sizeof(Int_t));
   fImplicitPivot = 0;
   fLU.ResizeTo(nrows,nrows);
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor for ([row_lwb..row_upb] x [row_lwb..row_upb]) matrix

TDecompLU::TDecompLU(Int_t row_lwb,Int_t row_upb)
{
   const Int_t nrows = row_upb-row_lwb+1;
   fSign = 1.0;
   fNIndex = nrows;
   fIndex = new Int_t[fNIndex];
   memset(fIndex,0,fNIndex*sizeof(Int_t));
   fRowLwb = row_lwb;
   fColLwb = row_lwb;
   fImplicitPivot = 0;
   fLU.ResizeTo(row_lwb,row_lwb+nrows-1,row_lwb,row_lwb+nrows-1);
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor for matrix a

TDecompLU::TDecompLU(const TMatrixD &a,Double_t tol,Int_t implicit)
{
   R__ASSERT(a.IsValid());

   if (a.GetNrows() != a.GetNcols() || a.GetRowLwb() != a.GetColLwb()) {
      Error("TDecompLU(const TMatrixD &","matrix should be square");
      return;
   }

   SetBit(kMatrixSet);
   fCondition = a.Norm1();
   fImplicitPivot = implicit;
   fTol = a.GetTol();
   if (tol > 0)
      fTol = tol;

   fSign = 1.0;
   fNIndex = a.GetNcols();
   fIndex = new Int_t[fNIndex];
   memset(fIndex,0,fNIndex*sizeof(Int_t));

   fRowLwb = a.GetRowLwb();
   fColLwb = a.GetColLwb();
   fLU.ResizeTo(a);
   fLU = a;
}

////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

TDecompLU::TDecompLU(const TDecompLU &another) : TDecompBase(another)
{
   fNIndex = 0;
   fIndex  = 0;
   *this = another;
}

////////////////////////////////////////////////////////////////////////////////
/// Matrix A is decomposed in components U and L so that P * A = U * L
/// If the decomposition succeeds, bit kDecomposed is set , otherwise kSingular

Bool_t TDecompLU::Decompose()
{
   if (TestBit(kDecomposed)) return kTRUE;

   if ( !TestBit(kMatrixSet) ) {
      Error("Decompose()","Matrix has not been set");
      return kFALSE;
   }

   Int_t nrZeros = 0;
   Bool_t ok;
   if (fImplicitPivot)
      ok = DecomposeLUCrout(fLU,fIndex,fSign,fTol,nrZeros);
   else
      ok = DecomposeLUGauss(fLU,fIndex,fSign,fTol,nrZeros);

   if (!ok) SetBit(kSingular);
   else     SetBit(kDecomposed);

   return ok;
}

////////////////////////////////////////////////////////////////////////////////
/// Reconstruct the original matrix using the decomposition parts

const TMatrixD TDecompLU::GetMatrix()
{
   if (TestBit(kSingular)) {
      Error("GetMatrix()","Matrix is singular");
      return TMatrixD();
   }
   if ( !TestBit(kDecomposed) ) {
      if (!Decompose()) {
         Error("GetMatrix()","Decomposition failed");
         return TMatrixD();
      }
   }

   TMatrixD mL = fLU;
   TMatrixD mU = fLU;
   Double_t * const pU = mU.GetMatrixArray();
   Double_t * const pL = mL.GetMatrixArray();
   const Int_t n = fLU.GetNcols();
   for (Int_t irow = 0; irow < n; irow++) {
      const Int_t off_row = irow*n;
         for (Int_t icol = 0; icol < n; icol++) {
            if (icol > irow)      pL[off_row+icol] = 0.;
            else if (icol < irow) pU[off_row+icol] = 0.;
            else                  pL[off_row+icol] = 1.;
         }
      }

   TMatrixD a = mL*mU;

   // swap rows

   Double_t * const pA = a.GetMatrixArray();
   for (Int_t i = n-1; i >= 0; i--) {
      const Int_t j = fIndex[i];
      if (j != i) {
         const Int_t off_j = j*n;
         const Int_t off_i = i*n;
         for (Int_t k = 0; k < n; k++) {
            const Double_t tmp = pA[off_j+k];
            pA[off_j+k] = pA[off_i+k];
            pA[off_i+k] = tmp;
         }
      }
   }

   return a;
}

////////////////////////////////////////////////////////////////////////////////
/// Set matrix to be decomposed

void TDecompLU::SetMatrix(const TMatrixD &a)
{
   R__ASSERT(a.IsValid());

   ResetStatus();
   if (a.GetNrows() != a.GetNcols() || a.GetRowLwb() != a.GetColLwb()) {
      Error("TDecompLU(const TMatrixD &","matrix should be square");
      return;
   }

   SetBit(kMatrixSet);
   fCondition = a.Norm1();

   fSign = 1.0;
   if (fNIndex != a.GetNcols()) {
      fNIndex = a.GetNcols();
      delete [] fIndex;
      fIndex = new Int_t[fNIndex];
      memset(fIndex,0,fNIndex*sizeof(Int_t));
   }

   fRowLwb = a.GetRowLwb();
   fColLwb = a.GetColLwb();
   fLU.ResizeTo(a);
   fLU = a;
}

////////////////////////////////////////////////////////////////////////////////
/// Solve Ax=b assuming the LU form of A is stored in fLU, but assume b has *not*
/// been transformed.  Solution returned in b.

Bool_t TDecompLU::Solve(TVectorD &b)
{
   R__ASSERT(b.IsValid());
   if (TestBit(kSingular)) {
      Error("Solve()","Matrix is singular");
      return kFALSE;
   }
   if ( !TestBit(kDecomposed) ) {
      if (!Decompose()) {
         Error("Solve()","Decomposition failed");
         return kFALSE;
      }
   }

   if (fLU.GetNrows() != b.GetNrows() || fLU.GetRowLwb() != b.GetLwb()) {
      Error("Solve(TVectorD &","vector and matrix incompatible");
      return kFALSE;
   }

   const Int_t n = fLU.GetNrows();

   const Double_t *pLU = fLU.GetMatrixArray();
   Double_t *pb  = b.GetMatrixArray();

   Int_t i;

   // Check for zero diagonals
   for (i = 0; i < n ; i++) {
      const Int_t off_i = i*n;
      if (TMath::Abs(pLU[off_i+i]) < fTol) {
         Error("Solve(TVectorD &b)","LU[%d,%d]=%.4e < %.4e",i,i,pLU[off_i+i],fTol);
         return kFALSE;
      }
   }

   // Transform b allowing for leading zeros
   Int_t nonzero = -1;
   for (i = 0; i < n; i++) {
      const Int_t off_i = i*n;
      const Int_t iperm = fIndex[i];
      Double_t r = pb[iperm];
      pb[iperm] = pb[i];
      if (nonzero >= 0)
         for (Int_t j = nonzero; j < i; j++)
            r -= pLU[off_i+j]*pb[j];
      else if (r != 0.0)
         nonzero = i;
      pb[i] = r;
   }

   // Backward substitution
   for (i = n-1; i >= 0; i--) {
      const Int_t off_i = i*n;
      Double_t r = pb[i];
      for (Int_t j = i+1; j < n; j++)
         r -= pLU[off_i+j]*pb[j];
      pb[i] = r/pLU[off_i+i];
   }

   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Solve Ax=b assuming the LU form of A is stored in fLU, but assume b has *not*
/// been transformed.  Solution returned in b.

Bool_t TDecompLU::Solve(TMatrixDColumn &cb)
{
   TMatrixDBase *b = const_cast<TMatrixDBase *>(cb.GetMatrix());
   R__ASSERT(b->IsValid());
   if (TestBit(kSingular)) {
      Error("Solve()","Matrix is singular");
      return kFALSE;
   }
   if ( !TestBit(kDecomposed) ) {
      if (!Decompose()) {
         Error("Solve()","Decomposition failed");
         return kFALSE;
      }
   }

   if (fLU.GetNrows() != b->GetNrows() || fLU.GetRowLwb() != b->GetRowLwb()) {
      Error("Solve(TMatrixDColumn &","vector and matrix incompatible");
      return kFALSE;
   }

   const Int_t     n   = fLU.GetNrows();
   const Double_t *pLU = fLU.GetMatrixArray();

   Int_t i;

   // Check for zero diagonals
   for (i = 0; i < n ; i++) {
      const Int_t off_i = i*n;
      if (TMath::Abs(pLU[off_i+i]) < fTol) {
         Error("Solve(TMatrixDColumn &cb)","LU[%d,%d]=%.4e < %.4e",i,i,pLU[off_i+i],fTol);
         return kFALSE;
      }
   }

   Double_t *pcb = cb.GetPtr();
   const Int_t     inc = cb.GetInc();

   // Transform b allowing for leading zeros
   Int_t nonzero = -1;
   for (i = 0; i < n; i++) {
      const Int_t off_i  = i*n;
      const Int_t off_i2 = i*inc;
      const Int_t iperm = fIndex[i];
      const Int_t off_iperm = iperm*inc;
      Double_t r = pcb[off_iperm];
      pcb[off_iperm] = pcb[off_i2];
      if (nonzero >=0)
         for (Int_t j = nonzero; j <= i-1; j++)
            r -= pLU[off_i+j]*pcb[j*inc];
      else if (r != 0.0)
         nonzero = i;
      pcb[off_i2] = r;
   }

   // Backward substitution
   for (i = n-1; i >= 0; i--) {
      const Int_t off_i  = i*n;
      const Int_t off_i2 = i*inc;
      Double_t r = pcb[off_i2];
      for (Int_t j = i+1; j < n; j++)
         r -= pLU[off_i+j]*pcb[j*inc];
      pcb[off_i2] = r/pLU[off_i+i];
   }

   return kTRUE;
}


////////////////////////////////////////////////////////////////////////////////
/// Solve A^T x=b assuming the LU form of A^T is stored in fLU, but assume b has *not*
/// been transformed.  Solution returned in b.

Bool_t TDecompLU::TransSolve(TVectorD &b)
{
   R__ASSERT(b.IsValid());
   if (TestBit(kSingular)) {
      Error("TransSolve()","Matrix is singular");
      return kFALSE;
   }
   if ( !TestBit(kDecomposed) ) {
      if (!Decompose()) {
         Error("TransSolve()","Decomposition failed");
         return kFALSE;
      }
   }

   if (fLU.GetNrows() != b.GetNrows() || fLU.GetRowLwb() != b.GetLwb()) {
      Error("TransSolve(TVectorD &","vector and matrix incompatible");
      return kFALSE;
   }

   const Int_t n = fLU.GetNrows();

   const Double_t *pLU = fLU.GetMatrixArray();
   Double_t *pb  = b.GetMatrixArray();

   Int_t i;

   // Check for zero diagonals
   for (i = 0; i < n ; i++) {
      const Int_t off_i = i*n;
      if (TMath::Abs(pLU[off_i+i]) < fTol) {
         Error("TransSolve(TVectorD &b)","LU[%d,%d]=%.4e < %.4e",i,i,pLU[off_i+i],fTol);
         return kFALSE;
      }
   }

   // Forward Substitution
   for (i = 0; i < n; i++) {
      const Int_t off_i = i*n;
      Double_t r = pb[i];
      for (Int_t j = 0; j < i ; j++) {
         const Int_t off_j = j*n;
         r -= pLU[off_j+i]*pb[j];
      }
      pb[i] = r/pLU[off_i+i];
   }

   // Transform b allowing for leading zeros
   Int_t nonzero = -1;
   for (i = n-1 ; i >= 0; i--) {
      Double_t r = pb[i];
      if (nonzero >= 0) {
         for (Int_t j = i+1; j <= nonzero; j++) {
            const Int_t off_j = j*n;
            r -= pLU[off_j+i]*pb[j];
         }
      } else if (r != 0.0)
         nonzero = i;
      const Int_t iperm = fIndex[i];
      pb[i]     = pb[iperm];
      pb[iperm] = r;
   }

   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Solve A^T x=b assuming the LU form of A^T is stored in fLU, but assume b has *not*
/// been transformed.  Solution returned in b.

Bool_t TDecompLU::TransSolve(TMatrixDColumn &cb)
{
   TMatrixDBase *b = const_cast<TMatrixDBase *>(cb.GetMatrix());
   R__ASSERT(b->IsValid());
   if (TestBit(kSingular)) {
      Error("TransSolve()","Matrix is singular");
      return kFALSE;
   }
   if ( !TestBit(kDecomposed) ) {
      if (!Decompose()) {
         Error("TransSolve()","Decomposition failed");
         return kFALSE;
      }
   }

   if (fLU.GetNrows() != b->GetNrows() || fLU.GetRowLwb() != b->GetRowLwb()) {
      Error("TransSolve(TMatrixDColumn &","vector and matrix incompatible");
      return kFALSE;
   }

   const Int_t n   = fLU.GetNrows();
   const Int_t lwb = fLU.GetRowLwb();

   const Double_t *pLU = fLU.GetMatrixArray();

   Int_t i;

   // Check for zero diagonals
   for (i = 0; i < n ; i++) {
      const Int_t off_i = i*n;
      if (TMath::Abs(pLU[off_i+i]) < fTol) {
         Error("TransSolve(TMatrixDColumn &cb)","LU[%d,%d]=%.4e < %.4e",i,i,pLU[off_i+i],fTol);
         return kFALSE;
      }
   }

   // Forward Substitution
   for (i = 0; i < n; i++) {
      const Int_t off_i = i*n;
      Double_t r = cb(i+lwb);
      for (Int_t j = 0; j < i ; j++) {
         const Int_t off_j = j*n;
         r -= pLU[off_j+i]*cb(j+lwb);
      }
      cb(i+lwb) = r/pLU[off_i+i];
   }

   // Transform b allowing for leading zeros
   Int_t nonzero = -1;
   for (i = n-1 ; i >= 0; i--) {
      Double_t r = cb(i+lwb);
      if (nonzero >= 0) {
         for (Int_t j = i+1; j <= nonzero; j++) {
            const Int_t off_j = j*n;
            r -= pLU[off_j+i]*cb(j+lwb);
         }
      } else if (r != 0.0)
         nonzero = i;
      const Int_t iperm = fIndex[i];
      cb(i+lwb)     = cb(iperm+lwb);
      cb(iperm+lwb) = r;
   }

   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Calculate determinant det = d1*TMath::Power(2.,d2)

void TDecompLU::Det(Double_t &d1,Double_t &d2)
{
   if ( !TestBit(kDetermined) ) {
      if ( !TestBit(kDecomposed) )
         Decompose();
      TDecompBase::Det(d1,d2);
      fDet1 *= fSign;
      SetBit(kDetermined);
   }
   d1 = fDet1;
   d2 = fDet2;
}

////////////////////////////////////////////////////////////////////////////////
/// For a matrix A(m,m), its inverse A_inv is defined as A * A_inv = A_inv * A = unit
/// (m x m) Ainv is returned .

Bool_t TDecompLU::Invert(TMatrixD &inv)
{
   if (inv.GetNrows()  != GetNrows()  || inv.GetNcols()  != GetNcols() ||
        inv.GetRowLwb() != GetRowLwb() || inv.GetColLwb() != GetColLwb()) {
      Error("Invert(TMatrixD &","Input matrix has wrong shape");
      return kFALSE;
   }

   inv.UnitMatrix();
   const Bool_t status = MultiSolve(inv);

   return status;
}

////////////////////////////////////////////////////////////////////////////////
/// For a matrix A(m,n), its inverse A_inv is defined as A * A_inv = A_inv * A = unit
/// (n x m) Ainv is returned .

TMatrixD TDecompLU::Invert(Bool_t &status)
{
   const Int_t rowLwb = GetRowLwb();
   const Int_t rowUpb = rowLwb+GetNrows()-1;

   TMatrixD inv(rowLwb,rowUpb,rowLwb,rowUpb);
   inv.UnitMatrix();
   status = MultiSolve(inv);

   return inv;
}

////////////////////////////////////////////////////////////////////////////////
///Print internals of this object

void TDecompLU::Print(Option_t *opt) const
{
   TDecompBase::Print(opt);
   printf("fImplicitPivot = %d\n",fImplicitPivot);
   printf("fSign          = %f\n",fSign);
   printf("fIndex:\n");
   for (Int_t i = 0; i < fNIndex; i++)
      printf("[%d] = %d\n",i,fIndex[i]);
   fLU.Print("fLU");
}

////////////////////////////////////////////////////////////////////////////////
///assignment operator

TDecompLU &TDecompLU::operator=(const TDecompLU &source)
{
   if (this != &source) {
      TDecompBase::operator=(source);
      fLU.ResizeTo(source.fLU);
      fLU   = source.fLU;
      fSign = source.fSign;
      fImplicitPivot = source.fImplicitPivot;
      if (fNIndex != source.fNIndex) {
         if (fIndex)
            delete [] fIndex;
         fNIndex = source.fNIndex;
         fIndex = new Int_t[fNIndex];
      }
      if (fIndex) memcpy(fIndex,source.fIndex,fNIndex*sizeof(Int_t));
   }
   return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// Crout/Doolittle algorithm of LU decomposing a square matrix, with implicit partial
/// pivoting.  The decomposition is stored in fLU: U is explicit in the upper triag
/// and L is in multiplier form in the subdiagionals .
/// Row permutations are mapped out in fIndex. fSign, used for calculating the
/// determinant, is +/- 1 for even/odd row permutations. .

Bool_t TDecompLU::DecomposeLUCrout(TMatrixD &lu,Int_t *index,Double_t &sign,
                                   Double_t tol,Int_t &nrZeros)
{
   const Int_t     n     = lu.GetNcols();
   Double_t *pLU   = lu.GetMatrixArray();

   Double_t work[kWorkMax];
   Double_t *allocated = nullptr;
   Double_t *scale = work;
   if (n > kWorkMax) {
      allocated = new Double_t[n];
      scale = allocated;
   }

   sign    = 1.0;
   nrZeros = 0;
   // Find implicit scaling factors for each row
   for (Int_t i = 0; i < n ; i++) {
      const Int_t off_i = i*n;
      Double_t max = 0.0;
      for (Int_t j = 0; j < n; j++) {
         const Double_t tmp = TMath::Abs(pLU[off_i+j]);
         if (tmp > max)
            max = tmp;
      }
      scale[i] = (max == 0.0 ? 0.0 : 1.0/max);
   }

   for (Int_t j = 0; j < n; j++) {
      const Int_t off_j = j*n;
      // Run down jth column from top to diag, to form the elements of U.
      for (Int_t i = 0; i < j; i++) {
         const Int_t off_i = i*n;
         Double_t r = pLU[off_i+j];
         for (Int_t k = 0; k < i; k++) {
            const Int_t off_k = k*n;
            r -= pLU[off_i+k]*pLU[off_k+j];
         }
         pLU[off_i+j] = r;
      }

      // Run down jth subdiag to form the residuals after the elimination of
      // the first j-1 subdiags.  These residuals divided by the appropriate
      // diagonal term will become the multipliers in the elimination of the jth.
      // subdiag. Find fIndex of largest scaled term in imax.

      Double_t max = 0.0;
      Int_t imax = 0;
      for (Int_t i = j; i < n; i++) {
         const Int_t off_i = i*n;
         Double_t r = pLU[off_i+j];
         for (Int_t k = 0; k < j; k++) {
            const Int_t off_k = k*n;
            r -= pLU[off_i+k]*pLU[off_k+j];
         }
         pLU[off_i+j] = r;
         const Double_t tmp = scale[i]*TMath::Abs(r);
         if (tmp >= max) {
            max = tmp;
            imax = i;
         }
      }

      // Permute current row with imax
      if (j != imax) {
         const Int_t off_imax = imax*n;
         for (Int_t k = 0; k < n; k++ ) {
            const Double_t tmp = pLU[off_imax+k];
            pLU[off_imax+k] = pLU[off_j+k];
            pLU[off_j+k]    = tmp;
         }
         sign = -sign;
         scale[imax] = scale[j];
      }
      index[j] = imax;

      // If diag term is not zero divide subdiag to form multipliers.
      if (pLU[off_j+j] != 0.0) {
         if (TMath::Abs(pLU[off_j+j]) < tol)
            nrZeros++;
         if (j != n-1) {
            const Double_t tmp = 1.0/pLU[off_j+j];
            for (Int_t i = j+1; i < n; i++) {
               const Int_t off_i = i*n;
               pLU[off_i+j] *= tmp;
            }
         }
      } else {
         ::Error("TDecompLU::DecomposeLUCrout","matrix is singular");
         if (allocated)  delete [] allocated;
         return kFALSE;
      }
   }

   if (allocated)
      delete [] allocated;

   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// LU decomposition using Gaussian Elimination with partial pivoting (See Golub &
/// Van Loan, Matrix Computations, Algorithm 3.4.1) of a square matrix .
/// The decomposition is stored in fLU: U is explicit in the upper triag and L is in
/// multiplier form in the subdiagionals . Row permutations are mapped out in fIndex.
/// fSign, used for calculating the determinant, is +/- 1 for even/odd row permutations.
/// Since this algorithm uses partial pivoting without scaling like in Crout/Doolitle.
/// it is somewhat faster but less precise .

Bool_t TDecompLU::DecomposeLUGauss(TMatrixD &lu,Int_t *index,Double_t &sign,
                                   Double_t tol,Int_t &nrZeros)
{
   const Int_t     n   = lu.GetNcols();
   Double_t *pLU = lu.GetMatrixArray();

   sign    = 1.0;
   nrZeros = 0;

   index[n-1] = n-1;
   for (Int_t j = 0; j < n-1; j++) {
      const Int_t off_j = j*n;

      // Find maximum in the j-th column

      Double_t max = TMath::Abs(pLU[off_j+j]);
      Int_t i_pivot = j;

      for (Int_t i = j+1; i < n; i++) {
         const Int_t off_i = i*n;
         const Double_t mLUij = TMath::Abs(pLU[off_i+j]);

         if (mLUij > max) {
            max = mLUij;
            i_pivot = i;
         }
      }

      if (i_pivot != j) {
         const Int_t off_ipov = i_pivot*n;
         for (Int_t k = 0; k < n; k++ ) {
            const Double_t tmp = pLU[off_ipov+k];
            pLU[off_ipov+k] = pLU[off_j+k];
            pLU[off_j+k]    = tmp;
         }
         sign = -sign;
      }
      index[j] = i_pivot;

      const Double_t mLUjj = pLU[off_j+j];

      if (mLUjj != 0.0) {
         if (TMath::Abs(mLUjj) < tol)
            nrZeros++;
         for (Int_t i = j+1; i < n; i++) {
            const Int_t off_i = i*n;
            const Double_t mLUij = pLU[off_i+j]/mLUjj;
            pLU[off_i+j] = mLUij;

            for (Int_t k = j+1; k < n; k++) {
               const Double_t mLUik = pLU[off_i+k];
               const Double_t mLUjk = pLU[off_j+k];
               pLU[off_i+k] = mLUik-mLUij*mLUjk;
            }
         }
      } else {
         ::Error("TDecompLU::DecomposeLUGauss","matrix is singular");
         return kFALSE;
      }
   }

   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Calculate matrix inversion through in place forward/backward substitution

Bool_t TDecompLU::InvertLU(TMatrixD &lu,Double_t tol,Double_t *det)
{
   if (det)
      *det = 0.0;

   if (lu.GetNrows() != lu.GetNcols() || lu.GetRowLwb() != lu.GetColLwb()) {
      ::Error("TDecompLU::InvertLU","matrix should be square");
      return kFALSE;
   }

   const Int_t     n   = lu.GetNcols();
   Double_t *pLU = lu.GetMatrixArray();

   Int_t worki[kWorkMax];
   Int_t *allocatedI = nullptr;
   Int_t *index = worki;
   if (n > kWorkMax) {
      allocatedI = new Int_t[n];
      index = allocatedI;
   }

   Double_t sign = 1.0;
   Int_t nrZeros = 0;
   if (!DecomposeLUCrout(lu,index,sign,tol,nrZeros) || nrZeros > 0) {
      if (allocatedI)
         delete [] allocatedI;
      ::Error("TDecompLU::InvertLU","matrix is singular, %d diag elements < tolerance of %.4e",nrZeros,tol);
      return kFALSE;
   }

   if (det) {
      Double_t d1;
      Double_t d2;
      const TVectorD diagv = TMatrixDDiag_const(lu);
      DiagProd(diagv,tol,d1,d2);
      d1 *= sign;
      *det = d1*TMath::Power(2.0,d2);
   }

   //  Form inv(U).

   Int_t j;

   for (j = 0; j < n; j++) {
      const Int_t off_j = j*n;

      pLU[off_j+j] = 1./pLU[off_j+j];
      const Double_t mLU_jj = -pLU[off_j+j];

//    Compute elements 0:j-1 of j-th column.

      Double_t *pX = pLU+j;
      Int_t k;
      for (k = 0; k <= j-1; k++) {
         const Int_t off_k = k*n;
         if ( pX[off_k] != 0.0 ) {
            const Double_t tmp = pX[off_k];
            for (Int_t i = 0; i <= k-1; i++) {
               const Int_t off_i = i*n;
               pX[off_i] += tmp*pLU[off_i+k];
            }
            pX[off_k] *= pLU[off_k+k];
         }
      }
      for (k = 0; k <= j-1; k++) {
         const Int_t off_k = k*n;
         pX[off_k] *= mLU_jj;
      }
   }

   // Solve the equation inv(A)*L = inv(U) for inv(A).

   Double_t workd[kWorkMax];
   Double_t *allocatedD = nullptr;
   Double_t *pWorkd = workd;
   if (n > kWorkMax) {
      allocatedD = new Double_t[n];
      pWorkd = allocatedD;
   }

   for (j = n-1; j >= 0; j--) {

      // Copy current column j of L to WORK and replace with zeros.
      for (Int_t i = j+1; i < n; i++) {
         const Int_t off_i = i*n;
         pWorkd[i] = pLU[off_i+j];
         pLU[off_i+j] = 0.0;
      }

      // Compute current column of inv(A).

      if (j < n-1) {
         const Double_t *mp = pLU+j+1;  // Matrix row ptr
         Double_t *tp = pLU+j;          // Target vector ptr

         for (Int_t irow = 0; irow < n; irow++) {
            Double_t sum = 0.;
            const Double_t *sp = pWorkd+j+1; // Source vector ptr
            for (Int_t icol = 0; icol < n-1-j ; icol++)
               sum += *mp++ * *sp++;
            *tp = -sum + *tp;
            mp += j+1;
            tp += n;
         }
      }
   }

   if (allocatedD)
      delete [] allocatedD;

   // Apply column interchanges.
   for (j = n-1; j >= 0; j--) {
      const Int_t jperm = index[j];
      if (jperm != j) {
         for (Int_t i = 0; i < n; i++) {
            const Int_t off_i = i*n;
            const Double_t tmp = pLU[off_i+jperm];
            pLU[off_i+jperm] = pLU[off_i+j];
            pLU[off_i+j]     = tmp;
         }
      }
   }

   if (allocatedI)
      delete [] allocatedI;

   return kTRUE;
}
