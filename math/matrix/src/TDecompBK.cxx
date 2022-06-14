// @(#)root/matrix:$Id$
// Authors: Fons Rademakers, Eddy Offermann  Sep 2004

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TDecompBK.h"
#include "TMath.h"
#include "TError.h"

ClassImp(TDecompBK);

/** \class TDecompBK
    \ingroup Matrix

 The Bunch-Kaufman diagonal pivoting method decomposes a real
 symmetric matrix A using

~~~
     A = U*D*U^T
~~~

  where U is a product of permutation and unit upper triangular
  matrices, U^T is the transpose of U, and D is symmetric and block
  diagonal with 1-by-1 and 2-by-2 diagonal blocks.

     U = P(n-1)*U(n-1)* ... *P(k)U(k)* ...,
  i.e., U is a product of terms P(k)*U(k), where k decreases from n-1
  to 0 in steps of 1 or 2, and D is a block diagonal matrix with 1-by-1
  and 2-by-2 diagonal blocks D(k).  P(k) is a permutation matrix as
  defined by IPIV(k), and U(k) is a unit upper triangular matrix, such
  that if the diagonal block D(k) is of order s (s = 1 or 2), then

~~~
             (   I    v    0   )   k-s
     U(k) =  (   0    I    0   )   s
             (   0    0    I   )   n-k
                k-s   s   n-k
~~~

  If s = 1, D(k) overwrites A(k,k), and v overwrites A(0:k-1,k).
  If s = 2, the upper triangle of D(k) overwrites A(k-1,k-1), A(k-1,k),
  and A(k,k), and v overwrites A(0:k-2,k-1:k).

 fU contains on entry the symmetric matrix A of which only the upper
 triangular part is referenced . On exit fU contains the block diagonal
 matrix D and the multipliers used to obtain the factor U, see above .

 fIpiv if dimension n contains details of the interchanges and the
 the block structure of D . If (fIPiv(k) > 0, then rows and columns k
 and fIPiv(k) were interchanged and D(k,k) is a 1-by-1 diagonal block.
 If IPiv(k) = fIPiv(k-1) < 0, rows and columns k-1 and -IPiv(k) were
 interchanged and D(k-1:k,k-1:k) is a 2-by-2 diagonal block.
*/

////////////////////////////////////////////////////////////////////////////////
/// Default constructor

TDecompBK::TDecompBK()
{
   fNIpiv = 0;
   fIpiv  = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor for (nrows x nrows) symmetric matrix

TDecompBK::TDecompBK(Int_t nrows)
{
   fNIpiv = nrows;
   fIpiv = new Int_t[fNIpiv];
   memset(fIpiv,0,fNIpiv*sizeof(Int_t));
   fU.ResizeTo(nrows,nrows);
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor for ([row_lwb..row_upb] x [row_lwb..row_upb]) symmetric matrix

TDecompBK::TDecompBK(Int_t row_lwb,Int_t row_upb)
{
   const Int_t nrows = row_upb-row_lwb+1;
   fNIpiv = nrows;
   fIpiv = new Int_t[fNIpiv];
   memset(fIpiv,0,fNIpiv*sizeof(Int_t));
   fColLwb = fRowLwb = row_lwb;
   fU.ResizeTo(nrows,nrows);
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor for symmetric matrix A

TDecompBK::TDecompBK(const TMatrixDSym &a,Double_t tol)
{
   R__ASSERT(a.IsValid());

   SetBit(kMatrixSet);
   fCondition = a.Norm1();
   fTol = a.GetTol();
   if (tol > 0)
      fTol = tol;

   fNIpiv = a.GetNcols();
   fIpiv = new Int_t[fNIpiv];
   memset(fIpiv,0,fNIpiv*sizeof(Int_t));

   const Int_t nRows = a.GetNrows();
   fColLwb = fRowLwb = a.GetRowLwb();
   fU.ResizeTo(nRows,nRows);
   memcpy(fU.GetMatrixArray(),a.GetMatrixArray(),nRows*nRows*sizeof(Double_t));
}

////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

TDecompBK::TDecompBK(const TDecompBK &another) : TDecompBase(another)
{
   fNIpiv = 0;
   fIpiv  = 0;
   *this = another;
}

////////////////////////////////////////////////////////////////////////////////
/// Matrix A is decomposed in components U and D so that A = U*D*U^T
/// If the decomposition succeeds, bit kDecomposed is set , otherwise kSingular

Bool_t TDecompBK::Decompose()
{
   if (TestBit(kDecomposed)) return kTRUE;

   if ( !TestBit(kMatrixSet) ) {
      Error("Decompose()","Matrix has not been set");
      return kFALSE;
   }

   Bool_t ok = kTRUE;

// Initialize alpha for use in choosing pivot block size.
   const Double_t alpha = (1.+TMath::Sqrt(17.))/8.;

// Factorize a as u*d*u' using the upper triangle of a .
//  k is the main loop index, decreasing from n-1 to 0 in steps of 1 or 2

   const Int_t     n  = fU.GetNcols();
         Double_t *pU = fU.GetMatrixArray();
   TMatrixDDiag diag(fU);

   Int_t imax = 0;
   Int_t k = n-1;
   while (k >= 0) {
      Int_t kstep = 1;

      // determine rows and columns to be interchanged and whether
      // a 1-by-1 or 2-by-2 pivot block will be used

      const Double_t absakk = TMath::Abs(diag(k));

      // imax is the row-index of the largest off-diagonal element in
      // column k, and colmax is its absolute value

      Double_t colmax;
      if ( k > 0 ) {
         TVectorD vcol = TMatrixDColumn_const(fU,k);
         vcol.Abs();
         imax = TMath::LocMax(k,vcol.GetMatrixArray());
         colmax = vcol[imax];
      } else {
         colmax = 0.;
      }

      Int_t kp;
      if (TMath::Max(absakk,colmax) <= fTol) {
         // the block diagonal matrix will be singular
         kp = k;
         ok = kFALSE;
      } else {
         if (absakk >= alpha*colmax) {
           // no interchange, use 1-by-1 pivot block
            kp = k;
         } else {
            // jmax is the column-index of the largest off-diagonal
            // element in row imax, and rowmax is its absolute value
            TVectorD vrow = TMatrixDRow_const(fU,imax);
            vrow.Abs();
            Int_t jmax = imax+1+TMath::LocMax(k-imax,vrow.GetMatrixArray()+imax+1);
            Double_t rowmax = vrow[jmax];
            if (imax > 0) {
               TVectorD vcol = TMatrixDColumn_const(fU,imax);
               vcol.Abs();
               jmax = TMath::LocMax(imax,vcol.GetMatrixArray());
               rowmax = TMath::Max(rowmax,vcol[jmax]);
            }

            if (absakk >= alpha*colmax*(colmax/rowmax)) {
               // No interchange, use 1-by-1 pivot block
               kp = k;
            } else if( TMath::Abs(diag(imax)) >= alpha*rowmax) {
               // Interchange rows and columns k and imax, use 1-by-1 pivot block
               kp = imax;
            } else {
               // Interchange rows and columns k-1 and imax, use 2-by-2 pivot block
               kp = imax;
               kstep = 2;
            }
         }

         const Int_t kk = k-kstep+1;
         if (kp != kk) {
            // Interchange rows and columns kk and kp in the leading submatrix a(0:k,0:k)
            Double_t *c_kk = pU+kk;
            Double_t *c_kp = pU+kp;
            for (Int_t irow = 0; irow < kp; irow++) {
               const Double_t t = *c_kk;
               *c_kk = *c_kp;
               *c_kp = t;
               c_kk += n;
               c_kp += n;
            }

            c_kk = pU+(kp+1)*n+kk;
            Double_t *r_kp = pU+kp*n+kp+1;
            for (Int_t icol = 0; icol < kk-kp-1; icol++) {
               const Double_t t = *c_kk;
               *c_kk = *r_kp;
               *r_kp = t;
               c_kk += n;
               r_kp += 1;
            }

            Double_t t = diag(kk);
            diag(kk) = diag(kp);
            diag(kp) = t;
            if (kstep == 2) {
               t = pU[(k-1)*n+k];
               pU[(k-1)*n+k] = pU[kp*n+k];
               pU[kp*n+k]    = t;
            }
         }

         // Update the leading submatrix

         if (kstep == 1 && k > 0) {
            // 1-by-1 pivot block d(k): column k now holds w(k) = u(k)*d(k)
            // where u(k) is the k-th column of u

            // perform a rank-1 update of a(0:k-1,0:k-1) as
            // a := a - u(k)*d(k)*u(k)' = a - w(k)*1/d(k)*w(k)'

            const Double_t r1 = 1./diag(k);
            TMatrixDSub sub1(fU,0,k-1,0,k-1);
            sub1.Rank1Update(TMatrixDColumn_const(fU,k),-r1);

            // store u(k) in column k
            TMatrixDSub sub2(fU,0,k-1,k,k);
            sub2 *= r1;
         } else {
            // 2-by-2 pivot block d(k): columns k and k-1 now hold
            // ( w(k-1) w(k) ) = ( u(k-1) u(k) )*d(k)
            // where u(k) and u(k-1) are the k-th and (k-1)-th columns of u

            // perform a rank-2 update of a(0:k-2,0:k-2) as
            // a := a - ( u(k-1) u(k) )*d(k)*( u(k-1) u(k) )'
            //    = a - ( w(k-1) w(k) )*inv(d(k))*( w(k-1) w(k) )'

            if ( k > 1 ) {
                     Double_t *pU_k1 = pU+(k-1)*n;
                     Double_t d12    = pU_k1[k];
               const Double_t d22    = pU_k1[k-1]/d12;
               const Double_t d11    = diag(k)/d12;
               const Double_t t      = 1./(d11*d22-1.);
               d12 = t/d12;

               for (Int_t j = k-2; j >= 0; j--) {
                  Double_t *pU_j = pU+j*n;
                  const Double_t wkm1 = d12*(d11*pU_j[k-1]-pU_j[k]);
                  const Double_t wk   = d12*(d22*pU_j[k]-pU_j[k-1]);
                  for (Int_t i = j; i >= 0; i--) {
                     Double_t *pU_i = pU+i*n;
                     pU_i[j] -= (pU_i[k]*wk+pU_i[k-1]*wkm1);
                  }
                  pU_j[k]   = wk;
                  pU_j[k-1] = wkm1;
               }
            }
         }

         // Store details of the interchanges in fIpiv
         if (kstep == 1) {
            fIpiv[k] = (kp+1);
         } else {
            fIpiv[k]   = -(kp+1);
            fIpiv[k-1] = -(kp+1);
         }
      }

      k -= kstep;
   }

   if (!ok) SetBit(kSingular);
   else     SetBit(kDecomposed);

   fU.Shift(fRowLwb,fRowLwb);

   return ok;
}

////////////////////////////////////////////////////////////////////////////////
/// Set the matrix to be decomposed, decomposition status is reset.

void TDecompBK::SetMatrix(const TMatrixDSym &a)
{
   R__ASSERT(a.IsValid());

   ResetStatus();

   SetBit(kMatrixSet);
   fCondition = a.Norm1();

   if (fNIpiv != a.GetNcols()) {
      fNIpiv = a.GetNcols();
      delete [] fIpiv;
      fIpiv = new Int_t[fNIpiv];
      memset(fIpiv,0,fNIpiv*sizeof(Int_t));
   }

   const Int_t nRows = a.GetNrows();
   fColLwb = fRowLwb = a.GetRowLwb();
   fU.ResizeTo(nRows,nRows);
   memcpy(fU.GetMatrixArray(),a.GetMatrixArray(),nRows*nRows*sizeof(Double_t));
}

////////////////////////////////////////////////////////////////////////////////
/// Solve Ax=b assuming the BK form of A is stored in fU . Solution returned in b.

Bool_t TDecompBK::Solve(TVectorD &b)
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

   if (fU.GetNrows() != b.GetNrows() || fU.GetRowLwb() != b.GetLwb()) {
      Error("Solve(TVectorD &","vector and matrix incompatible");
      return kFALSE;
   }

   const Int_t n = fU.GetNrows();

   TMatrixDDiag_const diag(fU);
   const Double_t *pU = fU.GetMatrixArray();
         Double_t *pb = b.GetMatrixArray();

   // solve a*x = b, where a = u*d*u'. First solve u*d*x = b, overwriting b with x.
   // k is the main loop index, decreasing from n-1 to 0 in steps of 1 or 2,
   // depending on the size of the diagonal blocks.

   Int_t k = n-1;
   while (k >= 0) {

      if (fIpiv[k] > 0) {

         //  1 x 1 diagonal block
         //  interchange rows k and ipiv(k).
         const Int_t kp = fIpiv[k]-1;
         if (kp != k) {
            const Double_t tmp = pb[k];
            pb[k]  = pb[kp];
            pb[kp] = tmp;
         }

         // multiply by inv(u(k)), where u(k) is the transformation
         // stored in column k of a.
         for (Int_t i = 0; i < k; i++)
            pb[i] -= pU[i*n+k]*pb[k];

         // multiply by the inverse of the diagonal block.
         pb[k] /= diag(k);
         k--;
      } else {

         // 2 x 2 diagonal block
         // interchange rows k-1 and -ipiv(k).
         const Int_t kp = -fIpiv[k]-1;
         if (kp != k-1) {
            const Double_t tmp = pb[k-1];
            pb[k-1]  = pb[kp];
            pb[kp]   = tmp;
         }

         // multiply by inv(u(k)), where u(k) is the transformation
         // stored in columns k-1 and k of a.
         Int_t i;
         for (i = 0; i < k-1; i++)
            pb[i] -= pU[i*n+k]*pb[k];
         for (i = 0; i < k-1; i++)
            pb[i] -= pU[i*n+k-1]*pb[k-1];

         // multiply by the inverse of the diagonal block.
         const Double_t *pU_k1 = pU+(k-1)*n;
         const Double_t ukm1k  = pU_k1[k];
         const Double_t ukm1   = pU_k1[k-1]/ukm1k;
         const Double_t uk     = diag(k)/ukm1k;
         const Double_t denom  = ukm1*uk-1.;
         const Double_t bkm1   = pb[k-1]/ukm1k;
         const Double_t bk     = pb[k]/ukm1k;
         pb[k-1] = (uk*bkm1-bk)/denom;
         pb[k]   = (ukm1*bk-bkm1)/denom;
         k -= 2;
      }
   }

   // Next solve u'*x = b, overwriting b with x.
   //
   //  k is the main loop index, increasing from 0 to n-1 in steps of
   //  1 or 2, depending on the size of the diagonal blocks.

   k = 0;
   while (k < n) {

      if (fIpiv[k] > 0) {
         // 1 x 1 diagonal block
         //  multiply by inv(u'(k)), where u(k) is the transformation
         //  stored in column k of a.
         for (Int_t i = 0; i < k; i++)
            pb[k] -= pU[i*n+k]*pb[i];

         // interchange elements k and ipiv(k).
         const Int_t kp = fIpiv[k]-1;
         if (kp != k) {
            const Double_t tmp = pb[k];
            pb[k]  = pb[kp];
            pb[kp] = tmp;
         }
         k++;
      } else {
         // 2 x 2 diagonal block
         // multiply by inv(u'(k+1)), where u(k+1) is the transformation
         // stored in columns k and k+1 of a.
         Int_t i ;
         for (i = 0; i < k; i++)
            pb[k] -= pU[i*n+k]*pb[i];
         for (i = 0; i < k; i++)
            pb[k+1] -= pU[i*n+k+1]*pb[i];

         // interchange elements k and -ipiv(k).
         const Int_t kp = -fIpiv[k]-1;
         if (kp != k) {
            const Double_t tmp = pb[k];
            pb[k]  = pb[kp];
            pb[kp] = tmp;
         }
         k += 2;
      }
   }

   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Solve Ax=b assuming the BK form of A is stored in fU . Solution returned in b.

Bool_t TDecompBK::Solve(TMatrixDColumn &cb)
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

   if (fU.GetNrows() != b->GetNrows() || fU.GetRowLwb() != b->GetRowLwb()) {
      Error("Solve(TMatrixDColumn &","vector and matrix incompatible");
      return kFALSE;
   }

   const Int_t n = fU.GetNrows();

   TMatrixDDiag_const diag(fU);
   const Double_t *pU  = fU.GetMatrixArray();
         Double_t *pcb = cb.GetPtr();
   const Int_t     inc = cb.GetInc();


  // solve a*x = b, where a = u*d*u'. First solve u*d*x = b, overwriting b with x.
  // k is the main loop index, decreasing from n-1 to 0 in steps of 1 or 2,
  // depending on the size of the diagonal blocks.

   Int_t k = n-1;
   while (k >= 0) {

      if (fIpiv[k] > 0) {

         //  1 x 1 diagonal block
         //  interchange rows k and ipiv(k).
         const Int_t kp = fIpiv[k]-1;
         if (kp != k) {
            const Double_t tmp = pcb[k*inc];
            pcb[k*inc]  = pcb[kp*inc];
            pcb[kp*inc] = tmp;
         }

         // multiply by inv(u(k)), where u(k) is the transformation
         // stored in column k of a.
         for (Int_t i = 0; i < k; i++)
            pcb[i*inc] -= pU[i*n+k]*pcb[k*inc];

         // multiply by the inverse of the diagonal block.
         pcb[k*inc] /= diag(k);
         k--;
      } else {

         // 2 x 2 diagonal block
         // interchange rows k-1 and -ipiv(k).
         const Int_t kp = -fIpiv[k]-1;
         if (kp != k-1) {
            const Double_t tmp = pcb[(k-1)*inc];
            pcb[(k-1)*inc] = pcb[kp*inc];
            pcb[kp*inc]    = tmp;
         }

         // multiply by inv(u(k)), where u(k) is the transformation
         // stored in columns k-1 and k of a.
         Int_t i;
         for (i = 0; i < k-1; i++)
            pcb[i*inc] -= pU[i*n+k]*pcb[k*inc];
         for (i = 0; i < k-1; i++)
            pcb[i*inc] -= pU[i*n+k-1]*pcb[(k-1)*inc];

         // multiply by the inverse of the diagonal block.
         const Double_t *pU_k1 = pU+(k-1)*n;
         const Double_t ukm1k  = pU_k1[k];
         const Double_t ukm1   = pU_k1[k-1]/ukm1k;
         const Double_t uk     = diag(k)/ukm1k;
         const Double_t denom  = ukm1*uk-1.;
         const Double_t bkm1   = pcb[(k-1)*inc]/ukm1k;
         const Double_t bk     = pcb[k*inc]/ukm1k;
         pcb[(k-1)*inc] = (uk*bkm1-bk)/denom;
         pcb[k*inc]     = (ukm1*bk-bkm1)/denom;
         k -= 2;
      }
   }

   // Next solve u'*x = b, overwriting b with x.
   //
   //  k is the main loop index, increasing from 0 to n-1 in steps of
   //  1 or 2, depending on the size of the diagonal blocks.

   k = 0;
   while (k < n) {

      if (fIpiv[k] > 0) {
         // 1 x 1 diagonal block
         //  multiply by inv(u'(k)), where u(k) is the transformation
         //  stored in column k of a.
         for (Int_t i = 0; i < k; i++)
            pcb[k*inc] -= pU[i*n+k]*pcb[i*inc];

         // interchange elements k and ipiv(k).
         const Int_t kp = fIpiv[k]-1;
         if (kp != k) {
            const Double_t tmp = pcb[k*inc];
            pcb[k*inc]  = pcb[kp*inc];
            pcb[kp*inc] = tmp;
         }
         k++;
      } else {
         // 2 x 2 diagonal block
         // multiply by inv(u'(k+1)), where u(k+1) is the transformation
         // stored in columns k and k+1 of a.
         Int_t i;
         for (i = 0; i < k; i++)
            pcb[k*inc] -= pU[i*n+k]*pcb[i*inc];
         for (i = 0; i < k; i++)
            pcb[(k+1)*inc] -= pU[i*n+k+1]*pcb[i*inc];

         // interchange elements k and -ipiv(k).
         const Int_t kp = -fIpiv[k]-1;
         if (kp != k) {
            const Double_t tmp = pcb[k*inc];
            pcb[k*inc]  = pcb[kp*inc];
            pcb[kp*inc] = tmp;
         }
         k += 2;
      }
   }

   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// For a symmetric matrix A(m,m), its inverse A_inv(m,m) is returned .

Bool_t TDecompBK::Invert(TMatrixDSym &inv)
{
   if (inv.GetNrows() != GetNrows() || inv.GetRowLwb() != GetRowLwb()) {
      Error("Invert(TMatrixDSym &","Input matrix has wrong shape");
      return kFALSE;
   }

   inv.UnitMatrix();

   const Int_t colLwb = inv.GetColLwb();
   const Int_t colUpb = inv.GetColUpb();
   Bool_t status = kTRUE;
   for (Int_t icol = colLwb; icol <= colUpb && status; icol++) {
      TMatrixDColumn b(inv,icol);
      status &= Solve(b);
   }

   return status;
}

////////////////////////////////////////////////////////////////////////////////
/// For a symmetric matrix A(m,m), its inverse A_inv(m,m) is returned .

TMatrixDSym TDecompBK::Invert(Bool_t &status)
{
   const Int_t rowLwb = GetRowLwb();
   const Int_t rowUpb = rowLwb+GetNrows()-1;

   TMatrixDSym inv(rowLwb,rowUpb);
   inv.UnitMatrix();
   status = Invert(inv);

   return inv;
}

////////////////////////////////////////////////////////////////////////////////
/// Print the class members

void TDecompBK::Print(Option_t *opt) const
{
   TDecompBase::Print(opt);
   printf("fIpiv:\n");
   for (Int_t i = 0; i < fNIpiv; i++)
      printf("[%d] = %d\n",i,fIpiv[i]);
   fU.Print("fU");
}

////////////////////////////////////////////////////////////////////////////////
/// Assignment operator

TDecompBK &TDecompBK::operator=(const TDecompBK &source)
{
   if (this != &source) {
      TDecompBase::operator=(source);
      fU.ResizeTo(source.fU);
      fU   = source.fU;
      if (fNIpiv != source.fNIpiv) {
         if (fIpiv)
            delete [] fIpiv;
         fNIpiv = source.fNIpiv;
         fIpiv = new Int_t[fNIpiv];
      }
      if (fIpiv) memcpy(fIpiv,source.fIpiv,fNIpiv*sizeof(Int_t));
   }
   return *this;
}
