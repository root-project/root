// @(#)root/matrix:$Id$
// Authors: Fons Rademakers, Eddy Offermann  Dec 2003

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMatrixDSymEigen                                                     //
//                                                                      //
// Eigenvalues and eigenvectors of a real symmetric matrix.             //
//                                                                      //
// If A is symmetric, then A = V*D*V' where the eigenvalue matrix D is  //
// diagonal and the eigenvector matrix V is orthogonal. That is, the    //
// diagonal values of D are the eigenvalues, and V*V' = I, where I is   //
// the identity matrix.  The columns of V represent the eigenvectors in //
// the sense that A*V = V*D.                                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TMatrixDSymEigen.h"
#include "TMath.h"

ClassImp(TMatrixDSymEigen)

//______________________________________________________________________________
TMatrixDSymEigen::TMatrixDSymEigen(const TMatrixDSym &a)
{
// Constructor for eigen-problem of symmetric matrix A .

   R__ASSERT(a.IsValid());

   const Int_t nRows  = a.GetNrows();
   const Int_t rowLwb = a.GetRowLwb();

   fEigenValues.ResizeTo(rowLwb,rowLwb+nRows-1);
   fEigenVectors.ResizeTo(a);

   fEigenVectors = a;

   TVectorD offDiag;
   Double_t work[kWorkMax];
   if (nRows > kWorkMax) offDiag.ResizeTo(nRows);
   else                  offDiag.Use(nRows,work);

   // Tridiagonalize matrix
   MakeTridiagonal(fEigenVectors,fEigenValues,offDiag);

   // Make eigenvectors and -values
   MakeEigenVectors(fEigenVectors,fEigenValues,offDiag);
}

//______________________________________________________________________________
TMatrixDSymEigen::TMatrixDSymEigen(const TMatrixDSymEigen &another)
{
// Copy constructor

   *this = another;
}

//______________________________________________________________________________
void TMatrixDSymEigen::MakeTridiagonal(TMatrixD &v,TVectorD &d,TVectorD &e)
{
// This is derived from the Algol procedures tred2 by Bowdler, Martin, Reinsch, and
// Wilkinson, Handbook for Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
// Fortran subroutine in EISPACK.

   Double_t *pV = v.GetMatrixArray();
   Double_t *pD = d.GetMatrixArray();
   Double_t *pE = e.GetMatrixArray();

   const Int_t n = v.GetNrows();

   Int_t i,j,k;
   Int_t off_n1 = (n-1)*n;
   for (j = 0; j < n; j++)
      pD[j] = pV[off_n1+j];

   // Householder reduction to tridiagonal form.

   for (i = n-1; i > 0; i--) {
      const Int_t off_i1 = (i-1)*n;
      const Int_t off_i  = i*n;

      // Scale to avoid under/overflow.

      Double_t scale = 0.0;
      Double_t h = 0.0;
      for (k = 0; k < i; k++)
         scale = scale+TMath::Abs(pD[k]);
      if (scale == 0.0) {
         pE[i] = pD[i-1];
         for (j = 0; j < i; j++) {
            const Int_t off_j = j*n;
            pD[j] = pV[off_i1+j];
            pV[off_i+j] = 0.0;
            pV[off_j+i] = 0.0;
         }
      } else {

         // Generate Householder vector.

         for (k = 0; k < i; k++) {
            pD[k] /= scale;
            h += pD[k]*pD[k];
         }
         Double_t f = pD[i-1];
         Double_t g = TMath::Sqrt(h);
         if (f > 0)
            g = -g;
         pE[i]   = scale*g;
         h       = h-f*g;
         pD[i-1] = f-g;
         for (j = 0; j < i; j++)
            pE[j] = 0.0;

         // Apply similarity transformation to remaining columns.

         for (j = 0; j < i; j++) {
            const Int_t off_j = j*n;
            f = pD[j];
            pV[off_j+i] = f;
            g = pE[j]+pV[off_j+j]*f;
            for (k = j+1; k <= i-1; k++) {
               const Int_t off_k = k*n;
               g += pV[off_k+j]*pD[k];
               pE[k] += pV[off_k+j]*f;
            }
            pE[j] = g;
         }
         f = 0.0;
         for (j = 0; j < i; j++) {
            pE[j] /= h;
            f += pE[j]*pD[j];
         }
         Double_t hh = f/(h+h);
         for (j = 0; j < i; j++)
            pE[j] -= hh*pD[j];
         for (j = 0; j < i; j++) {
            f = pD[j];
            g = pE[j];
            for (k = j; k <= i-1; k++) {
               const Int_t off_k = k*n;
               pV[off_k+j] -= (f*pE[k]+g*pD[k]);
            }
            pD[j] = pV[off_i1+j];
            pV[off_i+j] = 0.0;
         }
      }
      pD[i] = h;
   }

   // Accumulate transformations.

   for (i = 0; i < n-1; i++) {
      const Int_t off_i  = i*n;
      pV[off_n1+i] = pV[off_i+i];
      pV[off_i+i] = 1.0;
      Double_t h = pD[i+1];
      if (h != 0.0) {
         for (k = 0; k <= i; k++) {
            const Int_t off_k = k*n;
            pD[k] = pV[off_k+i+1]/h;
         }
         for (j = 0; j <= i; j++) {
            Double_t g = 0.0;
            for (k = 0; k <= i; k++) {
               const Int_t off_k = k*n;
               g += pV[off_k+i+1]*pV[off_k+j];
            }
            for (k = 0; k <= i; k++) {
               const Int_t off_k = k*n;
               pV[off_k+j] -= g*pD[k];
            }
         }
      }
      for (k = 0; k <= i; k++) {
         const Int_t off_k = k*n;
         pV[off_k+i+1] = 0.0;
      }
   }
   for (j = 0; j < n; j++) {
      pD[j] = pV[off_n1+j];
      pV[off_n1+j] = 0.0;
   }
   pV[off_n1+n-1] = 1.0;
   pE[0] = 0.0;
}

//______________________________________________________________________________
void TMatrixDSymEigen::MakeEigenVectors(TMatrixD &v,TVectorD &d,TVectorD &e)
{
// Symmetric tridiagonal QL algorithm.
// This is derived from the Algol procedures tql2, by Bowdler, Martin, Reinsch, and
// Wilkinson, Handbook for Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
// Fortran subroutine in EISPACK.

   Double_t *pV = v.GetMatrixArray();
   Double_t *pD = d.GetMatrixArray();
   Double_t *pE = e.GetMatrixArray();

   const Int_t n = v.GetNrows();

   Int_t i,j,k,l;
   for (i = 1; i < n; i++)
      pE[i-1] = pE[i];
   pE[n-1] = 0.0;

   Double_t f = 0.0;
   Double_t tst1 = 0.0;
   Double_t eps = TMath::Power(2.0,-52.0);
   for (l = 0; l < n; l++) {

      // Find small subdiagonal element

      tst1 = TMath::Max(tst1,TMath::Abs(pD[l])+TMath::Abs(pE[l]));
      Int_t m = l;

      // Original while-loop from Java code
      while (m < n) {
         if (TMath::Abs(pE[m]) <= eps*tst1) {
            break;
         }
         m++;
      }

      // If m == l, pD[l] is an eigenvalue,
      // otherwise, iterate.

      if (m > l) {
         Int_t iter = 0;
         do {
            if (iter++ == 30) {  // (check iteration count here.)
               Error("MakeEigenVectors","too many iterations");
               break;
            }

            // Compute implicit shift

            Double_t g = pD[l];
            Double_t p = (pD[l+1]-g)/(2.0*pE[l]);
            Double_t r = TMath::Hypot(p,1.0);
            if (p < 0)
               r = -r;
            pD[l] = pE[l]/(p+r);
            pD[l+1] = pE[l]*(p+r);
            Double_t dl1 = pD[l+1];
            Double_t h = g-pD[l];
            for (i = l+2; i < n; i++)
               pD[i] -= h;
            f = f+h;

            // Implicit QL transformation.

            p = pD[m];
            Double_t c = 1.0;
            Double_t c2 = c;
            Double_t c3 = c;
            Double_t el1 = pE[l+1];
            Double_t s = 0.0;
            Double_t s2 = 0.0;
            for (i = m-1; i >= l; i--) {
               c3 = c2;
               c2 = c;
               s2 = s;
               g = c*pE[i];
               h = c*p;
               r = TMath::Hypot(p,pE[i]);
               pE[i+1] = s*r;
               s = pE[i]/r;
               c = p/r;
               p = c*pD[i]-s*g;
               pD[i+1] = h+s*(c*g+s*pD[i]);

               // Accumulate transformation.
 
               for (k = 0; k < n; k++) {
                  const Int_t off_k = k*n;
                  h = pV[off_k+i+1];
                  pV[off_k+i+1] = s*pV[off_k+i]+c*h;
                  pV[off_k+i]   = c*pV[off_k+i]-s*h;
               }
            }
            p = -s*s2*c3*el1*pE[l]/dl1;
            pE[l] = s*p;
            pD[l] = c*p;

            // Check for convergence.

         } while (TMath::Abs(pE[l]) > eps*tst1);
      }
      pD[l] = pD[l]+f;
      pE[l] = 0.0;
   }

   // Sort eigenvalues and corresponding vectors.

   for (i = 0; i < n-1; i++) {
      k = i;
      Double_t p = pD[i];
      for (j = i+1; j < n; j++) {
         if (pD[j] > p) {
            k = j;
            p = pD[j];
         }
      }
      if (k != i) {
         pD[k] = pD[i];
         pD[i] = p;
         for (j = 0; j < n; j++) {
            const Int_t off_j = j*n;
            p = pV[off_j+i];
            pV[off_j+i] = pV[off_j+k];
            pV[off_j+k] = p;
         }
      }
   }
}

//______________________________________________________________________________
TMatrixDSymEigen &TMatrixDSymEigen::operator=(const TMatrixDSymEigen &source)
{
// Assignment operator

   if (this != &source) {
      fEigenVectors.ResizeTo(source.fEigenVectors);
      fEigenValues.ResizeTo(source.fEigenValues);
   }
   return *this;
}
