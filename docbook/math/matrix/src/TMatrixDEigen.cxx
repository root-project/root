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
// TMatrixDEigen                                                        //
//                                                                      //
// Eigenvalues and eigenvectors of a real matrix.                       //
//                                                                      //
// If A is not symmetric, then the eigenvalue matrix D is block         //
// diagonal with the real eigenvalues in 1-by-1 blocks and any complex  //
// eigenvalues, a + i*b, in 2-by-2 blocks, [a, b; -b, a].  That is, if  //
// the complex eigenvalues look like                                    //
//                                                                      //
//     u + iv     .        .          .      .    .                     //
//       .      u - iv     .          .      .    .                     //
//       .        .      a + ib       .      .    .                     //
//       .        .        .        a - ib   .    .                     //
//       .        .        .          .      x    .                     //
//       .        .        .          .      .    y                     //
//                                                                      //
// then D looks like                                                    //
//                                                                      //
//       u        v        .          .      .    .                     //
//      -v        u        .          .      .    .                     //
//       .        .        a          b      .    .                     //
//       .        .       -b          a      .    .                     //
//       .        .        .          .      x    .                     //
//       .        .        .          .      .    y                     //
//                                                                      //
// This keeps V a real matrix in both symmetric and non-symmetric       //
// cases, and A*V = V*D.                                                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TMatrixDEigen.h"
#include "TMath.h"

ClassImp(TMatrixDEigen)

//______________________________________________________________________________
TMatrixDEigen::TMatrixDEigen(const TMatrixD &a)
{
// Constructor for eigen-problem of matrix A .

   R__ASSERT(a.IsValid());
 
   const Int_t nRows  = a.GetNrows();
   const Int_t nCols  = a.GetNcols();
   const Int_t rowLwb = a.GetRowLwb();
   const Int_t colLwb = a.GetColLwb();

   if (nRows != nCols || rowLwb != colLwb)
   {
      Error("TMatrixDEigen(TMatrixD &)","matrix should be square");
      return;
   }

   const Int_t rowUpb = rowLwb+nRows-1;
   fEigenVectors.ResizeTo(rowLwb,rowUpb,rowLwb,rowUpb);
   fEigenValuesRe.ResizeTo(rowLwb,rowUpb);
   fEigenValuesIm.ResizeTo(rowLwb,rowUpb);

   TVectorD ortho;
   Double_t work[kWorkMax];
   if (nRows > kWorkMax) ortho.ResizeTo(nRows);
   else                  ortho.Use(nRows,work);

   TMatrixD mH = a;

   // Reduce to Hessenberg form.
   MakeHessenBerg(fEigenVectors,ortho,mH);

   // Reduce Hessenberg to real Schur form.
   MakeSchurr(fEigenVectors,fEigenValuesRe,fEigenValuesIm,mH);

   // Sort eigenvalues and corresponding vectors in descending order of Re^2+Im^2
   // of the complex eigenvalues .
   Sort(fEigenVectors,fEigenValuesRe,fEigenValuesIm);
}

//______________________________________________________________________________
TMatrixDEigen::TMatrixDEigen(const TMatrixDEigen &another)
{
// Copy constructor

   *this = another;
}

//______________________________________________________________________________
void TMatrixDEigen::MakeHessenBerg(TMatrixD &v,TVectorD &ortho,TMatrixD &H)
{
// Nonsymmetric reduction to Hessenberg form.
// This is derived from the Algol procedures orthes and ortran, by Martin and Wilkinson,
// Handbook for Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
// Fortran subroutines in EISPACK.

   Double_t *pV = v.GetMatrixArray();
   Double_t *pO = ortho.GetMatrixArray();
   Double_t *pH = H.GetMatrixArray();

   const Int_t n = v.GetNrows();

   const Int_t low  = 0;
   const Int_t high = n-1;

   Int_t i,j,m;
   for (m = low+1; m <= high-1; m++) {
      const Int_t off_m = m*n;

      // Scale column.

      Double_t scale = 0.0;
      for (i = m; i <= high; i++) {
         const Int_t off_i = i*n;
         scale = scale + TMath::Abs(pH[off_i+m-1]);
      }
      if (scale != 0.0) {

         // Compute Householder transformation.

         Double_t h = 0.0;
         for (i = high; i >= m; i--) {
            const Int_t off_i = i*n;
            pO[i] = pH[off_i+m-1]/scale;
            h += pO[i]*pO[i];
         }
         Double_t g = TMath::Sqrt(h);
         if (pO[m] > 0)
            g = -g;
         h = h-pO[m]*g;
         pO[m] = pO[m]-g;

         // Apply Householder similarity transformation
         // H = (I-u*u'/h)*H*(I-u*u')/h)

         for (j = m; j < n; j++) {
            Double_t f = 0.0;
            for (i = high; i >= m; i--) {
               const Int_t off_i = i*n;
               f += pO[i]*pH[off_i+j];
            }
            f = f/h;
            for (i = m; i <= high; i++) {
               const Int_t off_i = i*n;
               pH[off_i+j] -= f*pO[i];
            }
         }

         for (i = 0; i <= high; i++) {
            const Int_t off_i = i*n;
            Double_t f = 0.0;
            for (j = high; j >= m; j--)
               f += pO[j]*pH[off_i+j];
            f = f/h;
            for (j = m; j <= high; j++)
               pH[off_i+j] -= f*pO[j];
         }
         pO[m] = scale*pO[m];
         pH[off_m+m-1] = scale*g;
      }
   }

   // Accumulate transformations (Algol's ortran).

   for (i = 0; i < n; i++) {
      const Int_t off_i = i*n;
      for (j = 0; j < n; j++)
         pV[off_i+j] = (i == j ? 1.0 : 0.0);
   }

   for (m = high-1; m >= low+1; m--) {
      const Int_t off_m = m*n;
      if (pH[off_m+m-1] != 0.0) {
         for (i = m+1; i <= high; i++) {
            const Int_t off_i = i*n;
            pO[i] = pH[off_i+m-1];
         }
         for (j = m; j <= high; j++) {
            Double_t g = 0.0;
            for (i = m; i <= high; i++) {
               const Int_t off_i = i*n;
               g += pO[i]*pV[off_i+j];
            }
            // Double division avoids possible underflow
            g = (g/pO[m])/pH[off_m+m-1];
            for (i = m; i <= high; i++) {
               const Int_t off_i = i*n;
               pV[off_i+j] += g*pO[i];
            }
         }
      }
   }
}

//______________________________________________________________________________
static Double_t gCdivr, gCdivi;
static void cdiv(Double_t xr,Double_t xi,Double_t yr,Double_t yi) {
// Complex scalar division.
   Double_t r,d;
   if (TMath::Abs(yr) > TMath::Abs(yi)) {
      r = yi/yr;
      d = yr+r*yi;
      gCdivr = (xr+r*xi)/d;
      gCdivi = (xi-r*xr)/d;
   } else {
      r = yr/yi;
      d = yi+r*yr;
      gCdivr = (r*xr+xi)/d;
      gCdivi = (r*xi-xr)/d;
   }
}

//______________________________________________________________________________
void TMatrixDEigen::MakeSchurr(TMatrixD &v,TVectorD &d,TVectorD &e,TMatrixD &H)
{
// Nonsymmetric reduction from Hessenberg to real Schur form.
// This is derived from the Algol procedure hqr2, by Martin and Wilkinson,
// Handbook for Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
// Fortran subroutine in EISPACK.

   // Initialize

   const Int_t nn = v.GetNrows();
         Int_t n = nn-1;
   const Int_t low = 0;
   const Int_t high = nn-1;
   const Double_t eps = TMath::Power(2.0,-52.0);
   Double_t exshift = 0.0;
   Double_t p=0,q=0,r=0,s=0,z=0,t,w,x,y;

   Double_t *pV = v.GetMatrixArray();
   Double_t *pD = d.GetMatrixArray();
   Double_t *pE = e.GetMatrixArray();
   Double_t *pH = H.GetMatrixArray();

   // Store roots isolated by balanc and compute matrix norm

   Double_t norm = 0.0;
   Int_t i,j,k;
   for (i = 0; i < nn; i++) {
      const Int_t off_i = i*nn;
      if ((i < low) || (i > high)) {
         pD[i] = pH[off_i+i];
         pE[i] = 0.0;
      }
      for (j = TMath::Max(i-1,0); j < nn; j++)
         norm += TMath::Abs(pH[off_i+j]);
   }

   // Outer loop over eigenvalue index

   Int_t iter = 0;
   while (n >= low) {
      const Int_t off_n  = n*nn;
      const Int_t off_n1 = (n-1)*nn;

      // Look for single small sub-diagonal element

      Int_t l = n;
      while (l > low) {
         const Int_t off_l1 = (l-1)*nn;
         const Int_t off_l  = l*nn;
         s = TMath::Abs(pH[off_l1+l-1])+TMath::Abs(pH[off_l+l]);
         if (s == 0.0)
            s = norm;
         if (TMath::Abs(pH[off_l+l-1]) < eps*s)
            break;
         l--;
      }

      // Check for convergence
      // One root found

      if (l == n) {
         pH[off_n+n] = pH[off_n+n]+exshift;
         pD[n] = pH[off_n+n];
         pE[n] = 0.0;
         n--;
         iter = 0;

         // Two roots found

      } else if (l == n-1) {
         w = pH[off_n+n-1]*pH[off_n1+n];
         p = (pH[off_n1+n-1]-pH[off_n+n])/2.0;
         q = p*p+w;
         z = TMath::Sqrt(TMath::Abs(q));
         pH[off_n+n] = pH[off_n+n]+exshift;
         pH[off_n1+n-1] = pH[off_n1+n-1]+exshift;
         x = pH[off_n+n];

         // Double_t pair

         if (q >= 0) {
            if (p >= 0)
               z = p+z;
            else
               z = p-z;
            pD[n-1] = x+z;
            pD[n] = pD[n-1];
            if (z != 0.0)
               pD[n] = x-w/z;
            pE[n-1] = 0.0;
            pE[n] = 0.0;
            x = pH[off_n+n-1];
            s = TMath::Abs(x)+TMath::Abs(z);
            p = x/s;
            q = z/s;
            r = TMath::Sqrt((p*p)+(q*q));
            p = p/r;
            q = q/r;

            // Row modification

            for (j = n-1; j < nn; j++) {
               z = pH[off_n1+j];
               pH[off_n1+j] = q*z+p*pH[off_n+j];
               pH[off_n+j]  = q*pH[off_n+j]-p*z;
            }

            // Column modification

            for (i = 0; i <= n; i++) {
               const Int_t off_i = i*nn;
               z = pH[off_i+n-1];
               pH[off_i+n-1] = q*z+p*pH[off_i+n];
               pH[off_i+n]  = q*pH[off_i+n]-p*z;
            }

            // Accumulate transformations

            for (i = low; i <= high; i++) {
               const Int_t off_i = i*nn;
               z = pV[off_i+n-1];
               pV[off_i+n-1] = q*z+p*pV[off_i+n];
               pV[off_i+n]   = q*pV[off_i+n]-p*z;
            }

            // Complex pair

         } else {
            pD[n-1] = x+p;
            pD[n] = x+p;
            pE[n-1] = z;
            pE[n] = -z;
         }
         n = n-2;
         iter = 0;

         // No convergence yet

      } else {

         // Form shift

         x = pH[off_n+n];
         y = 0.0;
         w = 0.0;
         if (l < n) {
            y = pH[off_n1+n-1];
            w = pH[off_n+n-1]*pH[off_n1+n];
         }

         // Wilkinson's original ad hoc shift

         if (iter == 10) {
            exshift += x;
            for (i = low; i <= n; i++) {
               const Int_t off_i = i*nn;
               pH[off_i+i] -= x;
            }
            s = TMath::Abs(pH[off_n+n-1])+TMath::Abs(pH[off_n1+n-2]);
            x = y = 0.75*s;
            w = -0.4375*s*s;
         }

         // MATLAB's new ad hoc shift

         if (iter == 30) {
            s = (y-x)/2.0;
            s = s*s+w;
            if (s > 0) {
               s = TMath::Sqrt(s);
               if (y<x)
                  s = -s;
               s = x-w/((y-x)/2.0+s);
               for (i = low; i <= n; i++) {
                  const Int_t off_i = i*nn;
                  pH[off_i+i] -= s;
               }
               exshift += s;
               x = y = w = 0.964;
            }
         }

         if (iter++ == 50) {  // (check iteration count here.)
            Error("MakeSchurr","too many iterations");
            break;
         }

         // Look for two consecutive small sub-diagonal elements

         Int_t m = n-2;
         while (m >= l) {
            const Int_t off_m   = m*nn;
            const Int_t off_m_1 = (m-1)*nn;
            const Int_t off_m1  = (m+1)*nn;
            const Int_t off_m2  = (m+2)*nn;
            z = pH[off_m+m];
            r = x-z;
            s = y-z;
            p = (r*s-w)/pH[off_m1+m]+pH[off_m+m+1];
            q = pH[off_m1+m+1]-z-r-s;
            r = pH[off_m2+m+1];
            s = TMath::Abs(p)+TMath::Abs(q)+TMath::Abs(r);
            p = p /s;
            q = q/s;
            r = r/s;
            if (m == l)
               break;
            if (TMath::Abs(pH[off_m+m-1])*(TMath::Abs(q)+TMath::Abs(r)) <
               eps*(TMath::Abs(p)*(TMath::Abs(pH[off_m_1+m-1])+TMath::Abs(z)+
               TMath::Abs(pH[off_m1+m+1]))))
               break;
               m--;
            }

            for (i = m+2; i <= n; i++) {
               const Int_t off_i = i*nn;
               pH[off_i+i-2] = 0.0;
               if (i > m+2)
                  pH[off_i+i-3] = 0.0;
            }

            // Double QR step involving rows l:n and columns m:n

            for (k = m; k <= n-1; k++) {
               const Int_t off_k  = k*nn;
               const Int_t off_k1 = (k+1)*nn;
               const Int_t off_k2 = (k+2)*nn;
               const Int_t notlast = (k != n-1);
               if (k != m) {
                  p = pH[off_k+k-1];
                  q = pH[off_k1+k-1];
                  r = (notlast ? pH[off_k2+k-1] : 0.0);
                  x = TMath::Abs(p)+TMath::Abs(q)+TMath::Abs(r);
                  if (x != 0.0) {
                     p = p/x;
                     q = q/x;
                     r = r/x;
                  }
               }
               if (x == 0.0)
                  break;
               s = TMath::Sqrt(p*p+q*q+r*r);
               if (p < 0) {
                  s = -s;
               }
               if (s != 0) {
                  if (k != m)
                     pH[off_k+k-1] = -s*x;
                  else if (l != m)
                     pH[off_k+k-1] = -pH[off_k+k-1];
                  p = p+s;
                  x = p/s;
                  y = q/s;
                  z = r/s;
                  q = q/p;
                  r = r/p;

                  // Row modification

                  for (j = k; j < nn; j++) {
                     p = pH[off_k+j]+q*pH[off_k1+j];
                     if (notlast) {
                        p = p+r*pH[off_k2+j];
                        pH[off_k2+j] = pH[off_k2+j]-p*z;
                     }
                     pH[off_k+j]  = pH[off_k+j]-p*x;
                     pH[off_k1+j] = pH[off_k1+j]-p*y;
                  }

                  // Column modification

                  for (i = 0; i <= TMath::Min(n,k+3); i++) {
                     const Int_t off_i = i*nn;
                     p = x*pH[off_i+k]+y*pH[off_i+k+1];
                     if (notlast) {
                        p = p+z*pH[off_i+k+2];
                        pH[off_i+k+2] = pH[off_i+k+2]-p*r;
                     }
                     pH[off_i+k]   = pH[off_i+k]-p;
                     pH[off_i+k+1] = pH[off_i+k+1]-p*q;
                  }

                  // Accumulate transformations

                  for (i = low; i <= high; i++) {
                     const Int_t off_i = i*nn;
                     p = x*pV[off_i+k]+y*pV[off_i+k+1];
                     if (notlast) {
                        p = p+z*pV[off_i+k+2];
                        pV[off_i+k+2] = pV[off_i+k+2]-p*r;
                     }
                     pV[off_i+k]   = pV[off_i+k]-p;
                     pV[off_i+k+1] = pV[off_i+k+1]-p*q;
                  }
               }  // (s != 0)
            }  // k loop
         }  // check convergence
      }  // while (n >= low)

      // Backsubstitute to find vectors of upper triangular form

      if (norm == 0.0)
         return;

      for (n = nn-1; n >= 0; n--) {
         p = pD[n];
         q = pE[n];

         // Double_t vector

         const Int_t off_n = n*nn;
         if (q == 0) {
            Int_t l = n;
            pH[off_n+n] = 1.0;
            for (i = n-1; i >= 0; i--) {
               const Int_t off_i  = i*nn;
               const Int_t off_i1 = (i+1)*nn;
               w = pH[off_i+i]-p;
               r = 0.0;
               for (j = l; j <= n; j++) {
                  const Int_t off_j = j*nn;
                  r = r+pH[off_i+j]*pH[off_j+n];
               }
               if (pE[i] < 0.0) {
                  z = w;
                  s = r;
               } else {
                  l = i;
                  if (pE[i] == 0.0) {
                     if (w != 0.0)
                        pH[off_i+n] = -r/w;
                  else
                     pH[off_i+n] = -r/(eps*norm);

                  // Solve real equations

               } else {
                  x = pH[off_i+i+1];
                  y = pH[off_i1+i];
                  q = (pD[i]-p)*(pD[i]-p)+pE[i]*pE[i];
                  t = (x*s-z*r)/q;
                  pH[off_i+n] = t;
                  if (TMath::Abs(x) > TMath::Abs(z))
                     pH[i+1+n] = (-r-w*t)/x;
                  else
                     pH[i+1+n] = (-s-y*t)/z;
               }

               // Overflow control

               t = TMath::Abs(pH[off_i+n]);
               if ((eps*t)*t > 1) {
                  for (j = i; j <= n; j++) {
                     const Int_t off_j = j*nn;
                     pH[off_j+n] = pH[off_j+n]/t;
                  }
               }
            }
         }

         // Complex vector

      } else if (q < 0) {
         Int_t l = n-1;
         const Int_t off_n1 = (n-1)*nn;

         // Last vector component imaginary so matrix is triangular

         if (TMath::Abs(pH[off_n+n-1]) > TMath::Abs(pH[off_n1+n])) {
            pH[off_n1+n-1] = q/pH[off_n+n-1];
            pH[off_n1+n]   = -(pH[off_n+n]-p)/pH[off_n+n-1];
         } else {
            cdiv(0.0,-pH[off_n1+n],pH[off_n1+n-1]-p,q);
            pH[off_n1+n-1] = gCdivr;
            pH[off_n1+n]   = gCdivi;
         }
         pH[off_n+n-1] = 0.0;
         pH[off_n+n]   = 1.0;
         for (i = n-2; i >= 0; i--) {
            const Int_t off_i  = i*nn;
            const Int_t off_i1 = (i+1)*nn;
            Double_t ra = 0.0;
            Double_t sa = 0.0;
            for (j = l; j <= n; j++) {
               const Int_t off_j = j*nn;
               ra += pH[off_i+j]*pH[off_j+n-1];
               sa += pH[off_i+j]*pH[off_j+n];
            }
            w = pH[off_i+i]-p;

            if (pE[i] < 0.0) {
               z = w;
               r = ra;
               s = sa;
            } else {
               l = i;
               if (pE[i] == 0) {
                  cdiv(-ra,-sa,w,q);
                  pH[off_i+n-1] = gCdivr;
                  pH[off_i+n]   = gCdivi;
               } else {

                  // Solve complex equations

                  x = pH[off_i+i+1];
                  y = pH[off_i1+i];
                  Double_t vr = (pD[i]-p)*(pD[i]-p)+pE[i]*pE[i]-q*q;
                  Double_t vi = (pD[i]-p)*2.0*q;
                  if ((vr == 0.0) && (vi == 0.0)) {
                     vr = eps*norm*(TMath::Abs(w)+TMath::Abs(q)+
                          TMath::Abs(x)+TMath::Abs(y)+TMath::Abs(z));
                  }
                  cdiv(x*r-z*ra+q*sa,x*s-z*sa-q*ra,vr,vi);
                  pH[off_i+n-1] = gCdivr;
                  pH[off_i+n]   = gCdivi;
                  if (TMath::Abs(x) > (TMath::Abs(z)+TMath::Abs(q))) {
                     pH[off_i1+n-1] = (-ra-w*pH[off_i+n-1]+q*pH[off_i+n])/x;
                     pH[off_i1+n]   = (-sa-w*pH[off_i+n]-q*pH[off_i+n-1])/x;
                  } else {
                     cdiv(-r-y*pH[off_i+n-1],-s-y*pH[off_i+n],z,q);
                     pH[off_i1+n-1] = gCdivr;
                     pH[off_i1+n]   = gCdivi;
                  }
               }

               // Overflow control

               t = TMath::Max(TMath::Abs(pH[off_i+n-1]),TMath::Abs(pH[off_i+n]));
               if ((eps*t)*t > 1) {
                  for (j = i; j <= n; j++) {
                     const Int_t off_j = j*nn;
                     pH[off_j+n-1] = pH[off_j+n-1]/t;
                     pH[off_j+n]   = pH[off_j+n]/t;
                  }
               }
            }
         }
      }
   }

   // Vectors of isolated roots

   for (i = 0; i < nn; i++) {
      if (i < low || i > high) {
         const Int_t off_i = i*nn;
         for (j = i; j < nn; j++)
            pV[off_i+j] = pH[off_i+j];
      }
   }

   // Back transformation to get eigenvectors of original matrix

   for (j = nn-1; j >= low; j--) {
      for (i = low; i <= high; i++) {
         const Int_t off_i = i*nn;
         z = 0.0;
         for (k = low; k <= TMath::Min(j,high); k++) {
            const Int_t off_k = k*nn;
            z = z+pV[off_i+k]*pH[off_k+j];
         }
         pV[off_i+j] = z;
      }
   }

}

//______________________________________________________________________________
void TMatrixDEigen::Sort(TMatrixD &v,TVectorD &d,TVectorD &e)
{
// Sort eigenvalues and corresponding vectors in descending order of Re^2+Im^2
// of the complex eigenvalues .

   // Sort eigenvalues and corresponding vectors.
   Double_t *pV = v.GetMatrixArray();
   Double_t *pD = d.GetMatrixArray();
   Double_t *pE = e.GetMatrixArray();

   const Int_t n = v.GetNrows();

   for (Int_t i = 0; i < n-1; i++) {
      Int_t k = i;
      Double_t norm = pD[i]*pD[i]+pE[i]*pE[i];
      Int_t j;
      for (j = i+1; j < n; j++) {
         const Double_t norm_new = pD[j]*pD[j]+pE[j]*pE[j];
         if (norm_new > norm) {
            k = j;
            norm = norm_new;
         }
      }
      if (k != i) {
         Double_t tmp;
         tmp   = pD[k];
         pD[k] = pD[i];
         pD[i] = tmp;
         tmp   = pE[k];
         pE[k] = pE[i];
         pE[i] = tmp;
         for (j = 0; j < n; j++) {
            const Int_t off_j = j*n;
            tmp = pV[off_j+i];
            pV[off_j+i] = pV[off_j+k];
            pV[off_j+k] = tmp;
         }
      }
   }
}

//______________________________________________________________________________
TMatrixDEigen &TMatrixDEigen::operator=(const TMatrixDEigen &source)
{
// Assignment operator

   if (this != &source) {
      fEigenVectors.ResizeTo(source.fEigenVectors);
      fEigenValuesRe.ResizeTo(source.fEigenValuesRe);
      fEigenValuesIm.ResizeTo(source.fEigenValuesIm);
   }
   return *this;
}

//______________________________________________________________________________
const TMatrixD TMatrixDEigen::GetEigenValues() const
{
// Computes the block diagonal eigenvalue matrix.
// If the original matrix A is not symmetric, then the eigenvalue
// matrix D is block diagonal with the real eigenvalues in 1-by-1
// blocks and any complex eigenvalues,
//    a + i*b, in 2-by-2 blocks, [a, b; -b, a].
//  That is, if the complex eigenvalues look like
//
//     u + iv     .        .          .      .    .
//       .      u - iv     .          .      .    .
//       .        .      a + ib       .      .    .
//       .        .        .        a - ib   .    .
//       .        .        .          .      x    .
//       .        .        .          .      .    y
//
// then D looks like
//
//     u        v        .          .      .    .
//    -v        u        .          .      .    .
//     .        .        a          b      .    .
//     .        .       -b          a      .    .
//     .        .        .          .      x    .
//     .        .        .          .      .    y
//
// This keeps V a real matrix in both symmetric and non-symmetric
// cases, and A*V = V*D.
//
// Indexing:
//  If matrix A has the index/shape (rowLwb,rowUpb,rowLwb,rowUpb)
//  each eigen-vector must have the shape (rowLwb,rowUpb) .
//  For convinience, the column index of the eigen-vector matrix
//  also runs from rowLwb to rowUpb so that the returned matrix
//  has also index/shape (rowLwb,rowUpb,rowLwb,rowUpb) .
//
   const Int_t nrows  = fEigenVectors.GetNrows();
   const Int_t rowLwb = fEigenVectors.GetRowLwb();
   const Int_t rowUpb = rowLwb+nrows-1;

   TMatrixD mD(rowLwb,rowUpb,rowLwb,rowUpb);

   Double_t *pD = mD.GetMatrixArray();
   const Double_t * const pd = fEigenValuesRe.GetMatrixArray();
   const Double_t * const pe = fEigenValuesIm.GetMatrixArray();

   for (Int_t i = 0; i < nrows; i++) {
      const Int_t off_i = i*nrows;
      for (Int_t j = 0; j < nrows; j++)
         pD[off_i+j] = 0.0;
      pD[off_i+i] = pd[i];
      if (pe[i] > 0) {
         pD[off_i+i+1] = pe[i];
      } else if (pe[i] < 0) {
         pD[off_i+i-1] = pe[i];
      }
   }

   return mD;
}
