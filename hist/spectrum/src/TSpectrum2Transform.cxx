// @(#)root/spectrum:$Id$
// Author: Miroslav Morhac   25/09/06

/** \class TSpectrum2Transform
    \ingroup Spectrum
    \brief Advanced 2-dimensional orthogonal transform functions
    \author Miroslav Morhac

 \legacy{TSpectrum2Transform}

 Class to carry out transforms of 2D spectra, its filtering and
 enhancement. It allows to calculate classic Fourier, Cosine, Sin,
 Hartley, Walsh, Haar transforms as well as mixed transforms (Fourier-
 Walsh, Fourier-Haar, Walsh-Haar, Cosine-Walsh, Cosine-Haar, Sin-Walsh
 and Sin-Haar). All the transforms are fast.

 The algorithms in this class have been published in the following
 references:

   1. C.V. Hampton, B. Lian, Wm. C. McHarris: Fast-Fourier-transform
       spectral enhancement techniques for gamma-ray spectroscopy.NIM A353
       (1994) 280-284.
   2. Morhac M., Matousek V., New adaptive Cosine-Walsh  transform and
       its application to nuclear data compression, IEEE Transactions on
       Signal Processing 48 (2000) 2693.
   3. Morhac M., Matousek V., Data compression using new fast adaptive
       Cosine-Haar transforms, Digital Signal Processing 8 (1998) 63.
   4. Morhac M., Matousek V.: Multidimensional nuclear data compression
       using fast adaptive Walsh-Haar transform. Acta Physica Slovaca 51
      (2001) 307.
 */

#include "TSpectrum2Transform.h"
#include "TMath.h"

ClassImp(TSpectrum2Transform);

////////////////////////////////////////////////////////////////////////////////
/// Default constructor

TSpectrum2Transform::TSpectrum2Transform()
{
   fSizeX = 0, fSizeY = 0;
   fTransformType = kTransformCos;
   fDegree = 0;
   fDirection = kTransformForward;
   fXmin = 0;
   fXmax = 0;
   fYmin = 0;
   fYmax = 0;
   fFilterCoeff=0;
   fEnhanceCoeff=0.5;
}

////////////////////////////////////////////////////////////////////////////////
/// The constructor creates TSpectrum2Transform object. Its sizes must be > than zero and must be power of 2.
/// It sets default transform type to be Cosine transform. Transform parameters can be changed using setter functions.

TSpectrum2Transform::TSpectrum2Transform(Int_t sizeX, Int_t sizeY) :TObject()
{
   Int_t j1, j2, n;
   if (sizeX <= 0 || sizeY <= 0){
      Error ("TSpectrumTransform","Invalid length, must be > than 0");
      return;
   }
   j1 = 0;
   n = 1;
   for (; n < sizeX;) {
      j1 += 1;
      n = n * 2;
   }
   if (n != sizeX){
      Error ("TSpectrumTransform","Invalid length, must be power of 2");
      return;
   }
   j2 = 0;
   n = 1;
   for (; n < sizeY;) {
      j2 += 1;
      n = n * 2;
   }
   if (n != sizeY){
      Error ("TSpectrumTransform","Invalid length, must be power of 2");
      return;
   }
   fSizeX = sizeX, fSizeY = sizeY;
   fTransformType = kTransformCos;
   fDegree = 0;
   fDirection = kTransformForward;
   fXmin = sizeX/4;
   fXmax = sizeX-1;
   fYmin = sizeY/4;
   fYmax = sizeY-1;
   fFilterCoeff=0;
   fEnhanceCoeff=0.5;
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor

TSpectrum2Transform::~TSpectrum2Transform()
{
}

////////////////////////////////////////////////////////////////////////////////
///   This function calculates Haar transform of a part of data
///
///      Function parameters:
///              - working_space-pointer to vector of transformed data
///              - num-length of processed data
///              - direction-forward or inverse transform

void TSpectrum2Transform::Haar(Double_t *working_space, Int_t num, Int_t direction)
{
   Int_t i, ii, li, l2, l3, j, jj, jj1, lj, iter, m, jmin, jmax;
   Double_t a, b, c, wlk;
   Double_t val;
   for (i = 0; i < num; i++)
      working_space[i + num] = 0;
   i = num;
   iter = 0;
   for (; i > 1;) {
      iter += 1;
      i = i / 2;
   }
   if (direction == kTransformForward) {
      for (m = 1; m <= iter; m++) {
         li = iter + 1 - m;
         l2 = (Int_t) TMath::Power(2, li - 1);
         for (i = 0; i < (2 * l2); i++) {
            working_space[num + i] = working_space[i];
         }
         for (j = 0; j < l2; j++) {
            l3 = l2 + j;
            jj = 2 * j;
            val = working_space[jj + num] + working_space[jj + 1 + num];
            working_space[j] = val;
            val = working_space[jj + num] - working_space[jj + 1 + num];
            working_space[l3] = val;
         }
      }
   }
   val = working_space[0];
   val = val / TMath::Sqrt(TMath::Power(2, iter));
   working_space[0] = val;
   val = working_space[1];
   val = val / TMath::Sqrt(TMath::Power(2, iter));
   working_space[1] = val;
   for (ii = 2; ii <= iter; ii++) {
      i = ii - 1;
      wlk = 1 / TMath::Sqrt(TMath::Power(2, iter - i));
      jmin = (Int_t) TMath::Power(2, i);
      jmax = (Int_t) TMath::Power(2, ii) - 1;
      for (j = jmin; j <= jmax; j++) {
         val = working_space[j];
         a = val;
         a = a * wlk;
         val = a;
         working_space[j] = val;
      }
   }
   if (direction == kTransformInverse) {
      for (m = 1; m <= iter; m++) {
         a = 2;
         b = m - 1;
         c = TMath::Power(a, b);
         li = (Int_t) c;
         for (i = 0; i < (2 * li); i++) {
            working_space[i + num] = working_space[i];
         }
         for (j = 0; j < li; j++) {
            lj = li + j;
            jj = 2 * (j + 1) - 1;
            jj1 = jj - 1;
            val = working_space[j + num] - working_space[lj + num];
            working_space[jj] = val;
            val = working_space[j + num] + working_space[lj + num];
            working_space[jj1] = val;
         }
      }
   }
   return;
}

////////////////////////////////////////////////////////////////////////////////
///   This function calculates Walsh transform of a part of data
///
///      Function parameters:
///              - working_space-pointer to vector of transformed data
///              - num-length of processed data

void TSpectrum2Transform::Walsh(Double_t *working_space, Int_t num)
{
   Int_t i, m, nump = 1, mnum, mnum2, mp, ib, mp2, mnum21, iba, iter;
   Double_t a;
   Double_t val1, val2;
   for (i = 0; i < num; i++)
      working_space[i + num] = 0;
   i = num;
   iter = 0;
   for (; i > 1;) {
      iter += 1;
      i = i / 2;
   }
   for (m = 1; m <= iter; m++) {
      if (m == 1)
         nump = 1;

      else
         nump = nump * 2;
      mnum = num / nump;
      mnum2 = mnum / 2;
      for (mp = 0; mp < nump; mp++) {
         ib = mp * mnum;
         for (mp2 = 0; mp2 < mnum2; mp2++) {
            mnum21 = mnum2 + mp2 + ib;
            iba = ib + mp2;
            val1 = working_space[iba];
            val2 = working_space[mnum21];
            working_space[iba + num] = val1 + val2;
            working_space[mnum21 + num] = val1 - val2;
         }
      }
      for (i = 0; i < num; i++) {
         working_space[i] = working_space[i + num];
      }
   }
   a = num;
   a = TMath::Sqrt(a);
   val2 = a;
   for (i = 0; i < num; i++) {
      val1 = working_space[i];
      val1 = val1 / val2;
      working_space[i] = val1;
   }
   return;
}

////////////////////////////////////////////////////////////////////////////////
///   This function carries out bit-reverse reordering of data
///
///      Function parameters:
///              - working_space-pointer to vector of processed data
///              - num-length of processed data

void TSpectrum2Transform::BitReverse(Double_t *working_space, Int_t num)
{
   Int_t ipower[26];
   Int_t i, ib, il, ibd, ip, ifac, i1;
   for (i = 0; i < num; i++) {
      working_space[i + num] = working_space[i];
   }
   for (i = 1; i <= num; i++) {
      ib = i - 1;
      il = 1;
      lab9: ibd = ib / 2;
      ipower[il - 1] = 1;
      if (ib == (ibd * 2))
         ipower[il - 1] = 0;
      if (ibd == 0)
         goto lab10;
      ib = ibd;
      il = il + 1;
      goto lab9;
      lab10: ip = 1;
      ifac = num;
      for (i1 = 1; i1 <= il; i1++) {
         ifac = ifac / 2;
         ip = ip + ifac * ipower[i1 - 1];
      }
      working_space[ip - 1] = working_space[i - 1 + num];
   }
   return;
}

////////////////////////////////////////////////////////////////////////////////
///   This function calculates Fourier based transform of a part of data
///
///      Function parameters:
///              - working_space-pointer to vector of transformed data
///              - num-length of processed data
///              - hartley-1 if it is Hartley transform, 0 otherwise
///              - direction-forward or inverse transform

void TSpectrum2Transform::Fourier(Double_t *working_space, Int_t num, Int_t hartley,
                           Int_t direction, Int_t zt_clear)
{
   Int_t nxp2, nxp, i, j, k, m, iter, mxp, j1, j2, n1, n2, it;
   Double_t a, b, c, d, sign, wpwr, arg, wr, wi, tr, ti, pi =
       3.14159265358979323846;
   Double_t val1, val2, val3, val4;
   if (direction == kTransformForward && zt_clear == 0) {
      for (i = 0; i < num; i++)
         working_space[i + num] = 0;
   }
   i = num;
   iter = 0;
   for (; i > 1;) {
      iter += 1;
      i = i / 2;
   }
   sign = -1;
   if (direction == kTransformInverse)
      sign = 1;
   nxp2 = num;
   for (it = 1; it <= iter; it++) {
      nxp = nxp2;
      nxp2 = nxp / 2;
      a = nxp2;
      wpwr = pi / a;
      for (m = 1; m <= nxp2; m++) {
         a = m - 1;
         arg = a * wpwr;
         wr = TMath::Cos(arg);
         wi = sign * TMath::Sin(arg);
         for (mxp = nxp; mxp <= num; mxp += nxp) {
            j1 = mxp - nxp + m;
            j2 = j1 + nxp2;
            val1 = working_space[j1 - 1];
            val2 = working_space[j2 - 1];
            val3 = working_space[j1 - 1 + num];
            val4 = working_space[j2 - 1 + num];
            a = val1;
            b = val2;
            c = val3;
            d = val4;
            tr = a - b;
            ti = c - d;
            a = a + b;
            val1 = a;
            working_space[j1 - 1] = val1;
            c = c + d;
            val1 = c;
            working_space[j1 - 1 + num] = val1;
            a = tr * wr - ti * wi;
            val1 = a;
            working_space[j2 - 1] = val1;
            a = ti * wr + tr * wi;
            val1 = a;
            working_space[j2 - 1 + num] = val1;
         }
      }
   }
   n2 = num / 2;
   n1 = num - 1;
   j = 1;
   for (i = 1; i <= n1; i++) {
      if (i >= j)
         goto lab55;
      val1 = working_space[j - 1];
      val2 = working_space[j - 1 + num];
      val3 = working_space[i - 1];
      working_space[j - 1] = val3;
      working_space[j - 1 + num] = working_space[i - 1 + num];
      working_space[i - 1] = val1;
      working_space[i - 1 + num] = val2;
      lab55: k = n2;
      lab60: if (k >= j) goto lab65;
      j = j - k;
      k = k / 2;
      goto lab60;
      lab65: j = j + k;
   }
   a = num;
   a = TMath::Sqrt(a);
   for (i = 0; i < num; i++) {
      if (hartley == 0) {
         val1 = working_space[i];
         b = val1;
         b = b / a;
         val1 = b;
         working_space[i] = val1;
         b = working_space[i + num];
         b = b / a;
         working_space[i + num] = b;
      }

      else {
         b = working_space[i];
         c = working_space[i + num];
         b = (b + c) / a;
         working_space[i] = b;
         working_space[i + num] = 0;
      }
   }
   if (hartley == 1 && direction == kTransformInverse) {
      for (i = 1; i < num; i++)
         working_space[num - i + num] = working_space[i];
      working_space[0 + num] = working_space[0];
      for (i = 0; i < num; i++) {
         working_space[i] = working_space[i + num];
         working_space[i + num] = 0;
      }
   }
   return;
}

////////////////////////////////////////////////////////////////////////////////
///   This function carries out bit-reverse reordering for Haar transform
///
///      Function parameters:
///              - working_space-pointer to vector of processed data
///              - shift-shift of position of processing
///              - start-initial position of processed data
///              - num-length of processed data

void TSpectrum2Transform::BitReverseHaar(Double_t *working_space, Int_t shift, Int_t num,
                                  Int_t start)
{
   Int_t ipower[26];
   Int_t i, ib, il, ibd, ip, ifac, i1;
   for (i = 0; i < num; i++) {
      working_space[i + shift + start] = working_space[i + start];
      working_space[i + shift + start + 2 * shift] =
          working_space[i + start + 2 * shift];
   }
   for (i = 1; i <= num; i++) {
      ib = i - 1;
      il = 1;
      lab9:  ibd = ib / 2;
      ipower[il - 1] = 1;
      if (ib == (ibd * 2))
         ipower[il - 1] = 0;
      if (ibd == 0)
         goto lab10;
      ib = ibd;
      il = il + 1;
      goto lab9;
      lab10:  ip = 1;
      ifac = num;
      for (i1 = 1; i1 <= il; i1++) {
         ifac = ifac / 2;
         ip = ip + ifac * ipower[i1 - 1];
      }
      working_space[ip - 1 + start] =
          working_space[i - 1 + shift + start];
      working_space[ip - 1 + start + 2 * shift] =
          working_space[i - 1 + shift + start + 2 * shift];
   }
   return;
}

////////////////////////////////////////////////////////////////////////////////
///   This function calculates generalized (mixed) transforms of different degrees
///
///      Function parameters:
///              - working_space-pointer to vector of transformed data
///              - zt_clear-flag to clear imaginary data before staring
///              - num-length of processed data
///              - degree-degree of transform (see manual)
///              - type-type of mixed transform (see manual)

Int_t TSpectrum2Transform::GeneralExe(Double_t *working_space, Int_t zt_clear, Int_t num,
                             Int_t degree, Int_t type)
{
   Int_t i, j, k, m, nump, mnum, mnum2, mp, ib, mp2, mnum21, iba, iter,
       mp2step, mppom, ring;
   Double_t a, b, c, d, wpwr, arg, wr, wi, tr, ti, pi =
       3.14159265358979323846;
   Double_t val1, val2, val3, val4, a0oldr = 0, b0oldr = 0, a0r, b0r;
   if (zt_clear == 0) {
      for (i = 0; i < num; i++)
         working_space[i + 2 * num] = 0;
   }
   i = num;
   iter = 0;
   for (; i > 1;) {
      iter += 1;
      i = i / 2;
   }
   a = num;
   wpwr = 2.0 * pi / a;
   nump = num;
   mp2step = 1;
   ring = num;
   for (i = 0; i < iter - degree; i++)
      ring = ring / 2;
   for (m = 1; m <= iter; m++) {
      nump = nump / 2;
      mnum = num / nump;
      mnum2 = mnum / 2;
      if (m > degree
           && (type == kTransformFourierHaar
               || type == kTransformWalshHaar
               || type == kTransformCosHaar
               || type == kTransformSinHaar))
         mp2step *= 2;
      if (ring > 1)
         ring = ring / 2;
      for (mp = 0; mp < nump; mp++) {
         if (type != kTransformWalshHaar) {
            mppom = mp;
            mppom = mppom % ring;
            a = 0;
            j = 1;
            k = num / 4;
            for (i = 0; i < (iter - 1); i++) {
               if ((mppom & j) != 0)
                  a = a + k;
               j = j * 2;
               k = k / 2;
            }
            arg = a * wpwr;
            wr = TMath::Cos(arg);
            wi = TMath::Sin(arg);
         }

         else {
            wr = 1;
            wi = 0;
         }
         ib = mp * mnum;
         for (mp2 = 0; mp2 < mnum2; mp2++) {
            mnum21 = mnum2 + mp2 + ib;
            iba = ib + mp2;
            if (mp2 % mp2step == 0) {
               a0r = a0oldr;
               b0r = b0oldr;
               a0r = 1 / TMath::Sqrt(2.0);
               b0r = 1 / TMath::Sqrt(2.0);
            }

            else {
               a0r = 1;
               b0r = 0;
            }
            val1 = working_space[iba];
            val2 = working_space[mnum21];
            val3 = working_space[iba + 2 * num];
            val4 = working_space[mnum21 + 2 * num];
            a = val1;
            b = val2;
            c = val3;
            d = val4;
            tr = a * a0r + b * b0r;
            val1 = tr;
            working_space[num + iba] = val1;
            ti = c * a0r + d * b0r;
            val1 = ti;
            working_space[num + iba + 2 * num] = val1;
            tr =
                a * b0r * wr - c * b0r * wi - b * a0r * wr + d * a0r * wi;
            val1 = tr;
            working_space[num + mnum21] = val1;
            ti =
                c * b0r * wr + a * b0r * wi - d * a0r * wr - b * a0r * wi;
            val1 = ti;
            working_space[num + mnum21 + 2 * num] = val1;
         }
      }
      for (i = 0; i < num; i++) {
         val1 = working_space[num + i];
         working_space[i] = val1;
         val1 = working_space[num + i + 2 * num];
         working_space[i + 2 * num] = val1;
      }
   }
   return (0);
}

////////////////////////////////////////////////////////////////////////////////
///
///   This function calculates inverse generalized (mixed) transforms
///      Function parameters:
///              - working_space-pointer to vector of transformed data
///              - num-length of processed data
///              - degree-degree of transform (see manual)
///              - type-type of mixed transform (see manual)

Int_t TSpectrum2Transform::GeneralInv(Double_t *working_space, Int_t num, Int_t degree,
                             Int_t type)
{
   Int_t i, j, k, m, nump =
       1, mnum, mnum2, mp, ib, mp2, mnum21, iba, iter, mp2step, mppom,
       ring;
   Double_t a, b, c, d, wpwr, arg, wr, wi, tr, ti, pi =
       3.14159265358979323846;
   Double_t val1, val2, val3, val4, a0oldr = 0, b0oldr = 0, a0r, b0r;
   i = num;
   iter = 0;
   for (; i > 1;) {
      iter += 1;
      i = i / 2;
   }
   a = num;
   wpwr = 2.0 * pi / a;
   mp2step = 1;
   if (type == kTransformFourierHaar || type == kTransformWalshHaar
        || type == kTransformCosHaar || type == kTransformSinHaar) {
      for (i = 0; i < iter - degree; i++)
         mp2step *= 2;
   }
   ring = 1;
   for (m = 1; m <= iter; m++) {
      if (m == 1)
         nump = 1;

      else
         nump = nump * 2;
      mnum = num / nump;
      mnum2 = mnum / 2;
      if (m > iter - degree + 1)
         ring *= 2;
      for (mp = nump - 1; mp >= 0; mp--) {
         if (type != kTransformWalshHaar) {
            mppom = mp;
            mppom = mppom % ring;
            a = 0;
            j = 1;
            k = num / 4;
            for (i = 0; i < (iter - 1); i++) {
               if ((mppom & j) != 0)
                  a = a + k;
               j = j * 2;
               k = k / 2;
            }
            arg = a * wpwr;
            wr = TMath::Cos(arg);
            wi = TMath::Sin(arg);
         }

         else {
            wr = 1;
            wi = 0;
         }
         ib = mp * mnum;
         for (mp2 = 0; mp2 < mnum2; mp2++) {
            mnum21 = mnum2 + mp2 + ib;
            iba = ib + mp2;
            if (mp2 % mp2step == 0) {
               a0r = a0oldr;
               b0r = b0oldr;
               a0r = 1 / TMath::Sqrt(2.0);
               b0r = 1 / TMath::Sqrt(2.0);
            }

            else {
               a0r = 1;
               b0r = 0;
            }
            val1 = working_space[iba];
            val2 = working_space[mnum21];
            val3 = working_space[iba + 2 * num];
            val4 = working_space[mnum21 + 2 * num];
            a = val1;
            b = val2;
            c = val3;
            d = val4;
            tr = a * a0r + b * wr * b0r + d * wi * b0r;
            val1 = tr;
            working_space[num + iba] = val1;
            ti = c * a0r + d * wr * b0r - b * wi * b0r;
            val1 = ti;
            working_space[num + iba + 2 * num] = val1;
            tr = a * b0r - b * wr * a0r - d * wi * a0r;
            val1 = tr;
            working_space[num + mnum21] = val1;
            ti = c * b0r - d * wr * a0r + b * wi * a0r;
            val1 = ti;
            working_space[num + mnum21 + 2 * num] = val1;
         }
      }
      if (m <= iter - degree
           && (type == kTransformFourierHaar
               || type == kTransformWalshHaar
               || type == kTransformCosHaar
               || type == kTransformSinHaar))
         mp2step /= 2;
      for (i = 0; i < num; i++) {
         val1 = working_space[num + i];
         working_space[i] = val1;
         val1 = working_space[num + i + 2 * num];
         working_space[i + 2 * num] = val1;
      }
   }
   return (0);
}

////////////////////////////////////////////////////////////////////////////////
///
///   This function calculates 2D Haar and Walsh transforms
///      Function parameters:
///              - working_matrix-pointer to matrix of transformed data
///              - working_vector-pointer to vector where the data are processed
///              - numx,numy-lengths of processed data
///              - direction-forward or inverse
///              - type-type of transform (see manual)

void TSpectrum2Transform::HaarWalsh2(Double_t **working_matrix,
                              Double_t *working_vector, Int_t numx, Int_t numy,
                              Int_t direction, Int_t type)
{
   Int_t i, j;
   if (direction == kTransformForward) {
      for (j = 0; j < numy; j++) {
         for (i = 0; i < numx; i++) {
            working_vector[i] = working_matrix[i][j];
         }
         switch (type) {
         case kTransformHaar:
            Haar(working_vector, numx, kTransformForward);
            break;
         case kTransformWalsh:
            Walsh(working_vector, numx);
            BitReverse(working_vector, numx);
            break;
         }
         for (i = 0; i < numx; i++) {
            working_matrix[i][j] = working_vector[i];
         }
      }
      for (i = 0; i < numx; i++) {
         for (j = 0; j < numy; j++) {
            working_vector[j] = working_matrix[i][j];
         }
         switch (type) {
         case kTransformHaar:
            Haar(working_vector, numy, kTransformForward);
            break;
         case kTransformWalsh:
            Walsh(working_vector, numy);
            BitReverse(working_vector, numy);
            break;
         }
         for (j = 0; j < numy; j++) {
            working_matrix[i][j] = working_vector[j];
         }
      }
   }

   else if (direction == kTransformInverse) {
      for (i = 0; i < numx; i++) {
         for (j = 0; j < numy; j++) {
            working_vector[j] = working_matrix[i][j];
         }
         switch (type) {
         case kTransformHaar:
            Haar(working_vector, numy, kTransformInverse);
            break;
         case kTransformWalsh:
            BitReverse(working_vector, numy);
            Walsh(working_vector, numy);
            break;
         }
         for (j = 0; j < numy; j++) {
            working_matrix[i][j] = working_vector[j];
         }
      }
      for (j = 0; j < numy; j++) {
         for (i = 0; i < numx; i++) {
            working_vector[i] = working_matrix[i][j];
         }
         switch (type) {
         case kTransformHaar:
            Haar(working_vector, numx, kTransformInverse);
            break;
         case kTransformWalsh:
            BitReverse(working_vector, numx);
            Walsh(working_vector, numx);
            break;
         }
         for (i = 0; i < numx; i++) {
            working_matrix[i][j] = working_vector[i];
         }
      }
   }
   return;
}

////////////////////////////////////////////////////////////////////////////////
///
///   This function calculates 2D Fourier based transforms
///      Function parameters:
///              - working_matrix-pointer to matrix of transformed data
///              - working_vector-pointer to vector where the data are processed
///              - numx,numy-lengths of processed data
///              - direction-forward or inverse
///              - type-type of transform (see manual)

void TSpectrum2Transform::FourCos2(Double_t **working_matrix, Double_t *working_vector,
                            Int_t numx, Int_t numy, Int_t direction, Int_t type)
{
   Int_t i, j, iterx, itery, n, size;
   Double_t pi = 3.14159265358979323846;
   j = 0;
   n = 1;
   for (; n < numx;) {
      j += 1;
      n = n * 2;
   }
   j = 0;
   n = 1;
   for (; n < numy;) {
      j += 1;
      n = n * 2;
   }
   i = numx;
   iterx = 0;
   for (; i > 1;) {
      iterx += 1;
      i = i / 2;
   }
   i = numy;
   itery = 0;
   for (; i > 1;) {
      itery += 1;
      i = i / 2;
   }
   size = numx;
   if (size < numy)
      size = numy;
   if (direction == kTransformForward) {
      for (j = 0; j < numy; j++) {
         for (i = 0; i < numx; i++) {
            working_vector[i] = working_matrix[i][j];
         }
         switch (type) {
         case kTransformCos:
            for (i = 1; i <= numx; i++) {
               working_vector[2 * numx - i] = working_vector[i - 1];
            }
            Fourier(working_vector, 2 * numx, 0, kTransformForward, 0);
            for (i = 0; i < numx; i++) {
               working_vector[i] =
                   working_vector[i] / TMath::Cos(pi * i / (2 * numx));
            }
            working_vector[0] = working_vector[0] / TMath::Sqrt(2.);
            break;
         case kTransformSin:
            for (i = 1; i <= numx; i++) {
               working_vector[2 * numx - i] = -working_vector[i - 1];
            }
            Fourier(working_vector, 2 * numx, 0, kTransformForward, 0);
            for (i = 1; i < numx; i++) {
               working_vector[i - 1] =
                   working_vector[i] / TMath::Sin(pi * i / (2 * numx));
            }
            working_vector[numx - 1] =
                working_vector[numx] / TMath::Sqrt(2.);
            break;
         case kTransformFourier:
            Fourier(working_vector, numx, 0, kTransformForward, 0);
            break;
         case kTransformHartley:
            Fourier(working_vector, numx, 1, kTransformForward, 0);
            break;
         }
         for (i = 0; i < numx; i++) {
            working_matrix[i][j] = working_vector[i];
            if (type == kTransformFourier)
               working_matrix[i][j + numy] = working_vector[i + numx];

            else
               working_matrix[i][j + numy] = working_vector[i + 2 * numx];
         }
      }
      for (i = 0; i < numx; i++) {
         for (j = 0; j < numy; j++) {
            working_vector[j] = working_matrix[i][j];
            if (type == kTransformFourier)
               working_vector[j + numy] = working_matrix[i][j + numy];

            else
               working_vector[j + 2 * numy] = working_matrix[i][j + numy];
         }
         switch (type) {
         case kTransformCos:
            for (j = 1; j <= numy; j++) {
               working_vector[2 * numy - j] = working_vector[j - 1];
            }
            Fourier(working_vector, 2 * numy, 0, kTransformForward, 0);
            for (j = 0; j < numy; j++) {
               working_vector[j] =
                   working_vector[j] / TMath::Cos(pi * j / (2 * numy));
               working_vector[j + 2 * numy] = 0;
            }
            working_vector[0] = working_vector[0] / TMath::Sqrt(2.);
            break;
         case kTransformSin:
            for (j = 1; j <= numy; j++) {
               working_vector[2 * numy - j] = -working_vector[j - 1];
            }
            Fourier(working_vector, 2 * numy, 0, kTransformForward, 0);
            for (j = 1; j < numy; j++) {
               working_vector[j - 1] =
                   working_vector[j] / TMath::Sin(pi * j / (2 * numy));
               working_vector[j + numy] = 0;
            }
            working_vector[numy - 1] =
                working_vector[numy] / TMath::Sqrt(2.);
            working_vector[numy] = 0;
            break;
         case kTransformFourier:
            Fourier(working_vector, numy, 0, kTransformForward, 1);
            break;
         case kTransformHartley:
            Fourier(working_vector, numy, 1, kTransformForward, 0);
            break;
         }
         for (j = 0; j < numy; j++) {
            working_matrix[i][j] = working_vector[j];
            if (type == kTransformFourier)
               working_matrix[i][j + numy] = working_vector[j + numy];

            else
               working_matrix[i][j + numy] = working_vector[j + 2 * numy];
         }
      }
   }

   else if (direction == kTransformInverse) {
      for (i = 0; i < numx; i++) {
         for (j = 0; j < numy; j++) {
            working_vector[j] = working_matrix[i][j];
            if (type == kTransformFourier)
               working_vector[j + numy] = working_matrix[i][j + numy];

            else
               working_vector[j + 2 * numy] = working_matrix[i][j + numy];
         }
         switch (type) {
         case kTransformCos:
            working_vector[0] = working_vector[0] * TMath::Sqrt(2.);
            for (j = 0; j < numy; j++) {
               working_vector[j + 2 * numy] =
                   working_vector[j] * TMath::Sin(pi * j / (2 * numy));
               working_vector[j] =
                   working_vector[j] * TMath::Cos(pi * j / (2 * numy));
            }
            for (j = 1; j < numy; j++) {
               working_vector[2 * numy - j] = working_vector[j];
               working_vector[2 * numy - j + 2 * numy] =
                   -working_vector[j + 2 * numy];
            }
            working_vector[numy] = 0;
            working_vector[numy + 2 * numy] = 0;
            Fourier(working_vector, 2 * numy, 0, kTransformInverse, 1);
            break;
         case kTransformSin:
            working_vector[numy] =
                working_vector[numy - 1] * TMath::Sqrt(2.);
            for (j = numy - 1; j > 0; j--) {
               working_vector[j + 2 * numy] =
                   -working_vector[j -
                                   1] * TMath::Cos(pi * j / (2 * numy));
               working_vector[j] =
                   working_vector[j - 1] * TMath::Sin(pi * j / (2 * numy));
            }
            for (j = 1; j < numy; j++) {
               working_vector[2 * numy - j] = working_vector[j];
               working_vector[2 * numy - j + 2 * numy] =
                   -working_vector[j + 2 * numy];
            }
            working_vector[0] = 0;
            working_vector[0 + 2 * numy] = 0;
            working_vector[numy + 2 * numy] = 0;
            Fourier(working_vector, 2 * numy, 0, kTransformInverse, 1);
            break;
         case kTransformFourier:
            Fourier(working_vector, numy, 0, kTransformInverse, 1);
            break;
         case kTransformHartley:
            Fourier(working_vector, numy, 1, kTransformInverse, 1);
            break;
         }
         for (j = 0; j < numy; j++) {
            working_matrix[i][j] = working_vector[j];
            if (type == kTransformFourier)
               working_matrix[i][j + numy] = working_vector[j + numy];

            else
               working_matrix[i][j + numy] = working_vector[j + 2 * numy];
         }
      }
      for (j = 0; j < numy; j++) {
         for (i = 0; i < numx; i++) {
            working_vector[i] = working_matrix[i][j];
            if (type == kTransformFourier)
               working_vector[i + numx] = working_matrix[i][j + numy];

            else
               working_vector[i + 2 * numx] = working_matrix[i][j + numy];
         }
         switch (type) {
         case kTransformCos:
            working_vector[0] = working_vector[0] * TMath::Sqrt(2.);
            for (i = 0; i < numx; i++) {
               working_vector[i + 2 * numx] =
                   working_vector[i] * TMath::Sin(pi * i / (2 * numx));
               working_vector[i] =
                   working_vector[i] * TMath::Cos(pi * i / (2 * numx));
            }
            for (i = 1; i < numx; i++) {
               working_vector[2 * numx - i] = working_vector[i];
               working_vector[2 * numx - i + 2 * numx] =
                   -working_vector[i + 2 * numx];
            }
            working_vector[numx] = 0;
            working_vector[numx + 2 * numx] = 0;
            Fourier(working_vector, 2 * numx, 0, kTransformInverse, 1);
            break;
         case kTransformSin:
            working_vector[numx] =
                working_vector[numx - 1] * TMath::Sqrt(2.);
            for (i = numx - 1; i > 0; i--) {
               working_vector[i + 2 * numx] =
                   -working_vector[i -
                                   1] * TMath::Cos(pi * i / (2 * numx));
               working_vector[i] =
                   working_vector[i - 1] * TMath::Sin(pi * i / (2 * numx));
            }
            for (i = 1; i < numx; i++) {
               working_vector[2 * numx - i] = working_vector[i];
               working_vector[2 * numx - i + 2 * numx] =
                   -working_vector[i + 2 * numx];
            }
            working_vector[0] = 0;
            working_vector[0 + 2 * numx] = 0;
            working_vector[numx + 2 * numx] = 0;
            Fourier(working_vector, 2 * numx, 0, kTransformInverse, 1);
            break;
         case kTransformFourier:
            Fourier(working_vector, numx, 0, kTransformInverse, 1);
            break;
         case kTransformHartley:
            Fourier(working_vector, numx, 1, kTransformInverse, 1);
            break;
         }
         for (i = 0; i < numx; i++) {
            working_matrix[i][j] = working_vector[i];
         }
      }
   }
   return;
}

////////////////////////////////////////////////////////////////////////////////
///
///   This function calculates generalized (mixed) 2D transforms
///      Function parameters:
///              - working_matrix-pointer to matrix of transformed data
///              - working_vector-pointer to vector where the data are processed
///              - numx,numy-lengths of processed data
///              - direction-forward or inverse
///              - type-type of transform (see manual)
///              - degree-degree of transform (see manual)

void TSpectrum2Transform::General2(Double_t **working_matrix, Double_t *working_vector,
                            Int_t numx, Int_t numy, Int_t direction, Int_t type,
                            Int_t degree)
{
   Int_t i, j, jstup, kstup, l, m;
   Double_t val, valx, valz;
   Double_t a, b, pi = 3.14159265358979323846;
   if (direction == kTransformForward) {
      for (j = 0; j < numy; j++) {
         kstup = (Int_t) TMath::Power(2, degree);
         jstup = numx / kstup;
         for (i = 0; i < numx; i++) {
            val = working_matrix[i][j];
            if (type == kTransformCosWalsh
                 || type == kTransformCosHaar) {
               jstup = (Int_t) TMath::Power(2, degree) / 2;
               kstup = i / jstup;
               kstup = 2 * kstup * jstup;
               working_vector[kstup + i % jstup] = val;
               working_vector[kstup + 2 * jstup - 1 - i % jstup] = val;
            }

            else if (type == kTransformSinWalsh
                     || type == kTransformSinHaar) {
               jstup = (Int_t) TMath::Power(2, degree) / 2;
               kstup = i / jstup;
               kstup = 2 * kstup * jstup;
               working_vector[kstup + i % jstup] = val;
               working_vector[kstup + 2 * jstup - 1 - i % jstup] = -val;
            }

            else
               working_vector[i] = val;
         }
         switch (type) {
         case kTransformFourierWalsh:
         case kTransformFourierHaar:
         case kTransformWalshHaar:
            GeneralExe(working_vector, 0, numx, degree, type);
            for (i = 0; i < jstup; i++)
               BitReverseHaar(working_vector, numx, kstup, i * kstup);
            break;
         case kTransformCosWalsh:
         case kTransformCosHaar:
            m = (Int_t) TMath::Power(2, degree);
            l = 2 * numx / m;
            for (i = 0; i < l; i++)
               BitReverseHaar(working_vector, 2 * numx, m, i * m);
            GeneralExe(working_vector, 0, 2 * numx, degree, type);
            for (i = 0; i < numx; i++) {
               kstup = i / jstup;
               kstup = 2 * kstup * jstup;
               a = pi * (Double_t) (i % jstup) / (Double_t) (2 * jstup);
               a = TMath::Cos(a);
               b = working_vector[kstup + i % jstup];
               if (i % jstup == 0)
                  a = b / TMath::Sqrt(2.0);

               else
                  a = b / a;
               working_vector[i] = a;
               working_vector[i + 4 * numx] = 0;
            }
            break;
         case kTransformSinWalsh:
         case kTransformSinHaar:
            m = (Int_t) TMath::Power(2, degree);
            l = 2 * numx / m;
            for (i = 0; i < l; i++)
               BitReverseHaar(working_vector, 2 * numx, m, i * m);
            GeneralExe(working_vector, 0, 2 * numx, degree, type);
            for (i = 0; i < numx; i++) {
               kstup = i / jstup;
               kstup = 2 * kstup * jstup;
               a = pi * (Double_t) (i % jstup) / (Double_t) (2 * jstup);
               a = TMath::Cos(a);
               b = working_vector[jstup + kstup + i % jstup];
               if (i % jstup == 0)
                  a = b / TMath::Sqrt(2.0);

               else
                  a = b / a;
               working_vector[jstup + kstup / 2 - i % jstup - 1] = a;
               working_vector[i + 4 * numx] = 0;
            }
            break;
         }
         if (type > kTransformWalshHaar)
            kstup = (Int_t) TMath::Power(2, degree - 1);

         else
            kstup = (Int_t) TMath::Power(2, degree);
         jstup = numx / kstup;
         for (i = 0, l = 0; i < numx; i++, l = (l + kstup) % numx) {
            working_vector[numx + i] = working_vector[l + i / jstup];
            if (type == kTransformFourierWalsh
                 || type == kTransformFourierHaar
                 || type == kTransformWalshHaar)
               working_vector[numx + i + 2 * numx] =
                   working_vector[l + i / jstup + 2 * numx];

            else
               working_vector[numx + i + 4 * numx] =
                   working_vector[l + i / jstup + 4 * numx];
         }
         for (i = 0; i < numx; i++) {
            working_vector[i] = working_vector[numx + i];
            if (type == kTransformFourierWalsh
                 || type == kTransformFourierHaar
                 || type == kTransformWalshHaar)
               working_vector[i + 2 * numx] =
                   working_vector[numx + i + 2 * numx];

            else
               working_vector[i + 4 * numx] =
                   working_vector[numx + i + 4 * numx];
         }
         for (i = 0; i < numx; i++) {
            working_matrix[i][j] = working_vector[i];
            if (type == kTransformFourierWalsh
                 || type == kTransformFourierHaar
                 || type == kTransformWalshHaar)
               working_matrix[i][j + numy] = working_vector[i + 2 * numx];

            else
               working_matrix[i][j + numy] = working_vector[i + 4 * numx];
         }
      }
      for (i = 0; i < numx; i++) {
         kstup = (Int_t) TMath::Power(2, degree);
         jstup = numy / kstup;
         for (j = 0; j < numy; j++) {
            valx = working_matrix[i][j];
            valz = working_matrix[i][j + numy];
            if (type == kTransformCosWalsh
                 || type == kTransformCosHaar) {
               jstup = (Int_t) TMath::Power(2, degree) / 2;
               kstup = j / jstup;
               kstup = 2 * kstup * jstup;
               working_vector[kstup + j % jstup] = valx;
               working_vector[kstup + 2 * jstup - 1 - j % jstup] = valx;
               working_vector[kstup + j % jstup + 4 * numy] = valz;
               working_vector[kstup + 2 * jstup - 1 - j % jstup +
                               4 * numy] = valz;
            }

            else if (type == kTransformSinWalsh
                     || type == kTransformSinHaar) {
               jstup = (Int_t) TMath::Power(2, degree) / 2;
               kstup = j / jstup;
               kstup = 2 * kstup * jstup;
               working_vector[kstup + j % jstup] = valx;
               working_vector[kstup + 2 * jstup - 1 - j % jstup] = -valx;
               working_vector[kstup + j % jstup + 4 * numy] = valz;
               working_vector[kstup + 2 * jstup - 1 - j % jstup +
                               4 * numy] = -valz;
            }

            else {
               working_vector[j] = valx;
               working_vector[j + 2 * numy] = valz;
            }
         }
         switch (type) {
         case kTransformFourierWalsh:
         case kTransformFourierHaar:
         case kTransformWalshHaar:
            GeneralExe(working_vector, 1, numy, degree, type);
            for (j = 0; j < jstup; j++)
               BitReverseHaar(working_vector, numy, kstup, j * kstup);
            break;
         case kTransformCosWalsh:
         case kTransformCosHaar:
            m = (Int_t) TMath::Power(2, degree);
            l = 2 * numy / m;
            for (j = 0; j < l; j++)
               BitReverseHaar(working_vector, 2 * numy, m, j * m);
            GeneralExe(working_vector, 1, 2 * numy, degree, type);
            for (j = 0; j < numy; j++) {
               kstup = j / jstup;
               kstup = 2 * kstup * jstup;
               a = pi * (Double_t) (j % jstup) / (Double_t) (2 * jstup);
               a = TMath::Cos(a);
               b = working_vector[kstup + j % jstup];
               if (j % jstup == 0)
                  a = b / TMath::Sqrt(2.0);

               else
                  a = b / a;
               working_vector[j] = a;
               working_vector[j + 4 * numy] = 0;
            }
            break;
         case kTransformSinWalsh:
         case kTransformSinHaar:
            m = (Int_t) TMath::Power(2, degree);
            l = 2 * numy / m;
            for (j = 0; j < l; j++)
               BitReverseHaar(working_vector, 2 * numy, m, j * m);
            GeneralExe(working_vector, 1, 2 * numy, degree, type);
            for (j = 0; j < numy; j++) {
               kstup = j / jstup;
               kstup = 2 * kstup * jstup;
               a = pi * (Double_t) (j % jstup) / (Double_t) (2 * jstup);
               a = TMath::Cos(a);
               b = working_vector[jstup + kstup + j % jstup];
               if (j % jstup == 0)
                  a = b / TMath::Sqrt(2.0);

               else
                  a = b / a;
               working_vector[jstup + kstup / 2 - j % jstup - 1] = a;
               working_vector[j + 4 * numy] = 0;
            }
            break;
         }
         if (type > kTransformWalshHaar)
            kstup = (Int_t) TMath::Power(2, degree - 1);

         else
            kstup = (Int_t) TMath::Power(2, degree);
         jstup = numy / kstup;
         for (j = 0, l = 0; j < numy; j++, l = (l + kstup) % numy) {
            working_vector[numy + j] = working_vector[l + j / jstup];
            if (type == kTransformFourierWalsh
                 || type == kTransformFourierHaar
                 || type == kTransformWalshHaar)
               working_vector[numy + j + 2 * numy] =
                   working_vector[l + j / jstup + 2 * numy];

            else
               working_vector[numy + j + 4 * numy] =
                   working_vector[l + j / jstup + 4 * numy];
         }
         for (j = 0; j < numy; j++) {
            working_vector[j] = working_vector[numy + j];
            if (type == kTransformFourierWalsh
                 || type == kTransformFourierHaar
                 || type == kTransformWalshHaar)
               working_vector[j + 2 * numy] =
                   working_vector[numy + j + 2 * numy];

            else
               working_vector[j + 4 * numy] =
                   working_vector[numy + j + 4 * numy];
         }
         for (j = 0; j < numy; j++) {
            working_matrix[i][j] = working_vector[j];
            if (type == kTransformFourierWalsh
                 || type == kTransformFourierHaar
                 || type == kTransformWalshHaar)
               working_matrix[i][j + numy] = working_vector[j + 2 * numy];

            else
               working_matrix[i][j + numy] = working_vector[j + 4 * numy];
         }
      }
   }

   else if (direction == kTransformInverse) {
      for (i = 0; i < numx; i++) {
         kstup = (Int_t) TMath::Power(2, degree);
         jstup = numy / kstup;
         for (j = 0; j < numy; j++) {
            working_vector[j] = working_matrix[i][j];
            if (type == kTransformFourierWalsh
                 || type == kTransformFourierHaar
                 || type == kTransformWalshHaar)
               working_vector[j + 2 * numy] = working_matrix[i][j + numy];

            else
               working_vector[j + 4 * numy] = working_matrix[i][j + numy];
         }
         if (type > kTransformWalshHaar)
            kstup = (Int_t) TMath::Power(2, degree - 1);

         else
            kstup = (Int_t) TMath::Power(2, degree);
         jstup = numy / kstup;
         for (j = 0, l = 0; j < numy; j++, l = (l + kstup) % numy) {
            working_vector[numy + l + j / jstup] = working_vector[j];
            if (type == kTransformFourierWalsh
                 || type == kTransformFourierHaar
                 || type == kTransformWalshHaar)
               working_vector[numy + l + j / jstup + 2 * numy] =
                   working_vector[j + 2 * numy];

            else
               working_vector[numy + l + j / jstup + 4 * numy] =
                   working_vector[j + 4 * numy];
         }
         for (j = 0; j < numy; j++) {
            working_vector[j] = working_vector[numy + j];
            if (type == kTransformFourierWalsh
                 || type == kTransformFourierHaar
                 || type == kTransformWalshHaar)
               working_vector[j + 2 * numy] =
                   working_vector[numy + j + 2 * numy];

            else
               working_vector[j + 4 * numy] =
                   working_vector[numy + j + 4 * numy];
         }
         switch (type) {
         case kTransformFourierWalsh:
         case kTransformFourierHaar:
         case kTransformWalshHaar:
            for (j = 0; j < jstup; j++)
               BitReverseHaar(working_vector, numy, kstup, j * kstup);
            GeneralInv(working_vector, numy, degree, type);
            break;
         case kTransformCosWalsh:
         case kTransformCosHaar:
            jstup = (Int_t) TMath::Power(2, degree) / 2;
            m = (Int_t) TMath::Power(2, degree);
            l = 2 * numy / m;
            for (j = 0; j < numy; j++) {
               kstup = j / jstup;
               kstup = 2 * kstup * jstup;
               a = pi * (Double_t) (j % jstup) / (Double_t) (2 * jstup);
               if (j % jstup == 0) {
                  working_vector[2 * numy + kstup + j % jstup] =
                      working_vector[j] * TMath::Sqrt(2.0);
                  working_vector[2 * numy + kstup + j % jstup +
                                  4 * numy] = 0;
               }

               else {
                  b = TMath::Sin(a);
                  a = TMath::Cos(a);
                  working_vector[2 * numy + kstup + j % jstup +
                                  4 * numy] =
                      -(Double_t) working_vector[j] * b;
                  working_vector[2 * numy + kstup + j % jstup] =
                      (Double_t) working_vector[j] * a;
            } } for (j = 0; j < numy; j++) {
               kstup = j / jstup;
               kstup = 2 * kstup * jstup;
               if (j % jstup == 0) {
                  working_vector[2 * numy + kstup + jstup] = 0;
                  working_vector[2 * numy + kstup + jstup + 4 * numy] = 0;
               }

               else {
                  working_vector[2 * numy + kstup + 2 * jstup -
                                  j % jstup] =
                      working_vector[2 * numy + kstup + j % jstup];
                  working_vector[2 * numy + kstup + 2 * jstup -
                                  j % jstup + 4 * numy] =
                      -working_vector[2 * numy + kstup + j % jstup +
                                      4 * numy];
               }
            }
            for (j = 0; j < 2 * numy; j++) {
               working_vector[j] = working_vector[2 * numy + j];
               working_vector[j + 4 * numy] =
                   working_vector[2 * numy + j + 4 * numy];
            }
            GeneralInv(working_vector, 2 * numy, degree, type);
            m = (Int_t) TMath::Power(2, degree);
            l = 2 * numy / m;
            for (j = 0; j < l; j++)
               BitReverseHaar(working_vector, 2 * numy, m, j * m);
            break;
         case kTransformSinWalsh:
         case kTransformSinHaar:
            jstup = (Int_t) TMath::Power(2, degree) / 2;
            m = (Int_t) TMath::Power(2, degree);
            l = 2 * numy / m;
            for (j = 0; j < numy; j++) {
               kstup = j / jstup;
               kstup = 2 * kstup * jstup;
               a = pi * (Double_t) (j % jstup) / (Double_t) (2 * jstup);
               if (j % jstup == 0) {
                  working_vector[2 * numy + kstup + jstup + j % jstup] =
                      working_vector[jstup + kstup / 2 - j % jstup -
                                     1] * TMath::Sqrt(2.0);
                  working_vector[2 * numy + kstup + jstup + j % jstup +
                                  4 * numy] = 0;
               }

               else {
                  b = TMath::Sin(a);
                  a = TMath::Cos(a);
                  working_vector[2 * numy + kstup + jstup + j % jstup +
                                  4 * numy] =
                      -(Double_t) working_vector[jstup + kstup / 2 -
                                               j % jstup - 1] * b;
                  working_vector[2 * numy + kstup + jstup + j % jstup] =
                      (Double_t) working_vector[jstup + kstup / 2 -
                                              j % jstup - 1] * a;
            } } for (j = 0; j < numy; j++) {
               kstup = j / jstup;
               kstup = 2 * kstup * jstup;
               if (j % jstup == 0) {
                  working_vector[2 * numy + kstup] = 0;
                  working_vector[2 * numy + kstup + 4 * numy] = 0;
               }

               else {
                  working_vector[2 * numy + kstup + j % jstup] =
                      working_vector[2 * numy + kstup + 2 * jstup -
                                     j % jstup];
                  working_vector[2 * numy + kstup + j % jstup +
                                  4 * numy] =
                      -working_vector[2 * numy + kstup + 2 * jstup -
                                      j % jstup + 4 * numy];
               }
            }
            for (j = 0; j < 2 * numy; j++) {
               working_vector[j] = working_vector[2 * numy + j];
               working_vector[j + 4 * numy] =
                   working_vector[2 * numy + j + 4 * numy];
            }
            GeneralInv(working_vector, 2 * numy, degree, type);
            for (j = 0; j < l; j++)
               BitReverseHaar(working_vector, 2 * numy, m, j * m);
            break;
         }
         for (j = 0; j < numy; j++) {
            if (type > kTransformWalshHaar) {
               kstup = j / jstup;
               kstup = 2 * kstup * jstup;
               valx = working_vector[kstup + j % jstup];
               valz = working_vector[kstup + j % jstup + 4 * numy];
            }

            else {
               valx = working_vector[j];
               valz = working_vector[j + 2 * numy];
            }
            working_matrix[i][j] = valx;
            working_matrix[i][j + numy] = valz;
         }
      }
      for (j = 0; j < numy; j++) {
         kstup = (Int_t) TMath::Power(2, degree);
         jstup = numy / kstup;
         for (i = 0; i < numx; i++) {
            working_vector[i] = working_matrix[i][j];
            if (type == kTransformFourierWalsh
                 || type == kTransformFourierHaar
                 || type == kTransformWalshHaar)
               working_vector[i + 2 * numx] = working_matrix[i][j + numy];

            else
               working_vector[i + 4 * numx] = working_matrix[i][j + numy];
         }
         if (type > kTransformWalshHaar)
            kstup = (Int_t) TMath::Power(2, degree - 1);

         else
            kstup = (Int_t) TMath::Power(2, degree);
         jstup = numx / kstup;
         for (i = 0, l = 0; i < numx; i++, l = (l + kstup) % numx) {
            working_vector[numx + l + i / jstup] = working_vector[i];
            if (type == kTransformFourierWalsh
                 || type == kTransformFourierHaar
                 || type == kTransformWalshHaar)
               working_vector[numx + l + i / jstup + 2 * numx] =
                   working_vector[i + 2 * numx];

            else
               working_vector[numx + l + i / jstup + 4 * numx] =
                   working_vector[i + 4 * numx];
         }
         for (i = 0; i < numx; i++) {
            working_vector[i] = working_vector[numx + i];
            if (type == kTransformFourierWalsh
                 || type == kTransformFourierHaar
                 || type == kTransformWalshHaar)
               working_vector[i + 2 * numx] =
                   working_vector[numx + i + 2 * numx];

            else
               working_vector[i + 4 * numx] =
                   working_vector[numx + i + 4 * numx];
         }
         switch (type) {
         case kTransformFourierWalsh:
         case kTransformFourierHaar:
         case kTransformWalshHaar:
            for (i = 0; i < jstup; i++)
               BitReverseHaar(working_vector, numx, kstup, i * kstup);
            GeneralInv(working_vector, numx, degree, type);
            break;
         case kTransformCosWalsh:
         case kTransformCosHaar:
            jstup = (Int_t) TMath::Power(2, degree) / 2;
            m = (Int_t) TMath::Power(2, degree);
            l = 2 * numx / m;
            for (i = 0; i < numx; i++) {
               kstup = i / jstup;
               kstup = 2 * kstup * jstup;
               a = pi * (Double_t) (i % jstup) / (Double_t) (2 * jstup);
               if (i % jstup == 0) {
                  working_vector[2 * numx + kstup + i % jstup] =
                      working_vector[i] * TMath::Sqrt(2.0);
                  working_vector[2 * numx + kstup + i % jstup +
                                  4 * numx] = 0;
               }

               else {
                  b = TMath::Sin(a);
                  a = TMath::Cos(a);
                  working_vector[2 * numx + kstup + i % jstup +
                                  4 * numx] =
                      -(Double_t) working_vector[i] * b;
                  working_vector[2 * numx + kstup + i % jstup] =
                      (Double_t) working_vector[i] * a;
            } } for (i = 0; i < numx; i++) {
               kstup = i / jstup;
               kstup = 2 * kstup * jstup;
               if (i % jstup == 0) {
                  working_vector[2 * numx + kstup + jstup] = 0;
                  working_vector[2 * numx + kstup + jstup + 4 * numx] = 0;
               }

               else {
                  working_vector[2 * numx + kstup + 2 * jstup -
                                  i % jstup] =
                      working_vector[2 * numx + kstup + i % jstup];
                  working_vector[2 * numx + kstup + 2 * jstup -
                                  i % jstup + 4 * numx] =
                      -working_vector[2 * numx + kstup + i % jstup +
                                      4 * numx];
               }
            }
            for (i = 0; i < 2 * numx; i++) {
               working_vector[i] = working_vector[2 * numx + i];
               working_vector[i + 4 * numx] =
                   working_vector[2 * numx + i + 4 * numx];
            }
            GeneralInv(working_vector, 2 * numx, degree, type);
            m = (Int_t) TMath::Power(2, degree);
            l = 2 * numx / m;
            for (i = 0; i < l; i++)
               BitReverseHaar(working_vector, 2 * numx, m, i * m);
            break;
         case kTransformSinWalsh:
         case kTransformSinHaar:
            jstup = (Int_t) TMath::Power(2, degree) / 2;
            m = (Int_t) TMath::Power(2, degree);
            l = 2 * numx / m;
            for (i = 0; i < numx; i++) {
               kstup = i / jstup;
               kstup = 2 * kstup * jstup;
               a = pi * (Double_t) (i % jstup) / (Double_t) (2 * jstup);
               if (i % jstup == 0) {
                  working_vector[2 * numx + kstup + jstup + i % jstup] =
                      working_vector[jstup + kstup / 2 - i % jstup -
                                     1] * TMath::Sqrt(2.0);
                  working_vector[2 * numx + kstup + jstup + i % jstup +
                                  4 * numx] = 0;
               }

               else {
                  b = TMath::Sin(a);
                  a = TMath::Cos(a);
                  working_vector[2 * numx + kstup + jstup + i % jstup +
                                  4 * numx] =
                      -(Double_t) working_vector[jstup + kstup / 2 -
                                               i % jstup - 1] * b;
                  working_vector[2 * numx + kstup + jstup + i % jstup] =
                      (Double_t) working_vector[jstup + kstup / 2 -
                                              i % jstup - 1] * a;
            } } for (i = 0; i < numx; i++) {
               kstup = i / jstup;
               kstup = 2 * kstup * jstup;
               if (i % jstup == 0) {
                  working_vector[2 * numx + kstup] = 0;
                  working_vector[2 * numx + kstup + 4 * numx] = 0;
               }

               else {
                  working_vector[2 * numx + kstup + i % jstup] =
                      working_vector[2 * numx + kstup + 2 * jstup -
                                     i % jstup];
                  working_vector[2 * numx + kstup + i % jstup +
                                  4 * numx] =
                      -working_vector[2 * numx + kstup + 2 * jstup -
                                      i % jstup + 4 * numx];
               }
            }
            for (i = 0; i < 2 * numx; i++) {
               working_vector[i] = working_vector[2 * numx + i];
               working_vector[i + 4 * numx] =
                   working_vector[2 * numx + i + 4 * numx];
            }
            GeneralInv(working_vector, 2 * numx, degree, type);
            for (i = 0; i < l; i++)
               BitReverseHaar(working_vector, 2 * numx, m, i * m);
            break;
         }
         for (i = 0; i < numx; i++) {
            if (type > kTransformWalshHaar) {
               kstup = i / jstup;
               kstup = 2 * kstup * jstup;
               val = working_vector[kstup + i % jstup];
            }

            else
               val = working_vector[i];
            working_matrix[i][j] = val;
         }
      }
   }
   return;
}

////////////////////////////////////////////////////////////////////////////////
/// This function transforms the source spectrum. The calling program
///      should fill in input parameters.
/// Transformed data are written into dest spectrum.
///
/// Function parameters:
///  - fSource-pointer to the matrix of source spectrum, its size should
///    be fSizex*fSizey except for inverse FOURIER, FOUR-WALSH, FOUR-HAAR
///    transform. These need fSizex*2*fSizey length to supply real and
///    imaginary coefficients.
///  - fDest-pointer to the matrix of destination data, its size should be
///    fSizex*fSizey except for direct FOURIER, FOUR-WALSh, FOUR-HAAR. These
///    need fSizex*2*fSizey length to store real and imaginary coefficients
///  - fSizex,fSizey-basic dimensions of source and dest spectra
///
/// ### Transform methods
///
/// Goal: to analyse experimental data using orthogonal transforms
///
///  - orthogonal transforms can be successfully used for the processing of
///    nuclear spectra (not only)
///
///  - they can be used to remove high frequency noise, to increase
///    signal-to-background ratio as well as to enhance low intensity components [1],
/// to carry out e.g. Fourier analysis etc.
///
///   - we have implemented the function for the calculation of the commonly
/// used orthogonal transforms as well as functions for the filtration and
/// enhancement of experimental data
///
/// #### References:
///
/// [1] C.V. Hampton, B. Lian, Wm. C.
/// McHarris: Fast-Fourier-transform spectral enhancement techniques for gamma-ray
/// spectroscopy. NIM A353 (1994) 280-284.
///
/// [2] Morhac M., Matouoek V.,
/// New adaptive Cosine-Walsh transform and its application to nuclear data
/// compression, IEEE Transactions on Signal Processing 48 (2000) 2693.
///
/// [3] Morhac M., Matouoek V.,
/// Data compression using new fast adaptive Cosine-Haar transforms, Digital Signal
/// Processing 8 (1998) 63.
///
/// [4] Morhac M., Matouoek V.:
/// Multidimensional nuclear data compression using fast adaptive Walsh-Haar
/// transform. Acta Physica Slovaca 51 (2001) 307.
///
/// ### Example 1 - script Transform2.c:
///
/// \image html spectrum2transform_transform_image002.jpg Fig. 1 Original two-dimensional noisy spectrum
///
/// \image html spectrum2transform_transform_image003.jpg Fig. 2 Transformed spectrum from Fig. 1 using Cosine transform. Energy of the transformed data is concentrated around the beginning of the coordinate system
///
/// #### Script:
///
/// Example to illustrate Transform function (class TSpectrumTransform2).
/// To execute this example, do
///
/// `root > .x Transform2.C`
///
/// ~~~ {.cpp}
///   void Transform2() {
///      Int_t i, j;
///      Int_t nbinsx = 256;
///      Int_t nbinsy = 256;
///      Int_t xmin = 0;
///      Int_t xmax = nbinsx;
///      Int_t ymin = 0;
///      Int_t ymax = nbinsy;
///      Double_t ** source = new Double_t *[nbinsx];
///      Double_t ** dest = new Double_t *[nbinsx];
///      for (i=0;i<nbinsx;i++)
///         source[i]=newDouble_t[nbinsy];
///      for (i=0;i<nbinsx;i++)
///         dest[i]=newDouble_t[nbinsy];
///      TH2F *trans = newTH2F("trans","Background estimation",nbinsx,xmin,xmax,nbinsy,ymin,ymax);
///      TFile *f = new TFile("TSpectrum2.root");
///      trans=(TH2F*)f->Get("back3;1");
///      TCanvas *Tr = new TCanvas("Transform","Illustration of transform function",10,10,1000,700);
///      for (i = 0; i < nbinsx; i++){
///         for (j = 0; j < nbinsy; j++){
///            source[i][j] = trans->GetBinContent(i + 1,j + 1);
///         }
///      }
///      TSpectrumTransform2 *t = new TSpectrumTransform2(256,256);
///      t->SetTransformType(t->kTransformCos,0);
///      t->SetDirection(t->kTransformForward);
///      t->Transform(source,dest);
///      for (i = 0; i < nbinsx; i++){
///         for (j = 0; j < nbinsy; j++){
///            trans->SetBinContent(i+ 1, j + 1,dest[i][j]);
///         }
///      }
///      trans->Draw("SURF");
///   }
/// ~~~

void TSpectrum2Transform::Transform(const Double_t **fSource, Double_t **fDest)
{
   Int_t i, j;
   Int_t size;
   Double_t *working_vector = 0, **working_matrix = 0;
   size = (Int_t) TMath::Max(fSizeX, fSizeY);
   switch (fTransformType) {
   case kTransformHaar:
   case kTransformWalsh:
      working_vector = new Double_t[2 * size];
      working_matrix = new Double_t *[fSizeX];
      for (i = 0; i < fSizeX; i++)
         working_matrix[i] = new Double_t[fSizeY];
      break;
   case kTransformCos:
   case kTransformSin:
   case kTransformFourier:
   case kTransformHartley:
   case kTransformFourierWalsh:
   case kTransformFourierHaar:
   case kTransformWalshHaar:
      working_vector = new Double_t[4 * size];
      working_matrix = new Double_t *[fSizeX];
      for (i = 0; i < fSizeX; i++)
         working_matrix[i] = new Double_t[2 * fSizeY];
      break;
   case kTransformCosWalsh:
   case kTransformCosHaar:
   case kTransformSinWalsh:
   case kTransformSinHaar:
      working_vector = new Double_t[8 * size];
      working_matrix = new Double_t *[fSizeX];
      for (i = 0; i < fSizeX; i++)
         working_matrix[i] = new Double_t[2 * fSizeY];
      break;
   }
   if (fDirection == kTransformForward) {
      switch (fTransformType) {
      case kTransformHaar:
         for (i = 0; i < fSizeX; i++) {
            for (j = 0; j < fSizeY; j++) {
               working_matrix[i][j] = fSource[i][j];
            }
         }
         HaarWalsh2(working_matrix, working_vector, fSizeX, fSizeY,
                     fDirection, kTransformHaar);
         for (i = 0; i < fSizeX; i++) {
            for (j = 0; j < fSizeY; j++) {
               fDest[i][j] = working_matrix[i][j];
            }
         }
         break;
      case kTransformWalsh:
         for (i = 0; i < fSizeX; i++) {
            for (j = 0; j < fSizeY; j++) {
               working_matrix[i][j] = fSource[i][j];
            }
         }
         HaarWalsh2(working_matrix, working_vector, fSizeX, fSizeY,
                     fDirection, kTransformWalsh);
         for (i = 0; i < fSizeX; i++) {
            for (j = 0; j < fSizeY; j++) {
               fDest[i][j] = working_matrix[i][j];
            }
         }
         break;
      case kTransformCos:
         for (i = 0; i < fSizeX; i++) {
            for (j = 0; j < fSizeY; j++) {
               working_matrix[i][j] = fSource[i][j];
            }
         }
         FourCos2(working_matrix, working_vector, fSizeX, fSizeY, fDirection,
                   kTransformCos);
         for (i = 0; i < fSizeX; i++) {
            for (j = 0; j < fSizeY; j++) {
               fDest[i][j] = working_matrix[i][j];
            }
         }
         break;
      case kTransformSin:
         for (i = 0; i < fSizeX; i++) {
            for (j = 0; j < fSizeY; j++) {
               working_matrix[i][j] = fSource[i][j];
            }
         }
         FourCos2(working_matrix, working_vector, fSizeX, fSizeY, fDirection,
                   kTransformSin);
         for (i = 0; i < fSizeX; i++) {
            for (j = 0; j < fSizeY; j++) {
               fDest[i][j] = working_matrix[i][j];
            }
         }
         break;
      case kTransformFourier:
         for (i = 0; i < fSizeX; i++) {
            for (j = 0; j < fSizeY; j++) {
               working_matrix[i][j] = fSource[i][j];
            }
         }
         FourCos2(working_matrix, working_vector, fSizeX, fSizeY, fDirection,
                   kTransformFourier);
         for (i = 0; i < fSizeX; i++) {
            for (j = 0; j < fSizeY; j++) {
               fDest[i][j] = working_matrix[i][j];
            }
         }
         for (i = 0; i < fSizeX; i++) {
            for (j = 0; j < fSizeY; j++) {
               fDest[i][j + fSizeY] = working_matrix[i][j + fSizeY];
            }
         }
         break;
      case kTransformHartley:
         for (i = 0; i < fSizeX; i++) {
            for (j = 0; j < fSizeY; j++) {
               working_matrix[i][j] = fSource[i][j];
            }
         }
         FourCos2(working_matrix, working_vector, fSizeX, fSizeY, fDirection,
                   kTransformHartley);
         for (i = 0; i < fSizeX; i++) {
            for (j = 0; j < fSizeY; j++) {
               fDest[i][j] = working_matrix[i][j];
            }
         }
         break;
      case kTransformFourierWalsh:
      case kTransformFourierHaar:
      case kTransformWalshHaar:
      case kTransformCosWalsh:
      case kTransformCosHaar:
      case kTransformSinWalsh:
      case kTransformSinHaar:
         for (i = 0; i < fSizeX; i++) {
            for (j = 0; j < fSizeY; j++) {
               working_matrix[i][j] = fSource[i][j];
            }
         }
         General2(working_matrix, working_vector, fSizeX, fSizeY, fDirection,
                   fTransformType, fDegree);
         for (i = 0; i < fSizeX; i++) {
            for (j = 0; j < fSizeY; j++) {
               fDest[i][j] = working_matrix[i][j];
            }
         }
         if (fTransformType == kTransformFourierWalsh
              || fTransformType == kTransformFourierHaar) {
            for (i = 0; i < fSizeX; i++) {
               for (j = 0; j < fSizeY; j++) {
                  fDest[i][j + fSizeY] = working_matrix[i][j + fSizeY];
               }
            }
         }
         break;
      }
   }

   else if (fDirection == kTransformInverse) {
      switch (fTransformType) {
      case kTransformHaar:
         for (i = 0; i < fSizeX; i++) {
            for (j = 0; j < fSizeY; j++) {
               working_matrix[i][j] = fSource[i][j];
            }
         }
         HaarWalsh2(working_matrix, working_vector, fSizeX, fSizeY,
                     fDirection, kTransformHaar);
         for (i = 0; i < fSizeX; i++) {
            for (j = 0; j < fSizeY; j++) {
               fDest[i][j] = working_matrix[i][j];
            }
         }
         break;
      case kTransformWalsh:
         for (i = 0; i < fSizeX; i++) {
            for (j = 0; j < fSizeY; j++) {
               working_matrix[i][j] = fSource[i][j];
            }
         }
         HaarWalsh2(working_matrix, working_vector, fSizeX, fSizeY,
                     fDirection, kTransformWalsh);
         for (i = 0; i < fSizeX; i++) {
            for (j = 0; j < fSizeY; j++) {
               fDest[i][j] = working_matrix[i][j];
            }
         }
         break;
      case kTransformCos:
         for (i = 0; i < fSizeX; i++) {
            for (j = 0; j < fSizeY; j++) {
               working_matrix[i][j] = fSource[i][j];
            }
         }
         FourCos2(working_matrix, working_vector, fSizeX, fSizeY, fDirection,
                   kTransformCos);
         for (i = 0; i < fSizeX; i++) {
            for (j = 0; j < fSizeY; j++) {
               fDest[i][j] = working_matrix[i][j];
            }
         }
         break;
      case kTransformSin:
         for (i = 0; i < fSizeX; i++) {
            for (j = 0; j < fSizeY; j++) {
               working_matrix[i][j] = fSource[i][j];
            }
         }
         FourCos2(working_matrix, working_vector, fSizeX, fSizeY, fDirection,
                   kTransformSin);
         for (i = 0; i < fSizeX; i++) {
            for (j = 0; j < fSizeY; j++) {
               fDest[i][j] = working_matrix[i][j];
            }
         }
         break;
      case kTransformFourier:
         for (i = 0; i < fSizeX; i++) {
            for (j = 0; j < fSizeY; j++) {
               working_matrix[i][j] = fSource[i][j];
            }
         }
         for (i = 0; i < fSizeX; i++) {
            for (j = 0; j < fSizeY; j++) {
               working_matrix[i][j + fSizeY] = fSource[i][j + fSizeY];
            }
         }
         FourCos2(working_matrix, working_vector, fSizeX, fSizeY, fDirection,
                   kTransformFourier);
         for (i = 0; i < fSizeX; i++) {
            for (j = 0; j < fSizeY; j++) {
               fDest[i][j] = working_matrix[i][j];
            }
         }
         break;
      case kTransformHartley:
         for (i = 0; i < fSizeX; i++) {
            for (j = 0; j < fSizeY; j++) {
               working_matrix[i][j] = fSource[i][j];
            }
         }
         FourCos2(working_matrix, working_vector, fSizeX, fSizeY, fDirection,
                   kTransformHartley);
         for (i = 0; i < fSizeX; i++) {
            for (j = 0; j < fSizeY; j++) {
               fDest[i][j] = working_matrix[i][j];
            }
         }
         break;
      case kTransformFourierWalsh:
      case kTransformFourierHaar:
      case kTransformWalshHaar:
      case kTransformCosWalsh:
      case kTransformCosHaar:
      case kTransformSinWalsh:
      case kTransformSinHaar:
         for (i = 0; i < fSizeX; i++) {
            for (j = 0; j < fSizeY; j++) {
               working_matrix[i][j] = fSource[i][j];
            }
         }
         if (fTransformType == kTransformFourierWalsh
              || fTransformType == kTransformFourierHaar) {
            for (i = 0; i < fSizeX; i++) {
               for (j = 0; j < fSizeY; j++) {
                  working_matrix[i][j + fSizeY] = fSource[i][j + fSizeY];
               }
            }
         }
         General2(working_matrix, working_vector, fSizeX, fSizeY, fDirection,
                   fTransformType, fDegree);
         for (i = 0; i < fSizeX; i++) {
            for (j = 0; j < fSizeY; j++) {
               fDest[i][j] = working_matrix[i][j];
            }
         }
         break;
      }
   }
   for (i = 0; i < fSizeX; i++) {
      if (working_matrix) delete[]working_matrix[i];
   }
   delete[]working_matrix;
   delete[]working_vector;
   return;
}

////////////////////////////////////////////////////////////////////////////////
/// This function transforms the source spectrum. The calling program
///      should fill in input parameters. Then it sets transformed
///      coefficients in the given region to the given
///      filter_coeff and transforms it back
/// Filtered data are written into dest spectrum.
///
/// Function parameters:
///  - fSource-pointer to the matrix of source spectrum, its size should
///    be fSizeX*fSizeY
///  - fDest-pointer to the matrix of destination data, its size should be
///    fSizeX*fSizeY
///
/// ### Example of zonal filtering
///
/// This function transforms the
/// source spectrum (for details see Transform function). Before the FilterZonal
/// function is called the class must be created by constructor and the type of the
/// transform as well as some other parameters must be set using a set of setter
/// functions. The FilterZonal function sets transformed coefficients in the given
/// region (fXmin, fXmax) to the given fFilterCoeff and transforms it back. Filtered
/// data are written into dest spectrum.
///
/// ### Example - script Fitler2.c:
///
/// \image html spectrum2transform_filter_image001.jpg Fig. 1 Original two-dimensional noisy spectrum
/// \image html spectrum2transform_filter_image002.jpg Fig. 2 Filtered spectrum using Cosine transform and zonal filtration (channels in regions (128-255)x(0-255) and (0-255)x(128-255) were set to 0).
///
/// #### Script:
///
/// Example to illustrate zonal filtration (class TSpectrumTransform2). To execute this example, do
///
/// `root > .x Filter2.C`
///
/// ~~~ {.cpp}
///   void Filter2() {
///      Int_t i, j;
///      Int_t nbinsx = 256;
///      Int_t nbinsy = 256;
///      Int_t xmin = 0;
///      Int_t xmax = nbinsx;
///      Int_t ymin = 0;
///      Int_t ymax = nbinsy;
///      Double_t ** source = new Double_t *[nbinsx];
///      Double_t ** dest = new Double_t *[nbinsx];
///      for (i=0;i<nbinsx;i++)
///         source[i] = new Double_t[nbinsy];
///      for (i=0;i<nbinsx;i++)
///         dest[i]= new Double_t[nbinsy];
///      TH2F *trans = new TH2F("trans","Background estimation",nbinsx,xmin,xmax,nbinsy,ymin,ymax);
///      TFile *f = new TFile("TSpectrum2.root");
///      trans=(TH2F*)f->Get("back3;1");
///      TCanvas *Tr = new TCanvas("Transform","Illustration of transform function",10,10,1000,700);
///      for (i = 0; i < nbinsx;i++){
///         for (j = 0; j <nbinsy; j++){
///            source[i][j] = trans->GetBinContent(i + 1,j+ 1);
///         }
///      }
///      TSpectrumTransform2 *t = new TSpectrumTransform2(256,256);
///      t->SetTransformType(t->kTransformCos,0);
///      t->SetRegion(0,255,128,255);
///      t->FilterZonal(source,dest);
///      for (i = 0; i < nbinsx; i++){
///         for (j = 0; j <nbinsy; j++){
///            source[i][j] = dest[i][j];
///         }
///      }
///      t->SetRegion(128,255,0,255);
///      t->FilterZonal(source,dest);
///      trans->Draw("SURF");
///   }
/// ~~~

void TSpectrum2Transform::FilterZonal(const Double_t **fSource, Double_t **fDest)
{
   Int_t i, j;
   Double_t a, old_area = 0, new_area = 0;
   Int_t size;
   Double_t *working_vector = 0, **working_matrix = 0;
   size = (Int_t) TMath::Max(fSizeX, fSizeY);
   switch (fTransformType) {
   case kTransformHaar:
   case kTransformWalsh:
      working_vector = new Double_t[2 * size];
      working_matrix = new Double_t *[fSizeX];
      for (i = 0; i < fSizeX; i++)
         working_matrix[i] = new Double_t[fSizeY];
      break;
   case kTransformCos:
   case kTransformSin:
   case kTransformFourier:
   case kTransformHartley:
   case kTransformFourierWalsh:
   case kTransformFourierHaar:
   case kTransformWalshHaar:
      working_vector = new Double_t[4 * size];
      working_matrix = new Double_t *[fSizeX];
      for (i = 0; i < fSizeX; i++)
         working_matrix[i] = new Double_t[2 * fSizeY];
      break;
   case kTransformCosWalsh:
   case kTransformCosHaar:
   case kTransformSinWalsh:
   case kTransformSinHaar:
      working_vector = new Double_t[8 * size];
      working_matrix = new Double_t *[fSizeX];
      for (i = 0; i < fSizeX; i++)
         working_matrix[i] = new Double_t[2 * fSizeY];
      break;
   }
   switch (fTransformType) {
   case kTransformHaar:
      for (i = 0; i < fSizeX; i++) {
         for (j = 0; j < fSizeY; j++) {
            working_matrix[i][j] = fSource[i][j];
            old_area = old_area + fSource[i][j];
         }
      }
      HaarWalsh2(working_matrix, working_vector, fSizeX, fSizeY,
                  kTransformForward, kTransformHaar);
      break;
   case kTransformWalsh:
      for (i = 0; i < fSizeX; i++) {
         for (j = 0; j < fSizeY; j++) {
            working_matrix[i][j] = fSource[i][j];
            old_area = old_area + fSource[i][j];
         }
      }
      HaarWalsh2(working_matrix, working_vector, fSizeX, fSizeY,
                  kTransformForward, kTransformWalsh);
      break;
   case kTransformCos:
      for (i = 0; i < fSizeX; i++) {
         for (j = 0; j < fSizeY; j++) {
            working_matrix[i][j] = fSource[i][j];
            old_area = old_area + fSource[i][j];
         }
      }
      FourCos2(working_matrix, working_vector, fSizeX, fSizeY,
                kTransformForward, kTransformCos);
      break;
   case kTransformSin:
      for (i = 0; i < fSizeX; i++) {
         for (j = 0; j < fSizeY; j++) {
            working_matrix[i][j] = fSource[i][j];
            old_area = old_area + fSource[i][j];
         }
      }
      FourCos2(working_matrix, working_vector, fSizeX, fSizeY,
                kTransformForward, kTransformSin);
      break;
   case kTransformFourier:
      for (i = 0; i < fSizeX; i++) {
         for (j = 0; j < fSizeY; j++) {
            working_matrix[i][j] = fSource[i][j];
            old_area = old_area + fSource[i][j];
         }
      }
      FourCos2(working_matrix, working_vector, fSizeX, fSizeY,
                kTransformForward, kTransformFourier);
      break;
   case kTransformHartley:
      for (i = 0; i < fSizeX; i++) {
         for (j = 0; j < fSizeY; j++) {
            working_matrix[i][j] = fSource[i][j];
            old_area = old_area + fSource[i][j];
         }
      }
      FourCos2(working_matrix, working_vector, fSizeX, fSizeY,
                kTransformForward, kTransformHartley);
      break;
   case kTransformFourierWalsh:
   case kTransformFourierHaar:
   case kTransformWalshHaar:
   case kTransformCosWalsh:
   case kTransformCosHaar:
   case kTransformSinWalsh:
   case kTransformSinHaar:
      for (i = 0; i < fSizeX; i++) {
         for (j = 0; j < fSizeY; j++) {
            working_matrix[i][j] = fSource[i][j];
            old_area = old_area + fSource[i][j];
         }
      }
      General2(working_matrix, working_vector, fSizeX, fSizeY,
                kTransformForward, fTransformType, fDegree);
      break;
   }
   for (i = 0; i < fSizeX; i++) {
      for (j = 0; j < fSizeY; j++) {
         if (i >= fXmin && i <= fXmax && j >= fYmin && j <= fYmax)
            if (working_matrix) working_matrix[i][j] = fFilterCoeff;
      }
   }
   if (fTransformType == kTransformFourier || fTransformType == kTransformFourierWalsh
        || fTransformType == kTransformFourierHaar) {
      for (i = 0; i < fSizeX; i++) {
         for (j = 0; j < fSizeY; j++) {
            if (i >= fXmin && i <= fXmax && j >= fYmin && j <= fYmax)
               if (working_matrix) working_matrix[i][j + fSizeY] = fFilterCoeff;
         }
      }
   }
   switch (fTransformType) {
   case kTransformHaar:
      HaarWalsh2(working_matrix, working_vector, fSizeX, fSizeY,
                  kTransformInverse, kTransformHaar);
      for (i = 0; i < fSizeX; i++) {
         for (j = 0; j < fSizeY; j++) {
            new_area = new_area + working_matrix[i][j];
         }
      }
      if (new_area != 0) {
         a = old_area / new_area;
         for (i = 0; i < fSizeX; i++) {
            for (j = 0; j < fSizeY; j++) {
               fDest[i][j] = working_matrix[i][j] * a;
            }
         }
      }
      break;
   case kTransformWalsh:
      HaarWalsh2(working_matrix, working_vector, fSizeX, fSizeY,
                  kTransformInverse, kTransformWalsh);
      for (i = 0; i < fSizeX; i++) {
         for (j = 0; j < fSizeY; j++) {
            new_area = new_area + working_matrix[i][j];
         }
      }
      if (new_area != 0) {
         a = old_area / new_area;
         for (i = 0; i < fSizeX; i++) {
            for (j = 0; j < fSizeY; j++) {
               fDest[i][j] = working_matrix[i][j] * a;
            }
         }
      }
      break;
   case kTransformCos:
      FourCos2(working_matrix, working_vector, fSizeX, fSizeY,
                kTransformInverse, kTransformCos);
      for (i = 0; i < fSizeX; i++) {
         for (j = 0; j < fSizeY; j++) {
            new_area = new_area + working_matrix[i][j];
         }
      }
      if (new_area != 0) {
         a = old_area / new_area;
         for (i = 0; i < fSizeX; i++) {
            for (j = 0; j < fSizeY; j++) {
               fDest[i][j] = working_matrix[i][j] * a;
            }
         }
      }
      break;
   case kTransformSin:
      FourCos2(working_matrix, working_vector, fSizeX, fSizeY,
                kTransformInverse, kTransformSin);
      for (i = 0; i < fSizeX; i++) {
         for (j = 0; j < fSizeY; j++) {
            new_area = new_area + working_matrix[i][j];
         }
      }
      if (new_area != 0) {
         a = old_area / new_area;
         for (i = 0; i < fSizeX; i++) {
            for (j = 0; j < fSizeY; j++) {
               fDest[i][j] = working_matrix[i][j] * a;
            }
         }
      }
      break;
   case kTransformFourier:
      FourCos2(working_matrix, working_vector, fSizeX, fSizeY,
                kTransformInverse, kTransformFourier);
      for (i = 0; i < fSizeX; i++) {
         for (j = 0; j < fSizeY; j++) {
            new_area = new_area + working_matrix[i][j];
         }
      }
      if (new_area != 0) {
         a = old_area / new_area;
         for (i = 0; i < fSizeX; i++) {
            for (j = 0; j < fSizeY; j++) {
               fDest[i][j] = working_matrix[i][j] * a;
            }
         }
      }
      break;
   case kTransformHartley:
      FourCos2(working_matrix, working_vector, fSizeX, fSizeY,
                kTransformInverse, kTransformHartley);
      for (i = 0; i < fSizeX; i++) {
         for (j = 0; j < fSizeY; j++) {
            new_area = new_area + working_matrix[i][j];
         }
      }
      if (new_area != 0) {
         a = old_area / new_area;
         for (i = 0; i < fSizeX; i++) {
            for (j = 0; j < fSizeY; j++) {
               fDest[i][j] = working_matrix[i][j] * a;
            }
         }
      }
      break;
   case kTransformFourierWalsh:
   case kTransformFourierHaar:
   case kTransformWalshHaar:
   case kTransformCosWalsh:
   case kTransformCosHaar:
   case kTransformSinWalsh:
   case kTransformSinHaar:
      General2(working_matrix, working_vector, fSizeX, fSizeY,
                kTransformInverse, fTransformType, fDegree);
      for (i = 0; i < fSizeX; i++) {
         for (j = 0; j < fSizeY; j++) {
            new_area = new_area + working_matrix[i][j];
         }
      }
      if (new_area != 0) {
         a = old_area / new_area;
         for (i = 0; i < fSizeX; i++) {
            for (j = 0; j < fSizeY; j++) {
               fDest[i][j] = working_matrix[i][j] * a;
            }
         }
      }
      break;
   }
   for (i = 0; i < fSizeX; i++) {
      if (working_matrix) delete[]working_matrix[i];
   }
   delete[]working_matrix;
   delete[]working_vector;
   return;
}

////////////////////////////////////////////////////////////////////////////////
/// This function transforms the source spectrum. The calling program
///      should fill in input parameters. Then it multiplies transformed
///      coefficients in the given region by the given
///      enhance_coeff and transforms it back
///
/// Function parameters:
///  - fSource-pointer to the matrix of source spectrum, its size should
///    be fSizeX*fSizeY
///  - fDest-pointer to the matrix of destination data, its size should be
///    fSizeX*fSizeY
///
/// ### Example of enhancement
///
/// This function transforms the
/// source spectrum (for details see Transform function). Before the Enhance
/// function is called the class must be created by constructor and the type of the
/// transform as well as some other parameters must be set using a set of setter
/// functions. The Enhance function multiplies transformed coefficients in the given
/// region (fXmin, fXmax, fYmin, fYmax) by the given fEnhancCoeff and transforms it
/// back. Enhanced data are written into dest spectrum.
///
/// ### Example - script Enhance2.c:
///
/// \image html spectrum2transform_enhance_image001.jpg Fig. 1 Original two-dimensional noisy spectrum
/// \image html spectrum2transform_enhance_image002.jpg Fig. 2 Enhanced spectrum of the data from Fig. 1 using Cosine transform (channels in region (0-63)x(0-63) were multiplied by 5)
///
/// #### Script:
///
/// Example to illustrate enhancement (class TSpectrumTransform2). To execute this example, do
///
/// `root > .x Enhance2.C`
///
/// ~~~ {.cpp}
///   void Enhance2() {
///      Int_t i, j;
///      Int_t nbinsx = 256;
///      Int_t nbinsy = 256;
///      Int_t xmin = 0;
///      Int_t xmax = nbinsx;
///      Int_t ymin = 0;
///      Int_t ymax = nbinsy;
///      Double_t ** source = new Double_t *[nbinsx];
///      Double_t ** dest = new Double_t *[nbinsx];
///      for (i=0;i<nbinsx;i++)
///         source[i]= new Double_t[nbinsy];
///      for (i=0;i<nbinsx;i++)
///         dest[i]= new Double_t[nbinsy];
///      TH2F *trans = new TH2F("trans","Background estimation",nbinsx,xmin,xmax,nbinsy,ymin,ymax);
///      TFile *f = new TFile("TSpectrum2.root");
///      trans=(TH2F*)f->Get("back3;1");
///      TCanvas *Tr = new TCanvas("Transform","Illustration of transform function",10,10,1000,700);
///      for (i = 0; i < nbinsx; i++){
///         for (j = 0; j < nbinsy; j++){
///            source[i][j] = trans->GetBinContent(i + 1,j + 1);
///         }
///      }
///      TSpectrumTransform2 *t = new TSpectrumTransform2(256,256);
///      t->SetTransformType(t->kTransformCos,0);
///      t->SetRegion(0,63,0,63);
///      t->SetEnhanceCoeff(5);
///      t->Enhance(source,dest);
///      trans->Draw("SURF");
///   }
/// ~~~

void TSpectrum2Transform::Enhance(const Double_t **fSource, Double_t **fDest)
{
   Int_t i, j;
   Double_t a, old_area = 0, new_area = 0;
   Int_t size;
   Double_t *working_vector = 0, **working_matrix = 0;
   size = (Int_t) TMath::Max(fSizeX, fSizeY);
   switch (fTransformType) {
   case kTransformHaar:
   case kTransformWalsh:
      working_vector = new Double_t[2 * size];
      working_matrix = new Double_t *[fSizeX];
      for (i = 0; i < fSizeX; i++)
         working_matrix[i] = new Double_t[fSizeY];
      break;
   case kTransformCos:
   case kTransformSin:
   case kTransformFourier:
   case kTransformHartley:
   case kTransformFourierWalsh:
   case kTransformFourierHaar:
   case kTransformWalshHaar:
      working_vector = new Double_t[4 * size];
      working_matrix = new Double_t *[fSizeX];
      for (i = 0; i < fSizeX; i++)
         working_matrix[i] = new Double_t[2 * fSizeY];
      break;
   case kTransformCosWalsh:
   case kTransformCosHaar:
   case kTransformSinWalsh:
   case kTransformSinHaar:
      working_vector = new Double_t[8 * size];
      working_matrix = new Double_t *[fSizeX];
      for (i = 0; i < fSizeX; i++)
         working_matrix[i] = new Double_t[2 * fSizeY];
      break;
   }
   switch (fTransformType) {
   case kTransformHaar:
      for (i = 0; i < fSizeX; i++) {
         for (j = 0; j < fSizeY; j++) {
            working_matrix[i][j] = fSource[i][j];
            old_area = old_area + fSource[i][j];
         }
      }
      HaarWalsh2(working_matrix, working_vector, fSizeX, fSizeY,
                  kTransformForward, kTransformHaar);
      break;
   case kTransformWalsh:
      for (i = 0; i < fSizeX; i++) {
         for (j = 0; j < fSizeY; j++) {
            working_matrix[i][j] = fSource[i][j];
            old_area = old_area + fSource[i][j];
         }
      }
      HaarWalsh2(working_matrix, working_vector, fSizeX, fSizeY,
                  kTransformForward, kTransformWalsh);
      break;
   case kTransformCos:
      for (i = 0; i < fSizeX; i++) {
         for (j = 0; j < fSizeY; j++) {
            working_matrix[i][j] = fSource[i][j];
            old_area = old_area + fSource[i][j];
         }
      }
      FourCos2(working_matrix, working_vector, fSizeX, fSizeY,
                kTransformForward, kTransformCos);
      break;
   case kTransformSin:
      for (i = 0; i < fSizeX; i++) {
         for (j = 0; j < fSizeY; j++) {
            working_matrix[i][j] = fSource[i][j];
            old_area = old_area + fSource[i][j];
         }
      }
      FourCos2(working_matrix, working_vector, fSizeX, fSizeY,
                kTransformForward, kTransformSin);
      break;
   case kTransformFourier:
      for (i = 0; i < fSizeX; i++) {
         for (j = 0; j < fSizeY; j++) {
            working_matrix[i][j] = fSource[i][j];
            old_area = old_area + fSource[i][j];
         }
      }
      FourCos2(working_matrix, working_vector, fSizeX, fSizeY,
                kTransformForward, kTransformFourier);
      break;
   case kTransformHartley:
      for (i = 0; i < fSizeX; i++) {
         for (j = 0; j < fSizeY; j++) {
            working_matrix[i][j] = fSource[i][j];
            old_area = old_area + fSource[i][j];
         }
      }
      FourCos2(working_matrix, working_vector, fSizeX, fSizeY,
                kTransformForward, kTransformHartley);
      break;
   case kTransformFourierWalsh:
   case kTransformFourierHaar:
   case kTransformWalshHaar:
   case kTransformCosWalsh:
   case kTransformCosHaar:
   case kTransformSinWalsh:
   case kTransformSinHaar:
      for (i = 0; i < fSizeX; i++) {
         for (j = 0; j < fSizeY; j++) {
            working_matrix[i][j] = fSource[i][j];
            old_area = old_area + fSource[i][j];
         }
      }
      General2(working_matrix, working_vector, fSizeX, fSizeY,
                kTransformForward, fTransformType, fDegree);
      break;
   }
   for (i = 0; i < fSizeX; i++) {
      for (j = 0; j < fSizeY; j++) {
         if (i >= fXmin && i <= fXmax && j >= fYmin && j <= fYmax)
            if (working_matrix) working_matrix[i][j] *= fEnhanceCoeff;
      }
   }
   if (fTransformType == kTransformFourier || fTransformType == kTransformFourierWalsh
        || fTransformType == kTransformFourierHaar) {
      for (i = 0; i < fSizeX; i++) {
         for (j = 0; j < fSizeY; j++) {
            if (i >= fXmin && i <= fXmax && j >= fYmin && j <= fYmax)
               working_matrix[i][j + fSizeY] *= fEnhanceCoeff;
         }
      }
   }
   switch (fTransformType) {
   case kTransformHaar:
      HaarWalsh2(working_matrix, working_vector, fSizeX, fSizeY,
                  kTransformInverse, kTransformHaar);
      for (i = 0; i < fSizeX; i++) {
         for (j = 0; j < fSizeY; j++) {
            new_area = new_area + working_matrix[i][j];
         }
      }
      if (new_area != 0) {
         a = old_area / new_area;
         for (i = 0; i < fSizeX; i++) {
            for (j = 0; j < fSizeY; j++) {
               fDest[i][j] = working_matrix[i][j] * a;
            }
         }
      }
      break;
   case kTransformWalsh:
      HaarWalsh2(working_matrix, working_vector, fSizeX, fSizeY,
                  kTransformInverse, kTransformWalsh);
      for (i = 0; i < fSizeX; i++) {
         for (j = 0; j < fSizeY; j++) {
            new_area = new_area + working_matrix[i][j];
         }
      }
      if (new_area != 0) {
         a = old_area / new_area;
         for (i = 0; i < fSizeX; i++) {
            for (j = 0; j < fSizeY; j++) {
               fDest[i][j] = working_matrix[i][j] * a;
            }
         }
      }
      break;
   case kTransformCos:
      FourCos2(working_matrix, working_vector, fSizeX, fSizeY,
                kTransformInverse, kTransformCos);
      for (i = 0; i < fSizeX; i++) {
         for (j = 0; j < fSizeY; j++) {
            new_area = new_area + working_matrix[i][j];
         }
      }
      if (new_area != 0) {
         a = old_area / new_area;
         for (i = 0; i < fSizeX; i++) {
            for (j = 0; j < fSizeY; j++) {
               fDest[i][j] = working_matrix[i][j] * a;
            }
         }
      }
      break;
   case kTransformSin:
      FourCos2(working_matrix, working_vector, fSizeX, fSizeY,
                kTransformInverse, kTransformSin);
      for (i = 0; i < fSizeX; i++) {
         for (j = 0; j < fSizeY; j++) {
            new_area = new_area + working_matrix[i][j];
         }
      }
      if (new_area != 0) {
         a = old_area / new_area;
         for (i = 0; i < fSizeX; i++) {
            for (j = 0; j < fSizeY; j++) {
               fDest[i][j] = working_matrix[i][j] * a;
            }
         }
      }
      break;
   case kTransformFourier:
      FourCos2(working_matrix, working_vector, fSizeX, fSizeY,
                kTransformInverse, kTransformFourier);
      for (i = 0; i < fSizeX; i++) {
         for (j = 0; j < fSizeY; j++) {
            new_area = new_area + working_matrix[i][j];
         }
      }
      if (new_area != 0) {
         a = old_area / new_area;
         for (i = 0; i < fSizeX; i++) {
            for (j = 0; j < fSizeY; j++) {
               fDest[i][j] = working_matrix[i][j] * a;
            }
         }
      }
      break;
   case kTransformHartley:
      FourCos2(working_matrix, working_vector, fSizeX, fSizeY,
                kTransformInverse, kTransformHartley);
      for (i = 0; i < fSizeX; i++) {
         for (j = 0; j < fSizeY; j++) {
            new_area = new_area + working_matrix[i][j];
         }
      }
      if (new_area != 0) {
         a = old_area / new_area;
         for (i = 0; i < fSizeX; i++) {
            for (j = 0; j < fSizeY; j++) {
               fDest[i][j] = working_matrix[i][j] * a;
            }
         }
      }
      break;
   case kTransformFourierWalsh:
   case kTransformFourierHaar:
   case kTransformWalshHaar:
   case kTransformCosWalsh:
   case kTransformCosHaar:
   case kTransformSinWalsh:
   case kTransformSinHaar:
      General2(working_matrix, working_vector, fSizeX, fSizeY,
                kTransformInverse, fTransformType, fDegree);
      for (i = 0; i < fSizeX; i++) {
         for (j = 0; j < fSizeY; j++) {
            new_area = new_area + working_matrix[i][j];
         }
      }
      if (new_area != 0) {
         a = old_area / new_area;
         for (i = 0; i < fSizeX; i++) {
            for (j = 0; j < fSizeY; j++) {
               fDest[i][j] = working_matrix[i][j] * a;
            }
         }
      }
      break;
   }
   for (i = 0; i < fSizeX; i++) {
      if (working_matrix) delete[]working_matrix[i];
   }
   delete[]working_matrix;
   delete[]working_vector;
   return;
}

////////////////////////////////////////////////////////////////////////////////
/// This function sets the following parameters for transform:
///  - transType - type of transform (Haar, Walsh, Cosine, Sine, Fourier, Hartley, Fourier-Walsh, Fourier-Haar, Walsh-Haar, Cosine-Walsh, Cosine-Haar, Sine-Walsh, Sine-Haar)
///  - degree - degree of mixed transform, applies only for Fourier-Walsh, Fourier-Haar, Walsh-Haar, Cosine-Walsh, Cosine-Haar, Sine-Walsh, Sine-Haar transforms

void TSpectrum2Transform::SetTransformType(Int_t transType, Int_t degree)
{
   Int_t j1, j2, n;
   j1 = 0;
   n = 1;
   for (; n < fSizeX;) {
      j1 += 1;
      n = n * 2;
   }
   j2 = 0;
   n = 1;
   for (; n < fSizeY;) {
      j2 += 1;
      n = n * 2;
   }
   if (transType < kTransformHaar || transType > kTransformSinHaar){
      Error ("TSpectrumTransform","Invalid type of transform");
      return;
   }
   if (transType >= kTransformFourierWalsh && transType <= kTransformSinHaar) {
      if (degree > j1 || degree > j2 || degree < 1){
         Error ("TSpectrumTransform","Invalid degree of mixed transform");
         return;
      }
   }
   fTransformType = transType;
   fDegree = degree;
}

////////////////////////////////////////////////////////////////////////////////
/// This function sets the filtering or enhancement region:
///  - xmin, xmax, ymin, ymax

void TSpectrum2Transform::SetRegion(Int_t xmin, Int_t xmax, Int_t ymin, Int_t ymax)
{
   if(xmin<0 || xmax < xmin || xmax >= fSizeX){
      Error("TSpectrumTransform", "Wrong range");
      return;
   }
   if(ymin<0 || ymax < ymin || ymax >= fSizeY){
      Error("TSpectrumTransform", "Wrong range");
      return;
   }
   fXmin = xmin;
   fXmax = xmax;
   fYmin = ymin;
   fYmax = ymax;
}

////////////////////////////////////////////////////////////////////////////////
/// This function sets the direction of the transform:
///  - direction (forward or inverse)

void TSpectrum2Transform::SetDirection(Int_t direction)
{
   if(direction != kTransformForward && direction != kTransformInverse){
      Error("TSpectrumTransform", "Wrong direction");
      return;
   }
   fDirection = direction;
}

////////////////////////////////////////////////////////////////////////////////
/// This function sets the filter coefficient:
///  - filterCoeff - after the transform the filtered region (xmin, xmax, ymin, ymax) is replaced by this coefficient. Applies only for filtereng operation.

void TSpectrum2Transform::SetFilterCoeff(Double_t filterCoeff)
{
   fFilterCoeff = filterCoeff;
}

////////////////////////////////////////////////////////////////////////////////
/// This function sets the enhancement coefficient:
///  - enhanceCoeff - after the transform the enhanced region (xmin, xmax, ymin, ymax) is multiplied by this coefficient. Applies only for enhancement operation.

void TSpectrum2Transform::SetEnhanceCoeff(Double_t enhanceCoeff)
{
   fEnhanceCoeff = enhanceCoeff;
}

