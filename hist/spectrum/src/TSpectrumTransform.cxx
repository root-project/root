// @(#)root/spectrum:$Id$
// Author: Miroslav Morhac   25/09/06

/** \class TSpectrumTransform
    \ingroup Spectrum
    \brief Advanced 1-dimensional orthogonal transform functions
    \author Miroslav Morhac

 \unmaintained{TSpectrumTransform}


 Class to carry out transforms of 1D spectra, its filtering and
 enhancement. It allows to calculate classic Fourier, Cosine, Sin,
 Hartley, Walsh, Haar transforms as well as mixed transforms (Fourier-
 Walsh, Fourier-Haar, Walsh-Haar, Cosine-Walsh, Cosine-Haar, Sin-Walsh
 and Sin-Haar). All the transforms are fast.

 The algorithms in this class have been published in the following
 references:

  1. C.V. Hampton, B. Lian, Wm. C. McHarris: Fast-Fourier-transform
     spectral enhancement techniques for gamma-ray spectroscopy.NIM A353(1994) 280-284.
  2. Morhac M., Matousek V., New adaptive Cosine-Walsh  transform and
     its application to nuclear data compression, IEEE Transactions on
     Signal Processing 48 (2000) 2693.
  3. Morhac M., Matousek V., Data compression using new fast adaptive
     Cosine-Haar transforms, Digital Signal Processing 8 (1998) 63.
  4. Morhac M., Matousek V.: Multidimensional nuclear data compression
     using fast adaptive Walsh-Haar transform. Acta Physica Slovaca 51 (2001) 307.
 */

#include "TSpectrumTransform.h"
#include "TMath.h"

ClassImp(TSpectrumTransform);

////////////////////////////////////////////////////////////////////////////////
///default constructor

TSpectrumTransform::TSpectrumTransform()
{
   fSize=0;
   fTransformType=kTransformCos;
   fDegree=0;
   fDirection=kTransformForward;
   fXmin=0;
   fXmax=0;
   fFilterCoeff=0;
   fEnhanceCoeff=0.5;
}

////////////////////////////////////////////////////////////////////////////////
/// the constructor creates TSpectrumTransform object. Its size must be > than zero and must be power of 2.
/// It sets default transform type to be Cosine transform. Transform parameters can be changed using setter functions.

TSpectrumTransform::TSpectrumTransform(Int_t size):TNamed("SpectrumTransform", "Miroslav Morhac transformer")
{
   Int_t n;
   if (size <= 0){
      Error ("TSpectrumTransform","Invalid length, must be > than 0");
      return;
   }
   n = 1;
   for (; n < size;) {
      n = n * 2;
   }
   if (n != size){
      Error ("TSpectrumTransform","Invalid length, must be power of 2");
      return;
   }
   fSize=size;
   fTransformType=kTransformCos;
   fDegree=0;
   fDirection=kTransformForward;
   fXmin=size/4;
   fXmax=size-1;
   fFilterCoeff=0;
   fEnhanceCoeff=0.5;
}


////////////////////////////////////////////////////////////////////////////////
/// Destructor

TSpectrumTransform::~TSpectrumTransform()
{
}

////////////////////////////////////////////////////////////////////////////////
///   This function calculates Haar transform of a part of data
///      Function parameters:
///              - working_space-pointer to vector of transformed data
///              - num-length of processed data
///              - direction-forward or inverse transform

void TSpectrumTransform::Haar(Double_t *working_space, int num, int direction)
{
   int i, ii, li, l2, l3, j, jj, jj1, lj, iter, m, jmin, jmax;
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
///      Function parameters:
///              - working_space-pointer to vector of transformed data
///              - num-length of processed data

void TSpectrumTransform::Walsh(Double_t *working_space, int num)
{
   int i, m, nump = 1, mnum, mnum2, mp, ib, mp2, mnum21, iba, iter;
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
///      Function parameters:
///              - working_space-pointer to vector of processed data
///              - num-length of processed data

void TSpectrumTransform::BitReverse(Double_t *working_space, int num)
{
   int ipower[26];
   int i, ib, il, ibd, ip, ifac, i1;
   for (i = 0; i < num; i++) {
      working_space[i + num] = working_space[i];
   }
   for (i = 1; i <= num; i++) {
      ib = i - 1;
      il = 1;
   lab9:ibd = ib / 2;
      ipower[il - 1] = 1;
      if (ib == (ibd * 2))
         ipower[il - 1] = 0;
      if (ibd == 0)
         goto lab10;
      ib = ibd;
      il = il + 1;
      goto lab9;
   lab10:ip = 1;
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
///      Function parameters:
///              - working_space-pointer to vector of transformed data
///              - num-length of processed data
///              - hartley-1 if it is Hartley transform, 0 otherwise
///              - direction-forward or inverse transform

void TSpectrumTransform::Fourier(Double_t *working_space, int num, int hartley,
                          int direction, int zt_clear)
{
   int nxp2, nxp, i, j, k, m, iter, mxp, j1, j2, n1, n2, it;
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
///      Function parameters:
///              - working_space-pointer to vector of processed data
///              - shift-shift of position of processing
///              - start-initial position of processed data
///              - num-length of processed data

void TSpectrumTransform::BitReverseHaar(Double_t *working_space, int shift, int num,
                                 int start)
{
   int ipower[26];
   int i, ib, il, ibd, ip, ifac, i1;
   for (i = 0; i < num; i++) {
      working_space[i + shift + start] = working_space[i + start];
      working_space[i + shift + start + 2 * shift] =
          working_space[i + start + 2 * shift];
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
      working_space[ip - 1 + start] =
      working_space[i - 1 + shift + start];
      working_space[ip - 1 + start + 2 * shift] =
      working_space[i - 1 + shift + start + 2 * shift];
   }
   return;
}

////////////////////////////////////////////////////////////////////////////////
///   This function calculates generalized (mixed) transforms of different degrees
///      Function parameters:
///              - working_space-pointer to vector of transformed data
///              - zt_clear-flag to clear imaginary data before staring
///              - num-length of processed data
///              - degree-degree of transform (see manual)
///              - type-type of mixed transform (see manual)

int TSpectrumTransform::GeneralExe(Double_t *working_space, int zt_clear, int num,
                            int degree, int type)
{
   int i, j, k, m, nump, mnum, mnum2, mp, ib, mp2, mnum21, iba, iter,
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
///   This function calculates inverse generalized (mixed) transforms
///      Function parameters:
///              - working_space-pointer to vector of transformed data
///              - num-length of processed data
///              - degree-degree of transform (see manual)
///              - type-type of mixed transform (see manual)

int TSpectrumTransform::GeneralInv(Double_t *working_space, int num, int degree,
                            int type)
{
   int i, j, k, m, nump =
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
/// This function transforms the source spectrum. The calling program
/// should fill in input parameters.
/// Transformed data are written into dest spectrum.
///
/// Function parameters:
///  - source-pointer to the vector of source spectrum, its length should
///    be size except for inverse FOURIER, FOUR-WALSH, FOUR-HAAR
///    transform. These need 2*size length to supply real and
///    imaginary coefficients.
///  - destVector-pointer to the vector of dest data, its length should be
///    size except for direct FOURIER, FOUR-WALSH, FOUR-HAAR. These
///    need 2*size length to store real and imaginary coefficients
///
/// ### Transform methods
///
///  Goal: to analyse experimental data using orthogonal transforms
///
///  - orthogonal transforms can be successfully used for the processing of
///    nuclear spectra (not only)
///
///  - they can be used to remove high frequency noise, to increase
///    signal-to-background ratio as well as to enhance low intensity components [1],
///    to carry out e.g. Fourier analysis etc.
///
///  - we have implemented the function for the calculation of the commonly
///    used orthogonal transforms as well as functions for the filtration and
///    enhancement of experimental data
///
/// #### References:
///
/// [1] C.V. Hampton, B. Lian, Wm. C.
/// McHarris: Fast-Fourier-transform spectral enhancement techniques for gamma-ray
/// spectroscopy. NIM A353 (1994) 280-284.
///
/// [2] Morhac; M., Matouoek V.,
/// New adaptive Cosine-Walsh transform and its application to nuclear data
/// compression, IEEE Transactions on Signal Processing 48 (2000) 2693.
///
/// [3] Morhac; M., Matouoek V.,
/// Data compression using new fast adaptive Cosine-Haar transforms, Digital Signal
/// Processing 8 (1998) 63.
///
/// [4] Morhac; M., Matouoek V.:
/// Multidimensional nuclear data compression using fast adaptive Walsh-Haar
/// transform. Acta Physica Slovaca 51 (2001) 307.
///
/// ### Example - script Transform.c:
///
/// \image html spectrumtransform_transform_image002.jpg Fig. 1 Original gamma-ray spectrum
/// \image html spectrumtransform_transform_image003.jpg Fig. 2 Transformed spectrum from Fig. 1 using Cosine transform
///
/// #### Script:
/// Example to illustrate Transform function (class TSpectrumTransform).
/// To execute this example, do:
///
/// `root > .x Transform.C`
///
/// ~~~ {.cpp}
///   #include <TSpectrum>
///   #include <TSpectrumTransform>
///   void Transform() {
///      Int_t i;
///      Double_t nbins = 4096;
///      Double_t xmin = 0;
///      Double_t xmax = (Double_t)nbins;
///      Double_t * source = new Double_t[nbins];
///      Double_t * dest = new Double_t[nbins];
///      TH1F *h = new TH1F("h","Transformed spectrum using Cosine transform",nbins,xmin,xmax);
///      TFile *f = new TFile("spectra/TSpectrum.root");
///      h=(TH1F*) f->Get("transform1;1");
///      for (i = 0; i < nbins; i++) source[i]=h->GetBinContent(i + 1);
///      TCanvas *Transform1 = gROOT->GetListOfCanvases()->FindObject("Transform1");
///      if (!Transform1) Transform1 = new TCanvas("Transform","Transform1",10,10,1000,700);
///      TSpectrum *s = new TSpectrum();
///      TSpectrumTransform *t = new TSpectrumTransform(4096);
///      t->SetTransformType(t->kTransformCos,0);
///      t->SetDirection(t->kTransformForward);
///      t->Transform(source,dest);
///      for (i = 0; i < nbins; i++) h->SetBinContent(i + 1,dest[i]);
///      h->SetLineColor(kRed);
///      h->Draw("L");
///   }
/// ~~~

void TSpectrumTransform::Transform(const Double_t *source, Double_t *destVector)
{
   int i, j=0, k = 1, m, l;
   Double_t val;
   Double_t a, b, pi = 3.14159265358979323846;
   Double_t *working_space = 0;
   if (fTransformType >= kTransformFourierWalsh && fTransformType <= kTransformSinHaar) {
      if (fTransformType >= kTransformCosWalsh)
         fDegree += 1;
      k = (Int_t) TMath::Power(2, fDegree);
      j = fSize / k;
   }
   switch (fTransformType) {
   case kTransformHaar:
   case kTransformWalsh:
      working_space = new Double_t[2 * fSize];
      break;
   case kTransformCos:
   case kTransformSin:
   case kTransformFourier:
   case kTransformHartley:
   case kTransformFourierWalsh:
   case kTransformFourierHaar:
   case kTransformWalshHaar:
      working_space = new Double_t[4 * fSize];
      break;
   case kTransformCosWalsh:
   case kTransformCosHaar:
   case kTransformSinWalsh:
   case kTransformSinHaar:
      working_space = new Double_t[8 * fSize];
      break;
   }
   if (fDirection == kTransformForward) {
      switch (fTransformType) {
      case kTransformHaar:
         for (i = 0; i < fSize; i++) {
            working_space[i] = source[i];
         }
         Haar(working_space, fSize, fDirection);
         for (i = 0; i < fSize; i++) {
            destVector[i] = working_space[i];
         }
         break;
      case kTransformWalsh:
         for (i = 0; i < fSize; i++) {
            working_space[i] = source[i];
         }
         Walsh(working_space, fSize);
         BitReverse(working_space, fSize);
         for (i = 0; i < fSize; i++) {
            destVector[i] = working_space[i];
         }
         break;
      case kTransformCos:
         fSize = 2 * fSize;
         for (i = 1; i <= (fSize / 2); i++) {
            val = source[i - 1];
            working_space[i - 1] = val;
            working_space[fSize - i] = val;
         }
         Fourier(working_space, fSize, 0, kTransformForward, 0);
         for (i = 0; i < fSize / 2; i++) {
            a = pi * (Double_t) i / (Double_t) fSize;
            a = TMath::Cos(a);
            b = working_space[i];
            a = b / a;
            working_space[i] = a;
            working_space[i + fSize] = 0;
         } working_space[0] = working_space[0] / TMath::Sqrt(2.0);
         for (i = 0; i < fSize / 2; i++) {
            destVector[i] = working_space[i];
         }
         break;
      case kTransformSin:
         fSize = 2 * fSize;
         for (i = 1; i <= (fSize / 2); i++) {
            val = source[i - 1];
            working_space[i - 1] = val;
            working_space[fSize - i] = -val;
         }
         Fourier(working_space, fSize, 0, kTransformForward, 0);
         for (i = 0; i < fSize / 2; i++) {
            a = pi * (Double_t) i / (Double_t) fSize;
            a = TMath::Sin(a);
            b = working_space[i];
            if (a != 0)
               a = b / a;
            working_space[i - 1] = a;
            working_space[i + fSize] = 0;
         }
         working_space[fSize / 2 - 1] =
             working_space[fSize / 2] / TMath::Sqrt(2.0);
         for (i = 0; i < fSize / 2; i++) {
            destVector[i] = working_space[i];
         }
         break;
      case kTransformFourier:
         for (i = 0; i < fSize; i++) {
            working_space[i] = source[i];
         }
         Fourier(working_space, fSize, 0, kTransformForward, 0);
         for (i = 0; i < 2 * fSize; i++) {
            destVector[i] = working_space[i];
         }
         break;
      case kTransformHartley:
         for (i = 0; i < fSize; i++) {
            working_space[i] = source[i];
         }
         Fourier(working_space, fSize, 1, kTransformForward, 0);
         for (i = 0; i < fSize; i++) {
            destVector[i] = working_space[i];
         }
         break;
      case kTransformFourierWalsh:
      case kTransformFourierHaar:
      case kTransformWalshHaar:
      case kTransformCosWalsh:
      case kTransformCosHaar:
      case kTransformSinWalsh:
      case kTransformSinHaar:
         for (i = 0; i < fSize; i++) {
            val = source[i];
            if (fTransformType == kTransformCosWalsh
                 || fTransformType == kTransformCosHaar) {
               j = (Int_t) TMath::Power(2, fDegree) / 2;
               k = i / j;
               k = 2 * k * j;
               working_space[k + i % j] = val;
               working_space[k + 2 * j - 1 - i % j] = val;
            }

            else if (fTransformType == kTransformSinWalsh
                     || fTransformType == kTransformSinHaar) {
               j = (Int_t) TMath::Power(2, fDegree) / 2;
               k = i / j;
               k = 2 * k * j;
               working_space[k + i % j] = val;
               working_space[k + 2 * j - 1 - i % j] = -val;
            }

            else
               working_space[i] = val;
         }
         if (fTransformType == kTransformFourierWalsh
              || fTransformType == kTransformFourierHaar
              || fTransformType == kTransformWalshHaar) {
            for (i = 0; i < j; i++)
               BitReverseHaar(working_space, fSize, k, i * k);
            GeneralExe(working_space, 0, fSize, fDegree, fTransformType);
         }

         else if (fTransformType == kTransformCosWalsh
                  || fTransformType == kTransformCosHaar) {
            m = (Int_t) TMath::Power(2, fDegree);
            l = 2 * fSize / m;
            for (i = 0; i < l; i++)
               BitReverseHaar(working_space, 2 * fSize, m, i * m);
            GeneralExe(working_space, 0, 2 * fSize, fDegree, fTransformType);
            for (i = 0; i < fSize; i++) {
               k = i / j;
               k = 2 * k * j;
               a = pi * (Double_t) (i % j) / (Double_t) (2 * j);
               a = TMath::Cos(a);
               b = working_space[k + i % j];
               if (i % j == 0)
                  a = b / TMath::Sqrt(2.0);

               else
                  a = b / a;
               working_space[i] = a;
               working_space[i + 2 * fSize] = 0;
            }
         }

         else if (fTransformType == kTransformSinWalsh
                  || fTransformType == kTransformSinHaar) {
            m = (Int_t) TMath::Power(2, fDegree);
            l = 2 * fSize / m;
            for (i = 0; i < l; i++)
               BitReverseHaar(working_space, 2 * fSize, m, i * m);
            GeneralExe(working_space, 0, 2 * fSize, fDegree, fTransformType);
            for (i = 0; i < fSize; i++) {
               k = i / j;
               k = 2 * k * j;
               a = pi * (Double_t) (i % j) / (Double_t) (2 * j);
               a = TMath::Cos(a);
               b = working_space[j + k + i % j];
               if (i % j == 0)
                  a = b / TMath::Sqrt(2.0);

               else
                  a = b / a;
               working_space[j + k / 2 - i % j - 1] = a;
               working_space[i + 2 * fSize] = 0;
            }
         }
         if (fTransformType > kTransformWalshHaar)
            k = (Int_t) TMath::Power(2, fDegree - 1);

         else
            k = (Int_t) TMath::Power(2, fDegree);
         j = fSize / k;
         for (i = 0, l = 0; i < fSize; i++, l = (l + k) % fSize) {
            working_space[fSize + i] = working_space[l + i / j];
            working_space[fSize + i + 2 * fSize] =
                working_space[l + i / j + 2 * fSize];
         }
         for (i = 0; i < fSize; i++) {
            working_space[i] = working_space[fSize + i];
            working_space[i + 2 * fSize] =
                working_space[fSize + i + 2 * fSize];
         }
         for (i = 0; i < fSize; i++) {
            destVector[i] = working_space[i];
         }
         if (fTransformType == kTransformFourierWalsh
              || fTransformType == kTransformFourierHaar) {
            for (i = 0; i < fSize; i++) {
               destVector[fSize + i] = working_space[i + 2 * fSize];
            }
         }
         break;
      }
   }

   else if (fDirection == kTransformInverse) {
      switch (fTransformType) {
      case kTransformHaar:
         for (i = 0; i < fSize; i++) {
            working_space[i] = source[i];
         }
         Haar(working_space, fSize, fDirection);
         for (i = 0; i < fSize; i++) {
            destVector[i] = working_space[i];
         }
         break;
      case kTransformWalsh:
         for (i = 0; i < fSize; i++) {
            working_space[i] = source[i];
         }
         BitReverse(working_space, fSize);
         Walsh(working_space, fSize);
         for (i = 0; i < fSize; i++) {
            destVector[i] = working_space[i];
         }
         break;
      case kTransformCos:
         for (i = 0; i < fSize; i++) {
            working_space[i] = source[i];
         }
         fSize = 2 * fSize;
         working_space[0] = working_space[0] * TMath::Sqrt(2.0);
         for (i = 0; i < fSize / 2; i++) {
            a = pi * (Double_t) i / (Double_t) fSize;
            b = TMath::Sin(a);
            a = TMath::Cos(a);
            working_space[i + fSize] = (Double_t) working_space[i] * b;
            working_space[i] = (Double_t) working_space[i] * a;
         } for (i = 2; i <= (fSize / 2); i++) {
            working_space[fSize - i + 1] = working_space[i - 1];
            working_space[fSize - i + 1 + fSize] =
                -working_space[i - 1 + fSize];
         }
         working_space[fSize / 2] = 0;
         working_space[fSize / 2 + fSize] = 0;
         Fourier(working_space, fSize, 0, kTransformInverse, 1);
         for (i = 0; i < fSize / 2; i++) {
            destVector[i] = working_space[i];
         }
         break;
      case kTransformSin:
         for (i = 0; i < fSize; i++) {
            working_space[i] = source[i];
         }
         fSize = 2 * fSize;
         working_space[fSize / 2] =
             working_space[fSize / 2 - 1] * TMath::Sqrt(2.0);
         for (i = fSize / 2 - 1; i > 0; i--) {
            a = pi * (Double_t) i / (Double_t) fSize;
            working_space[i + fSize] =
                -(Double_t) working_space[i - 1] * TMath::Cos(a);
            working_space[i] =
                (Double_t) working_space[i - 1] * TMath::Sin(a);
         } for (i = 2; i <= (fSize / 2); i++) {
            working_space[fSize - i + 1] = working_space[i - 1];
            working_space[fSize - i + 1 + fSize] =
                -working_space[i - 1 + fSize];
         }
         working_space[0] = 0;
         working_space[fSize] = 0;
         working_space[fSize / 2 + fSize] = 0;
         Fourier(working_space, fSize, 0, kTransformInverse, 0);
         for (i = 0; i < fSize / 2; i++) {
            destVector[i] = working_space[i];
         }
         break;
      case kTransformFourier:
         for (i = 0; i < 2 * fSize; i++) {
            working_space[i] = source[i];
         }
         Fourier(working_space, fSize, 0, kTransformInverse, 0);
         for (i = 0; i < fSize; i++) {
            destVector[i] = working_space[i];
         }
         break;
      case kTransformHartley:
         for (i = 0; i < fSize; i++) {
            working_space[i] = source[i];
         }
         Fourier(working_space, fSize, 1, kTransformInverse, 0);
         for (i = 0; i < fSize; i++) {
            destVector[i] = working_space[i];
         }
         break;
      case kTransformFourierWalsh:
      case kTransformFourierHaar:
      case kTransformWalshHaar:
      case kTransformCosWalsh:
      case kTransformCosHaar:
      case kTransformSinWalsh:
      case kTransformSinHaar:
         for (i = 0; i < fSize; i++) {
            working_space[i] = source[i];
         }
         if (fTransformType == kTransformFourierWalsh
              || fTransformType == kTransformFourierHaar) {
            for (i = 0; i < fSize; i++) {
               working_space[i + 2 * fSize] = source[fSize + i];
            }
         }
         if (fTransformType > kTransformWalshHaar)
            k = (Int_t) TMath::Power(2, fDegree - 1);

         else
            k = (Int_t) TMath::Power(2, fDegree);
         j = fSize / k;
         for (i = 0, l = 0; i < fSize; i++, l = (l + k) % fSize) {
            working_space[fSize + l + i / j] = working_space[i];
            working_space[fSize + l + i / j + 2 * fSize] =
                working_space[i + 2 * fSize];
         }
         for (i = 0; i < fSize; i++) {
            working_space[i] = working_space[fSize + i];
            working_space[i + 2 * fSize] =
                working_space[fSize + i + 2 * fSize];
         }
         if (fTransformType == kTransformFourierWalsh
              || fTransformType == kTransformFourierHaar
              || fTransformType == kTransformWalshHaar) {
            GeneralInv(working_space, fSize, fDegree, fTransformType);
            for (i = 0; i < j; i++)
               BitReverseHaar(working_space, fSize, k, i * k);
         }

         else if (fTransformType == kTransformCosWalsh
                  || fTransformType == kTransformCosHaar) {
            j = (Int_t) TMath::Power(2, fDegree) / 2;
            m = (Int_t) TMath::Power(2, fDegree);
            l = 2 * fSize / m;
            for (i = 0; i < fSize; i++) {
               k = i / j;
               k = 2 * k * j;
               a = pi * (Double_t) (i % j) / (Double_t) (2 * j);
               if (i % j == 0) {
                  working_space[2 * fSize + k + i % j] =
                      working_space[i] * TMath::Sqrt(2.0);
                  working_space[4 * fSize + 2 * fSize + k + i % j] = 0;
               }

               else {
                  b = TMath::Sin(a);
                  a = TMath::Cos(a);
                  working_space[4 * fSize + 2 * fSize + k + i % j] =
                      -(Double_t) working_space[i] * b;
                  working_space[2 * fSize + k + i % j] =
                      (Double_t) working_space[i] * a;
            } } for (i = 0; i < fSize; i++) {
               k = i / j;
               k = 2 * k * j;
               if (i % j == 0) {
                  working_space[2 * fSize + k + j] = 0;
                  working_space[4 * fSize + 2 * fSize + k + j] = 0;
               }

               else {
                  working_space[2 * fSize + k + 2 * j - i % j] =
                      working_space[2 * fSize + k + i % j];
                  working_space[4 * fSize + 2 * fSize + k + 2 * j - i % j] =
                      -working_space[4 * fSize + 2 * fSize + k + i % j];
               }
            }
            for (i = 0; i < 2 * fSize; i++) {
               working_space[i] = working_space[2 * fSize + i];
               working_space[4 * fSize + i] =
                   working_space[4 * fSize + 2 * fSize + i];
            }
            GeneralInv(working_space, 2 * fSize, fDegree, fTransformType);
            m = (Int_t) TMath::Power(2, fDegree);
            l = 2 * fSize / m;
            for (i = 0; i < l; i++)
               BitReverseHaar(working_space, 2 * fSize, m, i * m);
         }

         else if (fTransformType == kTransformSinWalsh
                  || fTransformType == kTransformSinHaar) {
            j = (Int_t) TMath::Power(2, fDegree) / 2;
            m = (Int_t) TMath::Power(2, fDegree);
            l = 2 * fSize / m;
            for (i = 0; i < fSize; i++) {
               k = i / j;
               k = 2 * k * j;
               a = pi * (Double_t) (i % j) / (Double_t) (2 * j);
               if (i % j == 0) {
                  working_space[2 * fSize + k + j + i % j] =
                      working_space[j + k / 2 - i % j -
                                    1] * TMath::Sqrt(2.0);
                  working_space[4 * fSize + 2 * fSize + k + j + i % j] = 0;
               }

               else {
                  b = TMath::Sin(a);
                  a = TMath::Cos(a);
                  working_space[4 * fSize + 2 * fSize + k + j + i % j] =
                      -(Double_t) working_space[j + k / 2 - i % j - 1] * b;
                  working_space[2 * fSize + k + j + i % j] =
                      (Double_t) working_space[j + k / 2 - i % j - 1] * a;
            } } for (i = 0; i < fSize; i++) {
               k = i / j;
               k = 2 * k * j;
               if (i % j == 0) {
                  working_space[2 * fSize + k] = 0;
                  working_space[4 * fSize + 2 * fSize + k] = 0;
               }

               else {
                  working_space[2 * fSize + k + i % j] =
                      working_space[2 * fSize + k + 2 * j - i % j];
                  working_space[4 * fSize + 2 * fSize + k + i % j] =
                      -working_space[4 * fSize + 2 * fSize + k + 2 * j -
                                     i % j];
               }
            }
            for (i = 0; i < 2 * fSize; i++) {
               working_space[i] = working_space[2 * fSize + i];
               working_space[4 * fSize + i] =
                   working_space[4 * fSize + 2 * fSize + i];
            }
            GeneralInv(working_space, 2 * fSize, fDegree, fTransformType);
            for (i = 0; i < l; i++)
               BitReverseHaar(working_space, 2 * fSize, m, i * m);
         }
         for (i = 0; i < fSize; i++) {
            if (fTransformType >= kTransformCosWalsh) {
               k = i / j;
               k = 2 * k * j;
               val = working_space[k + i % j];
            }

            else
               val = working_space[i];
            destVector[i] = val;
         }
         break;
      }
   }
   delete[]working_space;
   return;
}

////////////////////////////////////////////////////////////////////////////////
/// This function transforms the source spectrum. The calling program
/// should fill in input parameters. Then it sets transformed
/// coefficients in the given region (fXmin, fXmax) to the given
/// fFilterCoeff and transforms it back.
/// Filtered data are written into dest spectrum.
///
/// Function parameters:
///  - source-pointer to the vector of source spectrum, its length should
///    be size except for inverse FOURIER, FOUR-WALSH, FOUR-HAAR
///    transform. These need 2*size length to supply real and
///    imaginary coefficients.
///  - destVector-pointer to the vector of dest data, its length should be
///    size except for direct FOURIER, FOUR-WALSH, FOUR-HAAR. These
///    need 2*size length to store real and imaginary coefficients
///
/// ### Example - script Filter.c:
///
/// \image html spectrumtransform_filter_image001.jpg Fig. 1 Original spectrum (black line) and filtered spectrum (red line) using Cosine transform and zonal filtration (channels 2048-4095 were set to 0)
///
/// #### Script:
///
/// Example to illustrate FilterZonal function (class TSpectrumTransform).
/// To execute this example, do:
///
/// `root > .x Filter.C`
///
/// ~~~ {.cpp}
///   #include <TSpectrum>
///   #include <TSpectrumTransform>
///   void Filter() {
///      Int_t i;
///      Double_t nbins = 4096;
///      Double_t xmin = 0;
///      Double_t xmax = (Double_t)nbins;
///      Double_t * source = new Double_t[nbins];
///      Double_t * dest = new Double_t[nbins];
///      TH1F *h = new TH1F("h","Zonal filtering using Cosine transform",nbins,xmin,xmax);
///      TH1F *d = new TH1F("d","",nbins,xmin,xmax);
///      TFile *f = new TFile("spectra/TSpectrum.root");
///      h=(TH1F*) f->Get("transform1;1");
///      for (i = 0; i < nbins; i++) source[i]=h->GetBinContent(i + 1);
///      TCanvas *Transform1 = gROOT->GetListOfCanvases()->FindObject("Transform1");
///      if (!Transform1) Transform1 = new TCanvas("Transform","Transform1",10,10,1000,700);
///      h->SetAxisRange(700,1024);
///      h->Draw("L");
///      TSpectrum *s = new TSpectrum();
///      TSpectrumTransform *t = new TSpectrumTransform(4096);
///      t->SetTransformType(t->kTransformCos,0);
///      t->SetRegion(2048, 4095);
///      t->FilterZonal(source,dest);
///      for (i = 0; i < nbins; i++) d->SetBinContent(i + 1,dest[i]);
///      d->SetLineColor(kRed);
///      d->Draw("SAME L");
///   }
/// ~~~

void TSpectrumTransform::FilterZonal(const Double_t *source, Double_t *destVector)
{
   int i, j=0, k = 1, m, l;
   Double_t val;
   Double_t *working_space = 0;
   Double_t a, b, pi = 3.14159265358979323846, old_area, new_area;
   if (fTransformType >= kTransformFourierWalsh && fTransformType <= kTransformSinHaar) {
      if (fTransformType >= kTransformCosWalsh)
         fDegree += 1;
      k = (Int_t) TMath::Power(2, fDegree);
      j = fSize / k;
   }
   switch (fTransformType) {
   case kTransformHaar:
   case kTransformWalsh:
      working_space = new Double_t[2 * fSize];
      break;
   case kTransformCos:
   case kTransformSin:
   case kTransformFourier:
   case kTransformHartley:
   case kTransformFourierWalsh:
   case kTransformFourierHaar:
   case kTransformWalshHaar:
      working_space = new Double_t[4 * fSize];
      break;
   case kTransformCosWalsh:
   case kTransformCosHaar:
   case kTransformSinWalsh:
   case kTransformSinHaar:
      working_space = new Double_t[8 * fSize];
      break;
   }
   switch (fTransformType) {
   case kTransformHaar:
      for (i = 0; i < fSize; i++) {
         working_space[i] = source[i];
      }
      Haar(working_space, fSize, kTransformForward);
      break;
   case kTransformWalsh:
      for (i = 0; i < fSize; i++) {
         working_space[i] = source[i];
      }
      Walsh(working_space, fSize);
      BitReverse(working_space, fSize);
      break;
   case kTransformCos:
      fSize = 2 * fSize;
      for (i = 1; i <= (fSize / 2); i++) {
         val = source[i - 1];
         working_space[i - 1] = val;
         working_space[fSize - i] = val;
      }
      Fourier(working_space, fSize, 0, kTransformForward, 0);
      for (i = 0; i < fSize / 2; i++) {
         a = pi * (Double_t) i / (Double_t) fSize;
         a = TMath::Cos(a);
         b = working_space[i];
         a = b / a;
         working_space[i] = a;
         working_space[i + fSize] = 0;
      } working_space[0] = working_space[0] / TMath::Sqrt(2.0);
      fSize = fSize / 2;
      break;
   case kTransformSin:
      fSize = 2 * fSize;
      for (i = 1; i <= (fSize / 2); i++) {
         val = source[i - 1];
         working_space[i - 1] = val;
         working_space[fSize - i] = -val;
      }
      Fourier(working_space, fSize, 0, kTransformForward, 0);
      for (i = 0; i < fSize / 2; i++) {
         a = pi * (Double_t) i / (Double_t) fSize;
         a = TMath::Sin(a);
         b = working_space[i];
         if (a != 0)
            a = b / a;
         working_space[i - 1] = a;
         working_space[i + fSize] = 0;
      }
      working_space[fSize / 2 - 1] =
          working_space[fSize / 2] / TMath::Sqrt(2.0);
      fSize = fSize / 2;
      break;
   case kTransformFourier:
      for (i = 0; i < fSize; i++) {
         working_space[i] = source[i];
      }
      Fourier(working_space, fSize, 0, kTransformForward, 0);
      break;
   case kTransformHartley:
      for (i = 0; i < fSize; i++) {
         working_space[i] = source[i];
      }
      Fourier(working_space, fSize, 1, kTransformForward, 0);
      break;
   case kTransformFourierWalsh:
   case kTransformFourierHaar:
   case kTransformWalshHaar:
   case kTransformCosWalsh:
   case kTransformCosHaar:
   case kTransformSinWalsh:
   case kTransformSinHaar:
      for (i = 0; i < fSize; i++) {
         val = source[i];
         if (fTransformType == kTransformCosWalsh || fTransformType == kTransformCosHaar) {
            j = (Int_t) TMath::Power(2, fDegree) / 2;
            k = i / j;
            k = 2 * k * j;
            working_space[k + i % j] = val;
            working_space[k + 2 * j - 1 - i % j] = val;
         }

         else if (fTransformType == kTransformSinWalsh
                  || fTransformType == kTransformSinHaar) {
            j = (Int_t) TMath::Power(2, fDegree) / 2;
            k = i / j;
            k = 2 * k * j;
            working_space[k + i % j] = val;
            working_space[k + 2 * j - 1 - i % j] = -val;
         }

         else
            working_space[i] = val;
      }
      if (fTransformType == kTransformFourierWalsh
           || fTransformType == kTransformFourierHaar
           || fTransformType == kTransformWalshHaar) {
         for (i = 0; i < j; i++)
            BitReverseHaar(working_space, fSize, k, i * k);
         GeneralExe(working_space, 0, fSize, fDegree, fTransformType);
      }

      else if (fTransformType == kTransformCosWalsh || fTransformType == kTransformCosHaar) {
         m = (Int_t) TMath::Power(2, fDegree);
         l = 2 * fSize / m;
         for (i = 0; i < l; i++)
            BitReverseHaar(working_space, 2 * fSize, m, i * m);
         GeneralExe(working_space, 0, 2 * fSize, fDegree, fTransformType);
         for (i = 0; i < fSize; i++) {
            k = i / j;
            k = 2 * k * j;
            a = pi * (Double_t) (i % j) / (Double_t) (2 * j);
            a = TMath::Cos(a);
            b = working_space[k + i % j];
            if (i % j == 0)
               a = b / TMath::Sqrt(2.0);

            else
               a = b / a;
            working_space[i] = a;
            working_space[i + 2 * fSize] = 0;
         }
      }

      else if (fTransformType == kTransformSinWalsh || fTransformType == kTransformSinHaar) {
         m = (Int_t) TMath::Power(2, fDegree);
         l = 2 * fSize / m;
         for (i = 0; i < l; i++)
            BitReverseHaar(working_space, 2 * fSize, m, i * m);
         GeneralExe(working_space, 0, 2 * fSize, fDegree, fTransformType);
         for (i = 0; i < fSize; i++) {
            k = i / j;
            k = 2 * k * j;
            a = pi * (Double_t) (i % j) / (Double_t) (2 * j);
            a = TMath::Cos(a);
            b = working_space[j + k + i % j];
            if (i % j == 0)
               a = b / TMath::Sqrt(2.0);

            else
               a = b / a;
            working_space[j + k / 2 - i % j - 1] = a;
            working_space[i + 2 * fSize] = 0;
         }
      }
      if (fTransformType > kTransformWalshHaar)
         k = (Int_t) TMath::Power(2, fDegree - 1);

      else
         k = (Int_t) TMath::Power(2, fDegree);
      j = fSize / k;
      for (i = 0, l = 0; i < fSize; i++, l = (l + k) % fSize) {
         working_space[fSize + i] = working_space[l + i / j];
         working_space[fSize + i + 2 * fSize] =
             working_space[l + i / j + 2 * fSize];
      }
      for (i = 0; i < fSize; i++) {
         working_space[i] = working_space[fSize + i];
         working_space[i + 2 * fSize] = working_space[fSize + i + 2 * fSize];
      }
      break;
   }
   if (!working_space) return;
   for (i = 0, old_area = 0; i < fSize; i++) {
      old_area += working_space[i];
   }
   for (i = 0, new_area = 0; i < fSize; i++) {
      if (i >= fXmin && i <= fXmax)
         working_space[i] = fFilterCoeff;
      new_area += working_space[i];
   }
   if (new_area != 0) {
      a = old_area / new_area;
      for (i = 0; i < fSize; i++) {
         working_space[i] *= a;
      }
   }
   if (fTransformType == kTransformFourier) {
      for (i = 0, old_area = 0; i < fSize; i++) {
         old_area += working_space[i + fSize];
      }
      for (i = 0, new_area = 0; i < fSize; i++) {
         if (i >= fXmin && i <= fXmax)
            working_space[i + fSize] = fFilterCoeff;
         new_area += working_space[i + fSize];
      }
      if (new_area != 0) {
         a = old_area / new_area;
         for (i = 0; i < fSize; i++) {
            working_space[i + fSize] *= a;
         }
      }
   }

   else if (fTransformType == kTransformFourierWalsh
            || fTransformType == kTransformFourierHaar) {
      for (i = 0, old_area = 0; i < fSize; i++) {
         old_area += working_space[i + 2 * fSize];
      }
      for (i = 0, new_area = 0; i < fSize; i++) {
         if (i >= fXmin && i <= fXmax)
            working_space[i + 2 * fSize] = fFilterCoeff;
         new_area += working_space[i + 2 * fSize];
      }
      if (new_area != 0) {
         a = old_area / new_area;
         for (i = 0; i < fSize; i++) {
            working_space[i + 2 * fSize] *= a;
         }
      }
   }
   switch (fTransformType) {
   case kTransformHaar:
      Haar(working_space, fSize, kTransformInverse);
      for (i = 0; i < fSize; i++) {
         destVector[i] = working_space[i];
      }
      break;
   case kTransformWalsh:
      BitReverse(working_space, fSize);
      Walsh(working_space, fSize);
      for (i = 0; i < fSize; i++) {
         destVector[i] = working_space[i];
      }
      break;
   case kTransformCos:
      fSize = 2 * fSize;
      working_space[0] = working_space[0] * TMath::Sqrt(2.0);
      for (i = 0; i < fSize / 2; i++) {
         a = pi * (Double_t) i / (Double_t) fSize;
         b = TMath::Sin(a);
         a = TMath::Cos(a);
         working_space[i + fSize] = (Double_t) working_space[i] * b;
         working_space[i] = (Double_t) working_space[i] * a;
      } for (i = 2; i <= (fSize / 2); i++) {
         working_space[fSize - i + 1] = working_space[i - 1];
         working_space[fSize - i + 1 + fSize] =
             -working_space[i - 1 + fSize];
      }
      working_space[fSize / 2] = 0;
      working_space[fSize / 2 + fSize] = 0;
      Fourier(working_space, fSize, 0, kTransformInverse, 1);
      for (i = 0; i < fSize / 2; i++) {
         destVector[i] = working_space[i];
      }
      break;
   case kTransformSin:
      fSize = 2 * fSize;
      working_space[fSize / 2] =
          working_space[fSize / 2 - 1] * TMath::Sqrt(2.0);
      for (i = fSize / 2 - 1; i > 0; i--) {
         a = pi * (Double_t) i / (Double_t) fSize;
         working_space[i + fSize] =
             -(Double_t) working_space[i - 1] * TMath::Cos(a);
         working_space[i] = (Double_t) working_space[i - 1] * TMath::Sin(a);
      } for (i = 2; i <= (fSize / 2); i++) {
         working_space[fSize - i + 1] = working_space[i - 1];
         working_space[fSize - i + 1 + fSize] =
             -working_space[i - 1 + fSize];
      }
      working_space[0] = 0;
      working_space[fSize] = 0;
      working_space[fSize / 2 + fSize] = 0;
      Fourier(working_space, fSize, 0, kTransformInverse, 0);
      for (i = 0; i < fSize / 2; i++) {
         destVector[i] = working_space[i];
      }
      break;
   case kTransformFourier:
      Fourier(working_space, fSize, 0, kTransformInverse, 0);
      for (i = 0; i < fSize; i++) {
         destVector[i] = working_space[i];
      }
      break;
   case kTransformHartley:
      Fourier(working_space, fSize, 1, kTransformInverse, 0);
      for (i = 0; i < fSize; i++) {
         destVector[i] = working_space[i];
      }
      break;
   case kTransformFourierWalsh:
   case kTransformFourierHaar:
   case kTransformWalshHaar:
   case kTransformCosWalsh:
   case kTransformCosHaar:
   case kTransformSinWalsh:
   case kTransformSinHaar:
      if (fTransformType > kTransformWalshHaar)
         k = (Int_t) TMath::Power(2, fDegree - 1);

      else
         k = (Int_t) TMath::Power(2, fDegree);
      j = fSize / k;
      for (i = 0, l = 0; i < fSize; i++, l = (l + k) % fSize) {
         working_space[fSize + l + i / j] = working_space[i];
         working_space[fSize + l + i / j + 2 * fSize] =
             working_space[i + 2 * fSize];
      }
      for (i = 0; i < fSize; i++) {
         working_space[i] = working_space[fSize + i];
         working_space[i + 2 * fSize] = working_space[fSize + i + 2 * fSize];
      }
      if (fTransformType == kTransformFourierWalsh
           || fTransformType == kTransformFourierHaar
           || fTransformType == kTransformWalshHaar) {
         GeneralInv(working_space, fSize, fDegree, fTransformType);
         for (i = 0; i < j; i++)
            BitReverseHaar(working_space, fSize, k, i * k);
      }

      else if (fTransformType == kTransformCosWalsh || fTransformType == kTransformCosHaar) {
         j = (Int_t) TMath::Power(2, fDegree) / 2;
         m = (Int_t) TMath::Power(2, fDegree);
         l = 2 * fSize / m;
         for (i = 0; i < fSize; i++) {
            k = i / j;
            k = 2 * k * j;
            a = pi * (Double_t) (i % j) / (Double_t) (2 * j);
            if (i % j == 0) {
               working_space[2 * fSize + k + i % j] =
                   working_space[i] * TMath::Sqrt(2.0);
               working_space[4 * fSize + 2 * fSize + k + i % j] = 0;
            }

            else {
               b = TMath::Sin(a);
               a = TMath::Cos(a);
               working_space[4 * fSize + 2 * fSize + k + i % j] =
                   -(Double_t) working_space[i] * b;
               working_space[2 * fSize + k + i % j] =
                   (Double_t) working_space[i] * a;
         } } for (i = 0; i < fSize; i++) {
            k = i / j;
            k = 2 * k * j;
            if (i % j == 0) {
               working_space[2 * fSize + k + j] = 0;
               working_space[4 * fSize + 2 * fSize + k + j] = 0;
            }

            else {
               working_space[2 * fSize + k + 2 * j - i % j] =
                   working_space[2 * fSize + k + i % j];
               working_space[4 * fSize + 2 * fSize + k + 2 * j - i % j] =
                   -working_space[4 * fSize + 2 * fSize + k + i % j];
            }
         }
         for (i = 0; i < 2 * fSize; i++) {
            working_space[i] = working_space[2 * fSize + i];
            working_space[4 * fSize + i] =
                working_space[4 * fSize + 2 * fSize + i];
         }
         GeneralInv(working_space, 2 * fSize, fDegree, fTransformType);
         m = (Int_t) TMath::Power(2, fDegree);
         l = 2 * fSize / m;
         for (i = 0; i < l; i++)
            BitReverseHaar(working_space, 2 * fSize, m, i * m);
      }

      else if (fTransformType == kTransformSinWalsh || fTransformType == kTransformSinHaar) {
         j = (Int_t) TMath::Power(2, fDegree) / 2;
         m = (Int_t) TMath::Power(2, fDegree);
         l = 2 * fSize / m;
         for (i = 0; i < fSize; i++) {
            k = i / j;
            k = 2 * k * j;
            a = pi * (Double_t) (i % j) / (Double_t) (2 * j);
            if (i % j == 0) {
               working_space[2 * fSize + k + j + i % j] =
                   working_space[j + k / 2 - i % j - 1] * TMath::Sqrt(2.0);
               working_space[4 * fSize + 2 * fSize + k + j + i % j] = 0;
            }

            else {
               b = TMath::Sin(a);
               a = TMath::Cos(a);
               working_space[4 * fSize + 2 * fSize + k + j + i % j] =
                   -(Double_t) working_space[j + k / 2 - i % j - 1] * b;
               working_space[2 * fSize + k + j + i % j] =
                   (Double_t) working_space[j + k / 2 - i % j - 1] * a;
         } } for (i = 0; i < fSize; i++) {
            k = i / j;
            k = 2 * k * j;
            if (i % j == 0) {
               working_space[2 * fSize + k] = 0;
               working_space[4 * fSize + 2 * fSize + k] = 0;
            }

            else {
               working_space[2 * fSize + k + i % j] =
                   working_space[2 * fSize + k + 2 * j - i % j];
               working_space[4 * fSize + 2 * fSize + k + i % j] =
                   -working_space[4 * fSize + 2 * fSize + k + 2 * j - i % j];
            }
         }
         for (i = 0; i < 2 * fSize; i++) {
            working_space[i] = working_space[2 * fSize + i];
            working_space[4 * fSize + i] =
                working_space[4 * fSize + 2 * fSize + i];
         }
         GeneralInv(working_space, 2 * fSize, fDegree, fTransformType);
         for (i = 0; i < l; i++)
            BitReverseHaar(working_space, 2 * fSize, m, i * m);
      }
      for (i = 0; i < fSize; i++) {
         if (fTransformType >= kTransformCosWalsh) {
            k = i / j;
            k = 2 * k * j;
            val = working_space[k + i % j];
         }

         else
            val = working_space[i];
         destVector[i] = val;
      }
      break;
   }
   delete[]working_space;
   return;
}


/////////////////////////////////////////////////////////////////////////////////
/// This function transforms the source spectrum. The calling program
/// should fill in input parameters. Then it multiplies transformed
/// coefficients in the given region (fXmin, fXmax) by the given
/// fEnhanceCoeff and transforms it back
/// Processed data are written into dest spectrum.
///
/// Function parameters:
///  - source-pointer to the vector of source spectrum, its length should
///    be size except for inverse FOURIER, FOUR-WALSh, FOUR-HAAR
///    transform. These need 2*size length to supply real and
///    imaginary coefficients.
///  - destVector-pointer to the vector of dest data, its length should be
///    size except for direct FOURIER, FOUR-WALSh, FOUR-HAAR. These
///    need 2*size length to store real and imaginary coefficients
///
/// ### Example - script Enhance.c:
///
/// \image html spectrumtransform_enhance_image001.jpg Fig. 1 Original spectrum (black line) and enhanced spectrum (red line) using Cosine transform (channels 0-1024 were multiplied by 2)
///
/// #### Script:
///
/// Example to illustrate Enhance function (class TSpectrumTransform).
/// To execute this example, do:
///
/// `root > .x Enhance.C`
///
/// ~~~ {.cpp}
///   void Enhance() {
///      Int_t i;
///      Double_t nbins = 4096;
///      Double_t xmin = 0;
///      Double_t xmax = (Double_t)nbins;
///      Double_t * source = new Double_t[nbins];
///      Double_t * dest = new Double_t[nbins];
///      TH1F *h = new TH1F("h","Enhancement using Cosine transform",nbins,xmin,xmax);
///      TH1F *d = new TH1F("d","",nbins,xmin,xmax);
///      TFile *f = new TFile("spectra/TSpectrum.root");
///      h=(TH1F*) f->Get("transform1;1");
///      for (i = 0; i < nbins; i++) source[i]=h->GetBinContent(i + 1);
///      TCanvas *Transform1 = gROOT->GetListOfCanvases()->FindObject("Transform1");
///      if (!Transform1) Transform1 = new TCanvas("Transform","Transform1",10,10,1000,700);
///      h->SetAxisRange(700,1024);
///      h->Draw("L");
///      TSpectrum *s = new TSpectrum();
///      TSpectrumTransform *t = new TSpectrumTransform(4096);
///      t->SetTransformType(t->kTransformCos,0);
///      t->SetRegion(0, 1024);
///      t->SetEnhanceCoeff(2);
///      t->Enhance(source,dest);
///      for (i = 0; i < nbins; i++) d->SetBinContent(i + 1,dest[i]);
///      d->SetLineColor(kRed);
///      d->Draw("SAME L");
///   }
/// ~~~

void TSpectrumTransform::Enhance(const Double_t *source, Double_t *destVector)
{
   int i, j=0, k = 1, m, l;
   Double_t val;
   Double_t *working_space = 0;
   Double_t a, b, pi = 3.14159265358979323846, old_area, new_area;
   if (fTransformType >= kTransformFourierWalsh && fTransformType <= kTransformSinHaar) {
      if (fTransformType >= kTransformCosWalsh)
         fDegree += 1;
      k = (Int_t) TMath::Power(2, fDegree);
      j = fSize / k;
   }
   switch (fTransformType) {
   case kTransformHaar:
   case kTransformWalsh:
      working_space = new Double_t[2 * fSize];
      break;
   case kTransformCos:
   case kTransformSin:
   case kTransformFourier:
   case kTransformHartley:
   case kTransformFourierWalsh:
   case kTransformFourierHaar:
   case kTransformWalshHaar:
      working_space = new Double_t[4 * fSize];
      break;
   case kTransformCosWalsh:
   case kTransformCosHaar:
   case kTransformSinWalsh:
   case kTransformSinHaar:
      working_space = new Double_t[8 * fSize];
      break;
   }
   switch (fTransformType) {
   case kTransformHaar:
      for (i = 0; i < fSize; i++) {
         working_space[i] = source[i];
      }
      Haar(working_space, fSize, kTransformForward);
      break;
   case kTransformWalsh:
      for (i = 0; i < fSize; i++) {
         working_space[i] = source[i];
      }
      Walsh(working_space, fSize);
      BitReverse(working_space, fSize);
      break;
   case kTransformCos:
      fSize = 2 * fSize;
      for (i = 1; i <= (fSize / 2); i++) {
         val = source[i - 1];
         working_space[i - 1] = val;
         working_space[fSize - i] = val;
      }
      Fourier(working_space, fSize, 0, kTransformForward, 0);
      for (i = 0; i < fSize / 2; i++) {
         a = pi * (Double_t) i / (Double_t) fSize;
         a = TMath::Cos(a);
         b = working_space[i];
         a = b / a;
         working_space[i] = a;
         working_space[i + fSize] = 0;
      } working_space[0] = working_space[0] / TMath::Sqrt(2.0);
      fSize = fSize / 2;
      break;
   case kTransformSin:
      fSize = 2 * fSize;
      for (i = 1; i <= (fSize / 2); i++) {
         val = source[i - 1];
         working_space[i - 1] = val;
         working_space[fSize - i] = -val;
      }
      Fourier(working_space, fSize, 0, kTransformForward, 0);
      for (i = 0; i < fSize / 2; i++) {
         a = pi * (Double_t) i / (Double_t) fSize;
         a = TMath::Sin(a);
         b = working_space[i];
         if (a != 0)
            a = b / a;
         working_space[i - 1] = a;
         working_space[i + fSize] = 0;
      }
      working_space[fSize / 2 - 1] =
          working_space[fSize / 2] / TMath::Sqrt(2.0);
      fSize = fSize / 2;
      break;
   case kTransformFourier:
      for (i = 0; i < fSize; i++) {
         working_space[i] = source[i];
      }
      Fourier(working_space, fSize, 0, kTransformForward, 0);
      break;
   case kTransformHartley:
      for (i = 0; i < fSize; i++) {
         working_space[i] = source[i];
      }
      Fourier(working_space, fSize, 1, kTransformForward, 0);
      break;
   case kTransformFourierWalsh:
   case kTransformFourierHaar:
   case kTransformWalshHaar:
   case kTransformCosWalsh:
   case kTransformCosHaar:
   case kTransformSinWalsh:
   case kTransformSinHaar:
      for (i = 0; i < fSize; i++) {
         val = source[i];
         if (fTransformType == kTransformCosWalsh || fTransformType == kTransformCosHaar) {
            j = (Int_t) TMath::Power(2, fDegree) / 2;
            k = i / j;
            k = 2 * k * j;
            working_space[k + i % j] = val;
            working_space[k + 2 * j - 1 - i % j] = val;
         }

         else if (fTransformType == kTransformSinWalsh
                  || fTransformType == kTransformSinHaar) {
            j = (Int_t) TMath::Power(2, fDegree) / 2;
            k = i / j;
            k = 2 * k * j;
            working_space[k + i % j] = val;
            working_space[k + 2 * j - 1 - i % j] = -val;
         }

         else
            working_space[i] = val;
      }
      if (fTransformType == kTransformFourierWalsh
           || fTransformType == kTransformFourierHaar
           || fTransformType == kTransformWalshHaar) {
         for (i = 0; i < j; i++)
            BitReverseHaar(working_space, fSize, k, i * k);
         GeneralExe(working_space, 0, fSize, fDegree, fTransformType);
      }

      else if (fTransformType == kTransformCosWalsh || fTransformType == kTransformCosHaar) {
         m = (Int_t) TMath::Power(2, fDegree);
         l = 2 * fSize / m;
         for (i = 0; i < l; i++)
            BitReverseHaar(working_space, 2 * fSize, m, i * m);
         GeneralExe(working_space, 0, 2 * fSize, fDegree, fTransformType);
         for (i = 0; i < fSize; i++) {
            k = i / j;
            k = 2 * k * j;
            a = pi * (Double_t) (i % j) / (Double_t) (2 * j);
            a = TMath::Cos(a);
            b = working_space[k + i % j];
            if (i % j == 0)
               a = b / TMath::Sqrt(2.0);

            else
               a = b / a;
            working_space[i] = a;
            working_space[i + 2 * fSize] = 0;
         }
      }

      else if (fTransformType == kTransformSinWalsh || fTransformType == kTransformSinHaar) {
         m = (Int_t) TMath::Power(2, fDegree);
         l = 2 * fSize / m;
         for (i = 0; i < l; i++)
            BitReverseHaar(working_space, 2 * fSize, m, i * m);
         GeneralExe(working_space, 0, 2 * fSize, fDegree, fTransformType);
         for (i = 0; i < fSize; i++) {
            k = i / j;
            k = 2 * k * j;
            a = pi * (Double_t) (i % j) / (Double_t) (2 * j);
            a = TMath::Cos(a);
            b = working_space[j + k + i % j];
            if (i % j == 0)
               a = b / TMath::Sqrt(2.0);

            else
               a = b / a;
            working_space[j + k / 2 - i % j - 1] = a;
            working_space[i + 2 * fSize] = 0;
         }
      }
      if (fTransformType > kTransformWalshHaar)
         k = (Int_t) TMath::Power(2, fDegree - 1);

      else
         k = (Int_t) TMath::Power(2, fDegree);
      j = fSize / k;
      for (i = 0, l = 0; i < fSize; i++, l = (l + k) % fSize) {
         working_space[fSize + i] = working_space[l + i / j];
         working_space[fSize + i + 2 * fSize] =
             working_space[l + i / j + 2 * fSize];
      }
      for (i = 0; i < fSize; i++) {
         working_space[i] = working_space[fSize + i];
         working_space[i + 2 * fSize] = working_space[fSize + i + 2 * fSize];
      }
      break;
   }
   if (!working_space) return;
   for (i = 0, old_area = 0; i < fSize; i++) {
      old_area += working_space[i];
   }
   for (i = 0, new_area = 0; i < fSize; i++) {
      if (i >= fXmin && i <= fXmax)
         working_space[i] *= fEnhanceCoeff;
      new_area += working_space[i];
   }
   if (new_area != 0) {
      a = old_area / new_area;
      for (i = 0; i < fSize; i++) {
         working_space[i] *= a;
      }
   }
   if (fTransformType == kTransformFourier) {
      for (i = 0, old_area = 0; i < fSize; i++) {
         old_area += working_space[i + fSize];
      }
      for (i = 0, new_area = 0; i < fSize; i++) {
         if (i >= fXmin && i <= fXmax)
            working_space[i + fSize] *= fEnhanceCoeff;
         new_area += working_space[i + fSize];
      }
      if (new_area != 0) {
         a = old_area / new_area;
         for (i = 0; i < fSize; i++) {
            working_space[i + fSize] *= a;
         }
      }
   }

   else if (fTransformType == kTransformFourierWalsh
            || fTransformType == kTransformFourierHaar) {
      for (i = 0, old_area = 0; i < fSize; i++) {
         old_area += working_space[i + 2 * fSize];
      }
      for (i = 0, new_area = 0; i < fSize; i++) {
         if (i >= fXmin && i <= fXmax)
            working_space[i + 2 * fSize] *= fEnhanceCoeff;
         new_area += working_space[i + 2 * fSize];
      }
      if (new_area != 0) {
         a = old_area / new_area;
         for (i = 0; i < fSize; i++) {
            working_space[i + 2 * fSize] *= a;
         }
      }
   }
   switch (fTransformType) {
   case kTransformHaar:
      Haar(working_space, fSize, kTransformInverse);
      for (i = 0; i < fSize; i++) {
         destVector[i] = working_space[i];
      }
      break;
   case kTransformWalsh:
      BitReverse(working_space, fSize);
      Walsh(working_space, fSize);
      for (i = 0; i < fSize; i++) {
         destVector[i] = working_space[i];
      }
      break;
   case kTransformCos:
      fSize = 2 * fSize;
      working_space[0] = working_space[0] * TMath::Sqrt(2.0);
      for (i = 0; i < fSize / 2; i++) {
         a = pi * (Double_t) i / (Double_t) fSize;
         b = TMath::Sin(a);
         a = TMath::Cos(a);
         working_space[i + fSize] = (Double_t) working_space[i] * b;
         working_space[i] = (Double_t) working_space[i] * a;
      } for (i = 2; i <= (fSize / 2); i++) {
         working_space[fSize - i + 1] = working_space[i - 1];
         working_space[fSize - i + 1 + fSize] =
             -working_space[i - 1 + fSize];
      }
      working_space[fSize / 2] = 0;
      working_space[fSize / 2 + fSize] = 0;
      Fourier(working_space, fSize, 0, kTransformInverse, 1);
      for (i = 0; i < fSize / 2; i++) {
         destVector[i] = working_space[i];
      }
      break;
   case kTransformSin:
      fSize = 2 * fSize;
      working_space[fSize / 2] =
          working_space[fSize / 2 - 1] * TMath::Sqrt(2.0);
      for (i = fSize / 2 - 1; i > 0; i--) {
         a = pi * (Double_t) i / (Double_t) fSize;
         working_space[i + fSize] =
             -(Double_t) working_space[i - 1] * TMath::Cos(a);
         working_space[i] = (Double_t) working_space[i - 1] * TMath::Sin(a);
      } for (i = 2; i <= (fSize / 2); i++) {
         working_space[fSize - i + 1] = working_space[i - 1];
         working_space[fSize - i + 1 + fSize] =
             -working_space[i - 1 + fSize];
      }
      working_space[0] = 0;
      working_space[fSize] = 0;
      working_space[fSize / 2 + fSize] = 0;
      Fourier(working_space, fSize, 0, kTransformInverse, 0);
      for (i = 0; i < fSize / 2; i++) {
         destVector[i] = working_space[i];
      }
      break;
   case kTransformFourier:
      Fourier(working_space, fSize, 0, kTransformInverse, 0);
      for (i = 0; i < fSize; i++) {
         destVector[i] = working_space[i];
      }
      break;
   case kTransformHartley:
      Fourier(working_space, fSize, 1, kTransformInverse, 0);
      for (i = 0; i < fSize; i++) {
         destVector[i] = working_space[i];
      }
      break;
   case kTransformFourierWalsh:
   case kTransformFourierHaar:
   case kTransformWalshHaar:
   case kTransformCosWalsh:
   case kTransformCosHaar:
   case kTransformSinWalsh:
   case kTransformSinHaar:
      if (fTransformType > kTransformWalshHaar)
         k = (Int_t) TMath::Power(2, fDegree - 1);

      else
         k = (Int_t) TMath::Power(2, fDegree);
      j = fSize / k;
      for (i = 0, l = 0; i < fSize; i++, l = (l + k) % fSize) {
         working_space[fSize + l + i / j] = working_space[i];
         working_space[fSize + l + i / j + 2 * fSize] =
             working_space[i + 2 * fSize];
      }
      for (i = 0; i < fSize; i++) {
         working_space[i] = working_space[fSize + i];
         working_space[i + 2 * fSize] = working_space[fSize + i + 2 * fSize];
      }
      if (fTransformType == kTransformFourierWalsh
           || fTransformType == kTransformFourierHaar
           || fTransformType == kTransformWalshHaar) {
         GeneralInv(working_space, fSize, fDegree, fTransformType);
         for (i = 0; i < j; i++)
            BitReverseHaar(working_space, fSize, k, i * k);
      }

      else if (fTransformType == kTransformCosWalsh || fTransformType == kTransformCosHaar) {
         j = (Int_t) TMath::Power(2, fDegree) / 2;
         m = (Int_t) TMath::Power(2, fDegree);
         l = 2 * fSize / m;
         for (i = 0; i < fSize; i++) {
            k = i / j;
            k = 2 * k * j;
            a = pi * (Double_t) (i % j) / (Double_t) (2 * j);
            if (i % j == 0) {
               working_space[2 * fSize + k + i % j] =
                   working_space[i] * TMath::Sqrt(2.0);
               working_space[4 * fSize + 2 * fSize + k + i % j] = 0;
            }

            else {
               b = TMath::Sin(a);
               a = TMath::Cos(a);
               working_space[4 * fSize + 2 * fSize + k + i % j] =
                   -(Double_t) working_space[i] * b;
               working_space[2 * fSize + k + i % j] =
                   (Double_t) working_space[i] * a;
         } } for (i = 0; i < fSize; i++) {
            k = i / j;
            k = 2 * k * j;
            if (i % j == 0) {
               working_space[2 * fSize + k + j] = 0;
               working_space[4 * fSize + 2 * fSize + k + j] = 0;
            }

            else {
               working_space[2 * fSize + k + 2 * j - i % j] =
                   working_space[2 * fSize + k + i % j];
               working_space[4 * fSize + 2 * fSize + k + 2 * j - i % j] =
                   -working_space[4 * fSize + 2 * fSize + k + i % j];
            }
         }
         for (i = 0; i < 2 * fSize; i++) {
            working_space[i] = working_space[2 * fSize + i];
            working_space[4 * fSize + i] =
                working_space[4 * fSize + 2 * fSize + i];
         }
         GeneralInv(working_space, 2 * fSize, fDegree, fTransformType);
         m = (Int_t) TMath::Power(2, fDegree);
         l = 2 * fSize / m;
         for (i = 0; i < l; i++)
            BitReverseHaar(working_space, 2 * fSize, m, i * m);
      }

      else if (fTransformType == kTransformSinWalsh || fTransformType == kTransformSinHaar) {
         j = (Int_t) TMath::Power(2, fDegree) / 2;
         m = (Int_t) TMath::Power(2, fDegree);
         l = 2 * fSize / m;
         for (i = 0; i < fSize; i++) {
            k = i / j;
            k = 2 * k * j;
            a = pi * (Double_t) (i % j) / (Double_t) (2 * j);
            if (i % j == 0) {
               working_space[2 * fSize + k + j + i % j] =
                   working_space[j + k / 2 - i % j - 1] * TMath::Sqrt(2.0);
               working_space[4 * fSize + 2 * fSize + k + j + i % j] = 0;
            }

            else {
               b = TMath::Sin(a);
               a = TMath::Cos(a);
               working_space[4 * fSize + 2 * fSize + k + j + i % j] =
                   -(Double_t) working_space[j + k / 2 - i % j - 1] * b;
               working_space[2 * fSize + k + j + i % j] =
                   (Double_t) working_space[j + k / 2 - i % j - 1] * a;
         } } for (i = 0; i < fSize; i++) {
            k = i / j;
            k = 2 * k * j;
            if (i % j == 0) {
               working_space[2 * fSize + k] = 0;
               working_space[4 * fSize + 2 * fSize + k] = 0;
            }

            else {
               working_space[2 * fSize + k + i % j] =
                   working_space[2 * fSize + k + 2 * j - i % j];
               working_space[4 * fSize + 2 * fSize + k + i % j] =
                   -working_space[4 * fSize + 2 * fSize + k + 2 * j - i % j];
            }
         }
         for (i = 0; i < 2 * fSize; i++) {
            working_space[i] = working_space[2 * fSize + i];
            working_space[4 * fSize + i] =
                working_space[4 * fSize + 2 * fSize + i];
         }
         GeneralInv(working_space, 2 * fSize, fDegree, fTransformType);
         for (i = 0; i < l; i++)
            BitReverseHaar(working_space, 2 * fSize, m, i * m);
      }
      for (i = 0; i < fSize; i++) {
         if (fTransformType >= kTransformCosWalsh) {
            k = i / j;
            k = 2 * k * j;
            val = working_space[k + i % j];
         }

         else
            val = working_space[i];
         destVector[i] = val;
      }
      break;
   }
   delete[]working_space;
   return;
}

////////////////////////////////////////////////////////////////////////////////
/// This function sets the following parameters for transform:
///  - transType - type of transform (Haar, Walsh, Cosine, Sine, Fourier, Hartley, Fourier-Walsh, Fourier-Haar, Walsh-Haar, Cosine-Walsh, Cosine-Haar, Sine-Walsh, Sine-Haar)
///  - degree - degree of mixed transform, applies only for Fourier-Walsh, Fourier-Haar, Walsh-Haar, Cosine-Walsh, Cosine-Haar, Sine-Walsh, Sine-Haar transforms

void TSpectrumTransform::SetTransformType(Int_t transType, Int_t degree)
{
   Int_t j, n;
   j = 0;
   n = 1;
   for (; n < fSize;) {
      j += 1;
      n = n * 2;
   }
   if (transType < kTransformHaar || transType > kTransformSinHaar){
      Error ("TSpectrumTransform","Invalid type of transform");
      return;
   }
   if (transType >= kTransformFourierWalsh && transType <= kTransformSinHaar) {
      if (degree > j || degree < 1){
         Error ("TSpectrumTransform","Invalid degree of mixed transform");
         return;
      }
   }
   fTransformType=transType;
   fDegree=degree;
}

////////////////////////////////////////////////////////////////////////////////
/// This function sets the filtering or enhancement region:
///  - xmin, xmax

void TSpectrumTransform::SetRegion(Int_t xmin, Int_t xmax)
{
   if(xmin<0 || xmax < xmin || xmax >= fSize){
      Error("TSpectrumTransform", "Wrong range");
      return;
   }
   fXmin = xmin;
   fXmax = xmax;
}

////////////////////////////////////////////////////////////////////////////////
/// This function sets the direction of the transform:
///  - direction (forward or inverse)

void TSpectrumTransform::SetDirection(Int_t direction)
{
   if(direction != kTransformForward && direction != kTransformInverse){
      Error("TSpectrumTransform", "Wrong direction");
      return;
   }
   fDirection = direction;
}

////////////////////////////////////////////////////////////////////////////////
/// This function sets the filter coefficient:
///  - filterCoeff - after the transform the filtered region (xmin, xmax) is replaced by this coefficient. Applies only for filtereng operation.

void TSpectrumTransform::SetFilterCoeff(Double_t filterCoeff)
{
   fFilterCoeff = filterCoeff;
}

////////////////////////////////////////////////////////////////////////////////
/// This function sets the enhancement coefficient:
///  - enhanceCoeff - after the transform the enhanced region (xmin, xmax) is multiplied by this coefficient. Applies only for enhancement operation.

void TSpectrumTransform::SetEnhanceCoeff(Double_t enhanceCoeff)
{
   fEnhanceCoeff = enhanceCoeff;
}
