// @(#)root/spectrum:$Id$
// Author: Miroslav Morhac   25/09/06

//__________________________________________________________________________
//   THIS CLASS CONTAINS ORTHOGONAL TRANSFORM  FUNCTIONS.                  //
//                                                                         //
//   These functions were written by:                                      //
//   Miroslav Morhac                                                       //
//   Institute of Physics                                                  //
//   Slovak Academy of Sciences                                            //
//   Dubravska cesta 9, 842 28 BRATISLAVA                                  //
//   SLOVAKIA                                                              //
//                                                                         //
//   email:fyzimiro@savba.sk,    fax:+421 7 54772479                       //
//                                                                         //
//  The original code in C has been repackaged as a C++ class by R.Brun    //
//                                                                         //
//  The algorithms in this class have been published in the following      //
//  references:                                                            //
//                                                                         //
//  [1] C.V. Hampton, B. Lian, Wm. C. McHarris: Fast-Fourier-transform     //
//      spectral enhancement techniques for gamma-ray spectroscopy.NIM A353//
//      (1994) 280-284.                                                    //
//  [2] Morhac M., Matousek V., New adaptive Cosine-Walsh  transform and   //
//      its application to nuclear data compression, IEEE Transactions on  //
//      Signal Processing 48 (2000) 2693.                                  //
//  [3] Morhac M., Matousek V., Data compression using new fast adaptive   //
//      Cosine-Haar transforms, Digital Signal Processing 8 (1998) 63.     //
//  [4] Morhac M., Matousek V.: Multidimensional nuclear data compression  //
//      using fast adaptive Walsh-Haar transform. Acta Physica Slovaca 51  //
//     (2001) 307.                                                         //
//____________________________________________________________________________

#include "TSpectrumTransform.h"
#include "TMath.h"

ClassImp(TSpectrumTransform)

//____________________________________________________________________________
TSpectrumTransform::TSpectrumTransform()
{
   //default constructor
   fSize=0;
   fTransformType=kTransformCos;
   fDegree=0;
   fDirection=kTransformForward;
   fXmin=0;
   fXmax=0;
   fFilterCoeff=0;
   fEnhanceCoeff=0.5;
}

//____________________________________________________________________________
TSpectrumTransform::TSpectrumTransform(Int_t size):TNamed("SpectrumTransform", "Miroslav Morhac transformer")
{
//the constructor creates TSpectrumTransform object. Its size must be > than zero and must be power of 2.
//It sets default transform type to be Cosine transform. Transform parameters can be changed using setter functions.
   Int_t j,n;
   if (size <= 0){
      Error ("TSpectrumTransform","Invalid length, must be > than 0");
      return;
   }
   j = 0;
   n = 1;
   for (; n < size;) {
      j += 1;
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


//______________________________________________________________________________
TSpectrumTransform::~TSpectrumTransform()
{
   //destructor
}

//_____________________________________________________________________________
void TSpectrumTransform::Haar(Double_t *working_space, int num, int direction)
{
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCION                                                          //
//                                                                              //
//   This function calculates Haar transform of a part of data                   //
//      Function parameters:                                                    //
//              -working_space-pointer to vector of transformed data            //
//              -num-length of processed data                                   //
//              -direction-forward or inverse transform                         //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
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

//____________________________________________________________________________
void TSpectrumTransform::Walsh(Double_t *working_space, int num)
{
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCION                                                          //
//                                                                              //
//   This function calculates Walsh transform of a part of data                  //
//      Function parameters:                                                    //
//              -working_space-pointer to vector of transformed data            //
//              -num-length of processed data                                   //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
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

//____________________________________________________________________________
void TSpectrumTransform::BitReverse(Double_t *working_space, int num)
{
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCION                                                          //
//                                                                              //
//   This function carries out bir-reverse reordering of data                    //
//      Function parameters:                                                    //
//              -working_space-pointer to vector of processed data              //
//              -num-length of processed data                                   //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
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

//____________________________________________________________________________
void TSpectrumTransform::Fourier(Double_t *working_space, int num, int hartley,
                          int direction, int zt_clear)
{
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCION                                                          //
//                                                                              //
//   This function calculates Fourier based transform of a part of data          //
//      Function parameters:                                                    //
//              -working_space-pointer to vector of transformed data            //
//              -num-length of processed data                                   //
//              -hartley-1 if it is Hartley transform, 0 othewise               //
//              -direction-forward or inverse transform                         //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
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

//____________________________________________________________________________
void TSpectrumTransform::BitReverseHaar(Double_t *working_space, int shift, int num,
                                 int start)
{
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCION                                                          //
//                                                                              //
//   This function carries out bir-reverse reordering for Haar transform         //
//      Function parameters:                                                    //
//              -working_space-pointer to vector of processed data              //
//              -shift-shift of position of processing                          //
//              -start-initial position of processed data                       //
//              -num-length of processed data                                   //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
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

//____________________________________________________________________________
int TSpectrumTransform::GeneralExe(Double_t *working_space, int zt_clear, int num,
                            int degree, int type)
{
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCION                                                          //
//                                                                              //
//   This function calculates generalized (mixed) transforms of different degrees//
//      Function parameters:                                                    //
//              -working_space-pointer to vector of transformed data            //
//              -zt_clear-flag to clear imaginary data before staring           //
//              -num-length of processed data                                   //
//              -degree-degree of transform (see manual)                        //
//              -type-type of mixed transform (see manual)                      //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
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

//____________________________________________________________________________
int TSpectrumTransform::GeneralInv(Double_t *working_space, int num, int degree,
                            int type)
{
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCION                                                          //
//                                                                              //
//   This function calculates inverse generalized (mixed) transforms             //
//      Function parameters:                                                    //
//              -working_space-pointer to vector of transformed data            //
//              -num-length of processed data                                   //
//              -degree-degree of transform (see manual)                        //
//              -type-type of mixed transform (see manual)                      //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
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


//////////END OF AUXILIARY FUNCTIONS FOR TRANSFORM! FUNCTION////////////////////////
//////////TRANSFORM FUNCTION - CALCULATES DIFFERENT 1-D DIRECT AND INVERSE ORTHOGONAL TRANSFORMS//////

//____________________________________________________________________________
void TSpectrumTransform::Transform(const Double_t *source, Double_t *destVector)
{
///////////////////////////////////////////////////////////////////////////////
//        ONE-DIMENSIONAL TRANSFORM FUNCTION
//        This function transforms the source spectrum. The calling program
//        should fill in input parameters.
//        Transformed data are written into dest spectrum.
//
//        Function parameters:
//        source-pointer to the vector of source spectrum, its length should
//             be size except for inverse FOURIER, FOUR-WALSH, FOUR-HAAR
//             transform. These need 2*size length to supply real and
//             imaginary coefficients.
//        destVector-pointer to the vector of dest data, its length should be
//             size except for direct FOURIER, FOUR-WALSH, FOUR-HAAR. These
//             need 2*size length to store real and imaginary coefficients
//
///////////////////////////////////////////////////////////////////////////////
//Begin_Html <!--
/* -->
<div class=Section1>

<p class=MsoNormal><b><span style='font-size:14.0pt'>Transform methods</span></b></p>

<p class=MsoNormal style='text-align:justify'><i>&nbsp;</i></p>

<p class=MsoNormal style='text-align:justify'><i>Goal: to analyze experimental
data using orthogonal transforms</i></p>

<p class=MsoNormal style='margin-left:36.0pt;text-align:justify;text-indent:
-18.0pt'>•<span style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span>orthogonal transforms can be successfully used for the processing of
nuclear spectra (not only) </p>

<p class=MsoNormal style='margin-left:36.0pt;text-align:justify;text-indent:
-18.0pt'>•<span style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span>they can be used to remove high frequency noise, to increase
signal-to-background ratio as well as to enhance low intensity components [1],
to carry out e.g. Fourier analysis etc. </p>

<p class=MsoNormal style='margin-left:36.0pt;text-align:justify;text-indent:
-18.0pt'>•<span style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span>we have implemented the function for the calculation of the commonly
used orthogonal transforms as well as functions for the filtration and
enhancement of experimental data</p>

<p class=MsoNormal><i>&nbsp;</i></p>

<p class=MsoNormal><i>Function:</i></p>

<p class=MsoNormal><b>void TSpectrumTransform::Transform(const <a
href="http://root.cern.ch/root/html/ListOfTypes.html#double">double</a> *fSource,
<a href="http://root.cern.ch/root/html/ListOfTypes.html#double">double</a>
*fDest)</b></p>

<p class=MsoNormal style='text-align:justify'>&nbsp;</p>

<p class=MsoNormal style='text-align:justify'>This function transforms the
source spectrum according to the given input parameters. Transformed data are
written into dest spectrum. Before the Transform function is called the class
must be created by constructor and the type of the transform as well as some
other parameters must be set using a set of setter functions.</p>

<p class=MsoNormal style='text-align:justify'>&nbsp;</p>

<p class=MsoNormal><i><span style='color:red'>Member variables of
TSpectrumTransform class:</span></i></p>

<p class=MsoNormal style='margin-left:25.65pt;text-align:justify'> <b>fSource</b>-pointer
to the vector of source spectrum. Its length should be equal to the “fSize”
parameter except for inverse FOURIER, FOUR-WALSH, FOUR-HAAR transforms. These
need “2*fSize” length to supply real and imaginary coefficients.                   </p>

<p class=MsoNormal style='margin-left:25.65pt;text-align:justify'><b>fDest</b>-pointer
to the vector of destination spectrum. Its length should be equal to the
“fSize” parameter except for inverse FOURIER, FOUR-WALSH, FOUR-HAAR transforms.
These need “2*fSize” length to store real and imaginary coefficients. </p>

<p class=MsoNormal style='text-align:justify'>        <b>fSize</b>-basic length
of the source and dest spectrum. <span style='color:fuchsia'>It should be power
of 2.</span></p>

<p class=MsoNormal style='margin-left:25.65pt;text-align:justify;text-indent:
-2.85pt'><b>fType</b>-type of transform</p>

<p class=MsoNormal style='text-align:justify'>            Classic transforms:</p>

<p class=MsoNormal style='text-align:justify'>                        kTransformHaar
</p>

<p class=MsoNormal style='text-align:justify'>                        kTransformWalsh
</p>

<p class=MsoNormal style='text-align:justify'>                        kTransformCos
</p>

<p class=MsoNormal style='text-align:justify'>                        kTransformSin
</p>

<p class=MsoNormal style='text-align:justify'>                        kTransformFourier
</p>

<p class=MsoNormal style='text-align:justify'>                        kTransformHartey
</p>

<p class=MsoNormal style='text-align:justify'>            Mixed transforms:</p>

<p class=MsoNormal style='text-align:justify'>                        kTransformFourierWalsh
</p>

<p class=MsoNormal style='text-align:justify'>                        kTransformFourierHaar
</p>

<p class=MsoNormal style='text-align:justify'>                        kTransformWalshHaar
</p>

<p class=MsoNormal style='text-align:justify'>                        kTransformCosWalsh
</p>

<p class=MsoNormal style='text-align:justify'>                        kTransformCosHaar
</p>

<p class=MsoNormal style='text-align:justify'>                        kTransformSinWalsh
</p>

<p class=MsoNormal style='text-align:justify'>                        kTransformSinHaar
</p>

<p class=MsoNormal style='text-align:justify;text-indent:22.8pt'><b>fDirection</b>-direction-transform
direction (forward, inverse)</p>

<p class=MsoNormal style='text-align:justify'>                        kTransformForward
</p>

<p class=MsoNormal style='text-align:justify'>                        kTransformInverse
</p>

<p class=MsoNormal style='text-align:justify;text-indent:22.8pt'><b>fDegree</b>-applies
only for mixed transforms [2], [3], [4]. </p>

<p class=MsoNormal style='text-align:justify;text-indent:22.8pt'>                
<span style='color:fuchsia'> Allowed range  <sub><img border=0 width=100
height=27 src="gif/spectrumtransform_transform_image001.gif"></sub>. </span></p>

<p class=MsoNormal style='text-align:justify'><b><i>References:</i></b></p>

<p class=MsoNormal style='text-align:justify'>[1] C.V. Hampton, B. Lian, Wm. C.
McHarris: Fast-Fourier-transform spectral enhancement techniques for gamma-ray
spectroscopy. NIM A353 (1994) 280-284. </p>

<p class=MsoNormal style='text-align:justify'>[2] Morhá&#269; M., Matoušek V.,
New adaptive Cosine-Walsh  transform and its application to nuclear data
compression, IEEE Transactions on Signal Processing 48 (2000) 2693.  </p>

<p class=MsoNormal style='text-align:justify'>[3] Morhá&#269; M., Matoušek V.,
Data compression using new fast adaptive Cosine-Haar transforms, Digital Signal
Processing 8 (1998) 63. </p>

<p class=MsoNormal style='text-align:justify'>[4] Morhá&#269; M., Matoušek V.:
Multidimensional nuclear data compression using fast adaptive Walsh-Haar
transform. Acta Physica Slovaca 51 (2001) 307. </p>

<p class=MsoNormal style='text-align:justify'>&nbsp;</p>

<p class=MsoNormal style='text-align:justify'><i>Example  – script Transform.c:</i></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:18.0pt'><img
width=600 height=324 src="gif/spectrumtransform_transform_image002.jpg"></span></p>

<p class=MsoNormal><b>Fig. 1 Original gamma-ray spectrum</b></p>

<p class=MsoNormal><b><span style='font-size:14.0pt'><img border=0 width=601
height=402 src="gif/spectrumtransform_transform_image003.jpg"></span></b></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:18.0pt'>&nbsp;</span></p>

<p class=MsoNormal><b>Fig. 2 Transformed spectrum from Fig. 1 using Cosine
transform</b></p>

<p class=MsoNormal><b><span style='font-size:16.0pt;color:#339966'>&nbsp;</span></b></p>

<p class=MsoNormal><b><span style='color:#339966'>Script:</span></b></p>

<p class=MsoNormal><span style='font-size:10.0pt'>// Example to illustrate
Transform function (class TSpectrumTransform).</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>// To execute this example,
do</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>// root &gt; .x Transform.C</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>#include &lt;TSpectrum&gt;</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>#include
&lt;TSpectrumTransform&gt;</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>&nbsp;</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>void Transform() {</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   Int_t i;</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   Double_t nbins =
4096;</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   Double_t xmin  =
0;</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   Double_t xmax  =
(Double_t)nbins;</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   </span><span
style='font-size:10.0pt'>Double_t * source = new Double_t[nbins];</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t * dest = new
Double_t[nbins];   </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   TH1F *h = new TH1F(&quot;h&quot;,&quot;Transformed
spectrum using Cosine transform&quot;,nbins,xmin,xmax);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   TFile *f = new
TFile(&quot;spectra\\TSpectrum.root&quot;);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   h=(TH1F*)
f-&gt;Get(&quot;transform1;1&quot;);   </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   for (i = 0; i &lt; nbins;
i++) source[i]=h-&gt;GetBinContent(i + 1);         </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   TCanvas *Transform1 =
gROOT-&gt;GetListOfCanvases()-&gt;FindObject(&quot;Transform1&quot;);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   if (!Transform1)
Transform1 = new
TCanvas(&quot;Transform&quot;,&quot;Transform1&quot;,10,10,1000,700);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   TSpectrum *s = new
TSpectrum();</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   TSpectrumTransform *t =
new TSpectrumTransform(4096);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   </span><span lang=FR
style='font-size:10.0pt'>t-&gt;SetTransformType(t-&gt;kTransformCos,0);</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>  
t-&gt;SetDirection(t-&gt;kTransformForward);</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   </span><span
style='font-size:10.0pt'>t-&gt;Transform(source,dest);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   for (i = 0; i &lt; nbins;
i++) h-&gt;SetBinContent(i + 1,dest[i]);   </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>  
h-&gt;SetLineColor(kRed);      </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   h-&gt;Draw(&quot;L&quot;);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>}</span></p>

</div>

<!-- */
// --> End_Html
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

//////////FilterZonal FUNCTION - CALCULATES DIFFERENT 1-D ORTHOGONAL TRANSFORMS, SETS GIVEN REGION TO FILTER COEFFICIENT AND TRANSFORMS IT BACK//////

//______________________________________________________________________________
void TSpectrumTransform::FilterZonal(const Double_t *source, Double_t *destVector)
{
////////////////////////////////////////////////////////////////////////////////
//        ONE-DIMENSIONAL FILTER ZONAL FUNCTION
//        This function transforms the source spectrum. The calling program
//        should fill in input parameters. Then it sets transformed
//        coefficients in the given region (fXmin, fXmax) to the given
//        fFilterCoeff and transforms it back.
//        Filtered data are written into dest spectrum.
//
//        Function parameters:
//        source-pointer to the vector of source spectrum, its length should
//             be size except for inverse FOURIER, FOUR-WALSH, FOUR-HAAR
//             transform. These need 2*size length to supply real and
//             imaginary coefficients.
//        destVector-pointer to the vector of dest data, its length should be
//           size except for direct FOURIER, FOUR-WALSH, FOUR-HAAR. These
//           need 2*size length to store real and imaginary coefficients
//
////////////////////////////////////////////////////////////////////////////////
//
//Begin_Html <!--
/* -->
<div class=Section2>

<p class=MsoNormal><b><span style='font-size:14.0pt'>Example of zonal filtering</span></b></p>

<p class=MsoNormal><i>&nbsp;</i></p>

<p class=MsoNormal><i>Function:</i></p>

<p class=MsoNormal><b>void TSpectrumTransform::FilterZonal(const <a
href="http://root.cern.ch/root/html/ListOfTypes.html#Double_t">Double_t</a> *fSource,
<a href="http://root.cern.ch/root/html/ListOfTypes.html#Double_t">Double_t</a> *fDest)</b></p>

<p class=MsoNormal style='text-align:justify'>&nbsp;</p>

<p class=MsoNormal style='text-align:justify'>This function transforms the
source spectrum (for details see Transform function). Before the FilterZonal
function is called the class must be created by constructor and the type of the
transform as well as some other parameters must be set using a set of setter
functions. The FilterZonal function sets transformed coefficients in the given
region (fXmin, fXmax) to the given fFilterCoeff and transforms it back.
Filtered data are written into dest spectrum. </p>

<p class=MsoNormal style='text-align:justify'><i><span style='font-size:16.0pt'>&nbsp;</span></i></p>

<p class=MsoNormal style='text-align:justify'><i>Example – script Filter.c:</i></p>

<p class=MsoNormal style='text-align:justify'><i><span style='font-size:16.0pt'><img
border=0 width=601 height=402 src="gif/spectrumtransform_filter_image001.jpg"></span></i></p>

<p class=MsoNormal style='text-align:justify'><b>Fig. 1 Original spectrum
(black line) and filtered spectrum (red line) using Cosine transform and zonal
filtration (channels 2048-4095 were set to 0) </b></p>

<p class=MsoNormal><b><span style='color:#339966'>&nbsp;</span></b></p>

<p class=MsoNormal><b><span style='color:#339966'>Script:</span></b></p>

<p class=MsoNormal><span style='font-size:10.0pt'>// Example to illustrate
FilterZonal function (class TSpectrumTransform).</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>// To execute this example,
do</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>// root &gt; .x Filter.C</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>#include &lt;TSpectrum&gt;</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>#include
&lt;TSpectrumTransform&gt;</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>&nbsp;</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>void Filter() {</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   Int_t i;</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   Double_t nbins =
4096;</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   Double_t xmin  =
0;</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   Double_t xmax  =
(Double_t)nbins;</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   </span><span
style='font-size:10.0pt'>Double_t * source = new Double_t[nbins];</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t * dest = new
Double_t[nbins];   </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   TH1F *h = new
TH1F(&quot;h&quot;,&quot;Zonal filtering using Cosine
transform&quot;,nbins,xmin,xmax);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   TH1F *d = new
TH1F(&quot;d&quot;,&quot;&quot;,nbins,xmin,xmax);         </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   TFile *f = new
TFile(&quot;spectra\\TSpectrum.root&quot;);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   h=(TH1F*)
f-&gt;Get(&quot;transform1;1&quot;);   </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   for (i = 0; i &lt; nbins;
i++) source[i]=h-&gt;GetBinContent(i + 1);     </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   TCanvas *Transform1 =
gROOT-&gt;GetListOfCanvases()-&gt;FindObject(&quot;Transform1&quot;);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   if (!Transform1)
Transform1 = new
TCanvas(&quot;Transform&quot;,&quot;Transform1&quot;,10,10,1000,700);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>  
h-&gt;SetAxisRange(700,1024);   </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>  
h-&gt;Draw(&quot;L&quot;);   </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   TSpectrum *s = new
TSpectrum();</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   TSpectrumTransform *t =
new TSpectrumTransform(4096);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   </span><span lang=FR
style='font-size:10.0pt'>t-&gt;SetTransformType(t-&gt;kTransformCos,0);</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>  
t-&gt;SetRegion(2048, 4095);</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>  
t-&gt;FilterZonal(source,dest);     </span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   </span><span
style='font-size:10.0pt'>for (i = 0; i &lt; nbins; i++) d-&gt;SetBinContent(i +
1,dest[i]);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>  
d-&gt;SetLineColor(kRed);   </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   d-&gt;Draw(&quot;SAME
L&quot;);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>}</span></p>

</div>

<!-- */
// --> End_Html
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

//////////ENHANCE FUNCTION - CALCULATES DIFFERENT 1-D ORTHOGONAL TRANSFORMS, MULTIPLIES GIVEN REGION BY ENHANCE COEFFICIENT AND TRANSFORMS IT BACK//////
//___________________________________________________________________________
void TSpectrumTransform::Enhance(const Double_t *source, Double_t *destVector)
{
////////////////////////////////////////////////////////////////////////////////
//        ONE-DIMENSIONAL ENHANCE ZONAL FUNCTION
//        This function transforms the source spectrum. The calling program
//      should fill in input parameters. Then it multiplies transformed
//      coefficients in the given region (fXmin, fXmax) by the given
//      fEnhanceCoeff and transforms it back
//        Processed data are written into dest spectrum.
//
//        Function parameters:
//        source-pointer to the vector of source spectrum, its length should
//             be size except for inverse FOURIER, FOUR-WALSh, FOUR-HAAR
//             transform. These need 2*size length to supply real and
//             imaginary coefficients.
//        destVector-pointer to the vector of dest data, its length should be
//           size except for direct FOURIER, FOUR-WALSh, FOUR-HAAR. These
//           need 2*size length to store real and imaginary coefficients
//
////////////////////////////////////////////////////////////////////////////////
//Begin_Html <!--
/* -->
<div class=Section3>

<p class=MsoNormal><b><span style='font-size:14.0pt'>Example of enhancement</span></b></p>

<p class=MsoNormal><i>&nbsp;</i></p>

<p class=MsoNormal><i>Function:</i></p>

<p class=MsoNormal><b>void TSpectrumTransform::Enhance(const <a
href="http://root.cern.ch/root/html/ListOfTypes.html#double">double</a> *fSource,
<a href="http://root.cern.ch/root/html/ListOfTypes.html#double">double</a>
*fDest)</b></p>

<p class=MsoNormal><b>&nbsp;</b></p>

<p class=MsoNormal style='text-align:justify'>This function transforms the
source spectrum (for details see Transform function). Before the Enhance
function is called the class must be created by constructor and the type of the
transform as well as some other parameters must be set using a set of setter functions.
The Enhance function multiplies transformed coefficients in the given region
(fXmin, fXmax) by the given fEnhancCoeff and transforms it back. Enhanced data
are written into dest spectrum.</p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal style='text-align:justify'><i>Example  – script Enhance.c:</i></p>

<p class=MsoNormal style='text-align:justify'><i><span style='font-size:16.0pt'><img
border=0 width=601 height=402 src="gif/spectrumtransform_enhance_image001.jpg"></span></i></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:18.0pt'>&nbsp;</span></p>

<p class=MsoNormal style='text-align:justify'><b>Fig. 1 Original spectrum (black
line) and enhanced spectrum (red line) using Cosine transform (channels 0-1024
were multiplied by 2) </b></p>

<p class=MsoNormal><b><span style='color:#339966'>&nbsp;</span></b></p>

<p class=MsoNormal><b><span style='color:#339966'>Script:</span></b></p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal><span style='font-size:10.0pt'>// Example to illustrate
Enhance function (class TSpectrumTransform).</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>// To execute this example,
do</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>// root &gt; .x Enhance.C</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>&nbsp;</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>void Enhance() {</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   Int_t i;</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   Double_t nbins =
4096;</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   Double_t xmin  =
0;</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   Double_t xmax  =
(Double_t)nbins;</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   </span><span
style='font-size:10.0pt'>Double_t * source = new Double_t[nbins];</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t * dest = new
Double_t[nbins];   </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   TH1F *h = new
TH1F(&quot;h&quot;,&quot;Enhancement using Cosine transform&quot;,nbins,xmin,xmax);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   TH1F *d = new
TH1F(&quot;d&quot;,&quot;&quot;,nbins,xmin,xmax);         </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   TFile *f = new
TFile(&quot;spectra\\TSpectrum.root&quot;);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   h=(TH1F*)
f-&gt;Get(&quot;transform1;1&quot;);   </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   for (i = 0; i &lt; nbins;
i++) source[i]=h-&gt;GetBinContent(i + 1);     </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   TCanvas *Transform1 = gROOT-&gt;GetListOfCanvases()-&gt;FindObject(&quot;Transform1&quot;);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   if (!Transform1)
Transform1 = new
TCanvas(&quot;Transform&quot;,&quot;Transform1&quot;,10,10,1000,700);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>  
h-&gt;SetAxisRange(700,1024);   </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>  
h-&gt;Draw(&quot;L&quot;);   </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   TSpectrum *s = new
TSpectrum();</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   TSpectrumTransform *t =
new TSpectrumTransform(4096);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   </span><span lang=FR
style='font-size:10.0pt'>t-&gt;SetTransformType(t-&gt;kTransformCos,0);</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   t-&gt;SetRegion(0,
1024);</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>  
t-&gt;SetEnhanceCoeff(2);</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>  
t-&gt;Enhance(source,dest);        </span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   </span><span
style='font-size:10.0pt'>for (i = 0; i &lt; nbins; i++) d-&gt;SetBinContent(i +
1,dest[i]);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>  
d-&gt;SetLineColor(kRed);   </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   d-&gt;Draw(&quot;SAME
L&quot;);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>}</span></p>

</div>

<!-- */
// --> End_Html
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

//___________________________________________________________________________
void TSpectrumTransform::SetTransformType(Int_t transType, Int_t degree)
{
//////////////////////////////////////////////////////////////////////////////
//   SETTER FUNCION
//
//   This function sets the following parameters for transform:
//         -transType - type of transform (Haar, Walsh, Cosine, Sine, Fourier, Hartley, Fourier-Walsh, Fourier-Haar, Walsh-Haar, Cosine-Walsh, Cosine-Haar, Sine-Walsh, Sine-Haar)
//         -degree - degree of mixed transform, applies only for Fourier-Walsh, Fourier-Haar, Walsh-Haar, Cosine-Walsh, Cosine-Haar, Sine-Walsh, Sine-Haar transforms
//////////////////////////////////////////////////////////////////////////////
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


//___________________________________________________________________________
void TSpectrumTransform::SetRegion(Int_t xmin, Int_t xmax)
{
//////////////////////////////////////////////////////////////////////////////
//   SETTER FUNCION
//
//   This function sets the filtering or enhancement region:
//         -xmin, xmax
//////////////////////////////////////////////////////////////////////////////
   if(xmin<0 || xmax < xmin || xmax >= fSize){
      Error("TSpectrumTransform", "Wrong range");
      return;
   }
   fXmin = xmin;
   fXmax = xmax;
}

//___________________________________________________________________________
void TSpectrumTransform::SetDirection(Int_t direction)
{
//////////////////////////////////////////////////////////////////////////////
//   SETTER FUNCION
//
//   This function sets the direction of the transform:
//         -direction (forward or inverse)
//////////////////////////////////////////////////////////////////////////////
   if(direction != kTransformForward && direction != kTransformInverse){
      Error("TSpectrumTransform", "Wrong direction");
      return;
   }
   fDirection = direction;
}

//___________________________________________________________________________
void TSpectrumTransform::SetFilterCoeff(Double_t filterCoeff)
{
//////////////////////////////////////////////////////////////////////////////
//   SETTER FUNCION
//
//   This function sets the filter coefficient:
//         -filterCoeff - after the transform the filtered region (xmin, xmax) is replaced by this coefficient. Applies only for filtereng operation.
//////////////////////////////////////////////////////////////////////////////
   fFilterCoeff = filterCoeff;
}

//___________________________________________________________________________
void TSpectrumTransform::SetEnhanceCoeff(Double_t enhanceCoeff)
{
//////////////////////////////////////////////////////////////////////////////
//   SETTER FUNCION
//
//   This function sets the enhancement coefficient:
//         -enhanceCoeff - after the transform the enhanced region (xmin, xmax) is multiplied by this coefficient. Applies only for enhancement operation.
//////////////////////////////////////////////////////////////////////////////
   fEnhanceCoeff = enhanceCoeff;
}
