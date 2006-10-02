// @(#)root/spectrum:$Name:  $:$Id: TSpectrumTransform.cxx,v 1.2 2006/09/29 15:51:52 brun Exp $
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
void TSpectrumTransform::Haar(float *working_space, int num, int direction) 
{   
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCION                                                          //
//                                                                              //
//   This funcion calculates Haar transform of a part of data                   //
//      Function parameters:                                                    //
//              -working_space-pointer to vector of transformed data            //
//              -num-length of processed data                                   //
//              -direction-forward or inverse transform                         //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   int i, ii, li, l2, l3, j, jj, jj1, lj, iter, m, jmin, jmax;
   double a, b, c, wlk;
   float val;
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
         l2 = (int) TMath::Power(2, li - 1);
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
      jmin = (int) TMath::Power(2, i);
      jmax = (int) TMath::Power(2, ii) - 1;
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
         li = (int) c;
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
void TSpectrumTransform::Walsh(float *working_space, int num) 
{
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCION                                                          //
//                                                                              //
//   This funcion calculates Walsh transform of a part of data                  //
//      Function parameters:                                                    //
//              -working_space-pointer to vector of transformed data            //
//              -num-length of processed data                                   //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   int i, m, nump = 1, mnum, mnum2, mp, ib, mp2, mnum21, iba, iter;
   double a;
   float val1, val2;
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
void TSpectrumTransform::BitReverse(float *working_space, int num) 
{   
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCION                                                          //
//                                                                              //
//   This funcion carries out bir-reverse reordering of data                    //
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
void TSpectrumTransform::Fourier(float *working_space, int num, int hartley,
                          int direction, int zt_clear) 
{   
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCION                                                          //
//                                                                              //
//   This funcion calculates Fourier based transform of a part of data          //
//      Function parameters:                                                    //
//              -working_space-pointer to vector of transformed data            //
//              -num-length of processed data                                   //
//              -hartley-1 if it is Hartley transform, 0 othewise               //
//              -direction-forward or inverse transform                         //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   int nxp2, nxp, i, j, k, m, iter, mxp, j1, j2, n1, n2, it;
   double a, b, c, d, sign, wpwr, arg, wr, wi, tr, ti, pi =
       3.14159265358979323846;
   float val1, val2, val3, val4;
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
void TSpectrumTransform::BitReverseHaar(float *working_space, int shift, int num,
                                 int start) 
{
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCION                                                          //
//                                                                              //
//   This funcion carries out bir-reverse reordering for Haar transform         //
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
int TSpectrumTransform::GeneralExe(float *working_space, int zt_clear, int num,
                            int degree, int type) 
{
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCION                                                          //
//                                                                              //
//   This funcion calculates generalized (mixed) transforms of different degrees//
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
   double a, b, c, d, wpwr, arg, wr, wi, tr, ti, pi =
       3.14159265358979323846;
   float val1, val2, val3, val4, a0oldr = 0, b0oldr = 0, a0r, b0r;
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
int TSpectrumTransform::GeneralInv(float *working_space, int num, int degree,
                            int type) 
{
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCION                                                          //
//                                                                              //
//   This funcion calculates inverse generalized (mixed) transforms             //
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
   double a, b, c, d, wpwr, arg, wr, wi, tr, ti, pi =
       3.14159265358979323846;
   float val1, val2, val3, val4, a0oldr = 0, b0oldr = 0, a0r, b0r;
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
void TSpectrumTransform::Transform(const float *source, float *destVector)
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
<div class=3DSection1>

<p class=3DMsoNormal><b><span lang=3DEN-US style=3D'font-size:14.0pt'>Trans=
form
methods</span></b><span lang=3DEN-US style=3D'font-size:14.0pt'><o:p></o:p>=
</span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><i><span lang=3DEN-US><o:=
p>&nbsp;</o:p></span></i></p>

<p class=3DMsoNormal style=3D'text-align:justify'><i><span lang=3DEN-US>Goa=
l: to
analyze experimental data using orthogonal transforms<o:p></o:p></span></i>=
</p>

<p class=3DMsoNormal style=3D'margin-left:.5in;text-align:justify;text-inde=
nt:-.25in;
mso-list:l10 level1 lfo10;tab-stops:list .5in'><![if !supportLists]><span
lang=3DEN-US><span style=3D'mso-list:Ignore'>&#8226;<span style=3D'font:7.0=
pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span></span></span><![endif]><span lang=3DEN-US>orthogonal transforms can=
 be
successfully used for the processing of nuclear spectra (not only) </span><=
/p>

<p class=3DMsoNormal style=3D'margin-left:.5in;text-align:justify;text-inde=
nt:-.25in;
mso-list:l10 level1 lfo10;tab-stops:list .5in'><![if !supportLists]><span
lang=3DEN-US><span style=3D'mso-list:Ignore'>&#8226;<span style=3D'font:7.0=
pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span></span></span><![endif]><span class=3DGramE><span lang=3DEN-US>they<=
/span></span><span
lang=3DEN-US> can be used to remove high frequency noise, to increase
signal-to-background ratio as well as to enhance low intensity components [=
1],
to carry out e.g. Fourier analysis etc. </span></p>

<p class=3DMsoNormal style=3D'margin-left:.5in;text-align:justify;text-inde=
nt:-.25in;
mso-list:l10 level1 lfo10;tab-stops:list .5in'><![if !supportLists]><span
lang=3DEN-US><span style=3D'mso-list:Ignore'>&#8226;<span style=3D'font:7.0=
pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span></span></span><![endif]><span lang=3DEN-US>we have implemented the
function for the calculation of the commonly used orthogonal transforms as =
well
as functions for the filtration and enhancement of experimental data</span>=
</p>

<p class=3DMsoNormal><i><span lang=3DEN-US><o:p>&nbsp;</o:p></span></i></p>

<p class=3DMsoNormal><i><span lang=3DEN-US>Function:</span></i></p>

<p class=3DMsoNormal><span class=3DGramE><b style=3D'mso-bidi-font-weight:n=
ormal'><span
lang=3DEN-US>void</span></b></span><b style=3D'mso-bidi-font-weight:normal'=
><span
lang=3DEN-US> <span class=3DSpellE>TSpectrumTransform::Transform</span>(con=
st <a
href=3D"http://root.cern.ch/root/html/ListOfTypes.html#float">float</a> *<s=
pan
class=3DSpellE>fSource</span>, <a
href=3D"http://root.cern.ch/root/html/ListOfTypes.html#float">float</a> *<s=
pan
class=3DSpellE>fDest</span>)<o:p></o:p></span></b></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US><o:p>&=
nbsp;</o:p></span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US>This f=
unction
transforms the source spectrum according to the given input parameters.
Transformed data are written into <span class=3DSpellE>dest</span> spectrum=
. Before
the Transform function is called the class must be created by constructor a=
nd
the type of the transform as well as some other parameters must be set usin=
g a
set of setter functions.</span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US><o:p>&=
nbsp;</o:p></span></p>

<p class=3DMsoNormal><i style=3D'mso-bidi-font-style:normal'><span lang=3DE=
N-US
style=3D'color:red'>Member variables of <span class=3DSpellE>TSpectrumTrans=
form</span>
class:<o:p></o:p></span></i></p>

<p class=3DMsoNormal style=3D'margin-left:25.65pt;text-align:justify;tab-st=
ops:
14.2pt'><span lang=3DEN-US><span style=3D'mso-spacerun:yes'>&nbsp;</span><s=
pan
class=3DSpellE><span class=3DGramE><b style=3D'mso-bidi-font-weight:normal'=
>fSource</b></span></span><span
class=3DGramE>-pointer</span> to the vector of source spectrum. Its length =
should
be equal to the &#8220;<span class=3DSpellE>fSize</span>&#8221; parameter e=
xcept
for inverse FOURIER, FOUR-WALSH, FOUR-HAAR transforms. These need &#8220;2*=
<span
class=3DSpellE>fSize</span>&#8221; length to supply real and imaginary
coefficients. <span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span></span><=
/p>

<p class=3DMsoNormal style=3D'margin-left:25.65pt;text-align:justify;tab-st=
ops:
14.2pt'><span class=3DSpellE><span class=3DGramE><b style=3D'mso-bidi-font-=
weight:
normal'><span lang=3DEN-US>fDest</span></b></span></span><span class=3DGram=
E><span
lang=3DEN-US>-pointer</span></span><span lang=3DEN-US> to the vector of des=
tination
spectrum. Its length should be equal to the &#8220;<span class=3DSpellE>fSi=
ze</span>&#8221;
parameter except for inverse FOURIER, FOUR-WALSH, FOUR-HAAR transforms. The=
se
need &#8220;2*<span class=3DSpellE>fSize</span>&#8221; length to store real=
 and
imaginary coefficients. </span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; </spa=
n><span
class=3DSpellE><span class=3DGramE><b style=3D'mso-bidi-font-weight:normal'=
>fSize</b></span></span><span
class=3DGramE>-basic</span> length of the source and <span class=3DSpellE>d=
est</span>
spectrum. <span style=3D'color:fuchsia'>It should be power of 2.<o:p></o:p>=
</span></span></p>

<p class=3DMsoNormal style=3D'margin-left:25.65pt;text-align:justify;text-i=
ndent:
-2.85pt'><span class=3DSpellE><span class=3DGramE><b style=3D'mso-bidi-font=
-weight:
normal'><span lang=3DEN-US>fType</span></b></span></span><span class=3DGram=
E><span
lang=3DEN-US>-type</span></span><span lang=3DEN-US> of transform</span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US><span
style=3D'mso-tab-count:1'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&=
nbsp;&nbsp;&nbsp; </span>Classic
transforms:</span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US><span
style=3D'mso-tab-count:2'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&=
nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbs=
p;&nbsp;&nbsp; </span><span
class=3DSpellE><span class=3DGramE>kTransformHaar</span></span> </span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US><span
style=3D'mso-tab-count:2'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&=
nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbs=
p;&nbsp;&nbsp; </span><span
class=3DSpellE><span class=3DGramE>kTransformWalsh</span></span> </span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US><span
style=3D'mso-tab-count:2'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&=
nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbs=
p;&nbsp;&nbsp; </span><span
class=3DSpellE><span class=3DGramE>kTransformCos</span></span> </span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US><span
style=3D'mso-tab-count:2'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&=
nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbs=
p;&nbsp;&nbsp; </span><span
class=3DSpellE><span class=3DGramE>kTransformSin</span></span> </span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US><span
style=3D'mso-tab-count:2'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&=
nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbs=
p;&nbsp;&nbsp; </span><span
class=3DSpellE><span class=3DGramE>kTransformFourier</span></span> </span><=
/p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US><span
style=3D'mso-tab-count:2'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&=
nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbs=
p;&nbsp;&nbsp; </span><span
class=3DSpellE><span class=3DGramE>kTransformHartey</span></span> </span></=
p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US><span
style=3D'mso-tab-count:1'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&=
nbsp;&nbsp;&nbsp; </span>Mixed
transforms:</span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US><span
style=3D'mso-tab-count:2'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&=
nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbs=
p;&nbsp;&nbsp; </span><span
class=3DSpellE><span class=3DGramE>kTransformFourierWalsh</span></span> </s=
pan></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US><span
style=3D'mso-tab-count:2'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&=
nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbs=
p;&nbsp;&nbsp; </span><span
class=3DSpellE><span class=3DGramE>kTransformFourierHaar</span></span> </sp=
an></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US><span
style=3D'mso-tab-count:2'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&=
nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbs=
p;&nbsp;&nbsp; </span><span
class=3DSpellE><span class=3DGramE>kTransformWalshHaar</span></span> </span=
></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US><span
style=3D'mso-tab-count:2'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&=
nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbs=
p;&nbsp;&nbsp; </span><span
class=3DSpellE><span class=3DGramE>kTransformCosWalsh</span></span> </span>=
</p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US><span
style=3D'mso-tab-count:2'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&=
nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbs=
p;&nbsp;&nbsp; </span><span
class=3DSpellE><span class=3DGramE>kTransformCosHaar</span></span> </span><=
/p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US><span
style=3D'mso-tab-count:2'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&=
nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbs=
p;&nbsp;&nbsp; </span><span
class=3DSpellE><span class=3DGramE>kTransformSinWalsh</span></span> <b><o:p=
></o:p></b></span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US><span
style=3D'mso-tab-count:2'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&=
nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbs=
p;&nbsp;&nbsp; </span><span
class=3DSpellE><span class=3DGramE>kTransformSinHaar</span></span> </span><=
/p>

<p class=3DMsoNormal style=3D'text-align:justify;text-indent:22.8pt'><span
class=3DSpellE><span class=3DGramE><b style=3D'mso-bidi-font-weight:normal'=
><span
lang=3DEN-US>fDirection</span></b></span></span><span class=3DGramE><span
lang=3DEN-US>-direction-transform</span></span><span lang=3DEN-US> direction
(forward, inverse)</span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US><span
style=3D'mso-tab-count:2'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&=
nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbs=
p;&nbsp;&nbsp; </span><span
class=3DSpellE><span class=3DGramE>kTransformForward</span></span> </span><=
/p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US><span
style=3D'mso-tab-count:2'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&=
nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbs=
p;&nbsp;&nbsp; </span><span
class=3DSpellE><span class=3DGramE>kTransformInverse</span></span> </span><=
/p>

<p class=3DMsoNormal style=3D'text-align:justify;text-indent:22.8pt'><span
class=3DSpellE><span class=3DGramE><b style=3D'mso-bidi-font-weight:normal'=
><span
lang=3DEN-US>fDegree</span></b></span></span><span class=3DGramE><span lang=
=3DEN-US>-</span></span><span
lang=3DEN-US>applies only for mixed transforms [2], [3], [4]. </span></p>

<p class=3DMsoNormal style=3D'text-align:justify;text-indent:22.8pt'><span
lang=3DEN-US><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span><span style=3D'color:fuchsia'><span
style=3D'mso-spacerun:yes'>&nbsp;</span>Allowed <span class=3DGramE>range<s=
pan
style=3D'mso-spacerun:yes'>&nbsp; </span></span><sub><!--[if gte vml 1]><v:=
shapetype
 id=3D"_x0000_t75" coordsize=3D"21600,21600" o:spt=3D"75" o:preferrelative=
=3D"t"
 path=3D"m@4@5l@4@11@9@11@9@5xe" filled=3D"f" stroked=3D"f">
 <v:stroke joinstyle=3D"miter"/>
 <v:formulas>
  <v:f eqn=3D"if lineDrawn pixelLineWidth 0"/>
  <v:f eqn=3D"sum @0 1 0"/>
  <v:f eqn=3D"sum 0 0 @1"/>
  <v:f eqn=3D"prod @2 1 2"/>
  <v:f eqn=3D"prod @3 21600 pixelWidth"/>
  <v:f eqn=3D"prod @3 21600 pixelHeight"/>
  <v:f eqn=3D"sum @0 0 1"/>
  <v:f eqn=3D"prod @6 1 2"/>
  <v:f eqn=3D"prod @7 21600 pixelWidth"/>
  <v:f eqn=3D"sum @8 21600 0"/>
  <v:f eqn=3D"prod @7 21600 pixelHeight"/>
  <v:f eqn=3D"sum @10 21600 0"/>
 </v:formulas>
 <v:path o:extrusionok=3D"f" gradientshapeok=3D"t" o:connecttype=3D"rect"/>
 <o:lock v:ext=3D"edit" aspectratio=3D"t"/>
</v:shapetype><v:shape id=3D"_x0000_i1026" type=3D"#_x0000_t75" style=3D'wi=
dth:75pt;
 height:20.25pt' o:ole=3D"">
 <v:imagedata src=3D"Transform_files/image001.wmz" o:title=3D""/>
</v:shape><![endif]--><![if !vml]><img border=3D0 width=3D100 height=3D27
src=3D"Transform_files/image002.gif" v:shapes=3D"_x0000_i1026"><![endif]></=
sub><!--[if gte mso 9]><xml>
 <o:OLEObject Type=3D"Embed" ProgID=3D"Equation.DSMT4" ShapeID=3D"_x0000_i1=
026"
  DrawAspect=3D"Content" ObjectID=3D"_1220804604">
 </o:OLEObject>
</xml><![endif]-->. <o:p></o:p></span></span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><b><i><span lang=3DEN-US>=
References:<o:p></o:p></span></i></b></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US>[1] C.=
V.
Hampton, B. <span class=3DSpellE>Lian</span>, Wm. C. <span class=3DSpellE>M=
cHarris</span>:
Fast-Fourier-transform spectral enhancement techniques for gamma-ray
spectroscopy. NIM A353 (1994) 280-284. </span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US>[2] <s=
pan
class=3DSpellE>Morh&aacute;&#269;</span> M., <span class=3DSpellE>Matou&#35=
3;ek</span>
V., New adaptive Cosine-<span class=3DGramE>Walsh<span
style=3D'mso-spacerun:yes'>&nbsp; </span>transform</span> and its applicati=
on to
nuclear data compression, IEEE Transactions on Signal Processing 48 (2000)
2693.<span style=3D'mso-spacerun:yes'>&nbsp; </span></span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US>[3] <s=
pan
class=3DSpellE>Morh&aacute;&#269;</span> M., <span class=3DSpellE>Matou&#35=
3;ek</span>
V., Data compression using new fast adaptive Cosine-<span class=3DSpellE>Ha=
ar</span>
transforms, Digital Signal Processing 8 (1998) 63. </span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US>[4] <s=
pan
class=3DSpellE>Morh&aacute;&#269;</span> M., <span class=3DSpellE>Matou&#35=
3;ek</span>
V.: Multidimensional nuclear data <span class=3DGramE>compression using fast
adaptive Walsh-<span class=3DSpellE>Haar</span> transform</span>. <span
class=3DSpellE><span class=3DGramE>Acta</span></span><span class=3DGramE> <=
span
class=3DSpellE>Physica</span> <span class=3DSpellE>Slovaca</span> 51 (2001)=
 307.</span>
</span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US><o:p>&=
nbsp;</o:p></span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span class=3DGramE><i><s=
pan
lang=3DEN-US>Example <span style=3D'mso-spacerun:yes'>&nbsp;</span>&#8211;<=
/span></i></span><i><span
lang=3DEN-US> script <span class=3DSpellE>Transform.c</span>:<o:p></o:p></s=
pan></i></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US
style=3D'font-size:18.0pt'><!--[if gte vml 1]><v:shape id=3D"_x0000_s1040" =
type=3D"#_x0000_t75"
 style=3D'width:450.3pt;height:242.75pt;mso-position-horizontal-relative:ch=
ar;
 mso-position-vertical-relative:line'>
 <v:imagedata src=3D"Transform_files/image003.png" o:title=3D"Transform1"/>
 <w:wrap type=3D"none"/>
 <w:anchorlock/>
</v:shape><![endif]--><![if !vml]><img width=3D600 height=3D324
src=3D"Transform_files/image004.jpg" v:shapes=3D"_x0000_s1040"><![endif]><o=
:p></o:p></span></p>

<p class=3DMsoNormal><b><span lang=3DEN-US>Fig. 1 Original gamma-ray spectr=
um<o:p></o:p></span></b></p>

<p class=3DMsoNormal><b><span lang=3DEN-US style=3D'font-size:14.0pt'><!--[=
if gte vml 1]><v:shape
 id=3D"_x0000_i1027" type=3D"#_x0000_t75" style=3D'width:450.75pt;height:30=
1.5pt'>
 <v:imagedata src=3D"Transform_files/image005.jpg" o:title=3D"Transform"/>
</v:shape><![endif]--><![if !vml]><img border=3D0 width=3D601 height=3D402
src=3D"Transform_files/image006.jpg" v:shapes=3D"_x0000_i1027"><![endif]><o=
:p></o:p></span></b></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US
style=3D'font-size:18.0pt'><o:p>&nbsp;</o:p></span></p>

<p class=3DMsoNormal><b><span lang=3DEN-US>Fig. 2 Transformed <span class=
=3DGramE>spectrum
from Fig. 1 using Cosine transform</span></span></b></p>

<p class=3DMsoNormal><b style=3D'mso-bidi-font-weight:normal'><span lang=3D=
EN-US
style=3D'font-size:16.0pt;color:#339966'><o:p>&nbsp;</o:p></span></b></p>

<p class=3DMsoNormal><b style=3D'mso-bidi-font-weight:normal'><span lang=3D=
EN-US
style=3D'color:#339966'>Script:<o:p></o:p></span></b></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'>// Examp=
le to
illustrate Transform function (class <span class=3DSpellE>TSpectrumTransfor=
m</span>).<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'>// <span
class=3DGramE>To</span> execute this example, do<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'>// root =
&gt; .x <span
class=3DSpellE>Transform.C</span><o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'>#include=
 &lt;<span
class=3DSpellE>TSpectrum</span>&gt;<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'>#include=
 &lt;<span
class=3DSpellE>TSpectrumTransform</span>&gt;<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><o:p>&nb=
sp;</o:p></span></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'>void Transform() {<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp; </span><span
style=3D'mso-spacerun:yes'>&nbsp;</span>Int_t i;<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span>Double_t nbins =3D=
 4096;<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span>Double_t xmin<span
style=3D'mso-spacerun:yes'>&nbsp; </span>=3D 0;<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span>Double_t xmax<span
style=3D'mso-spacerun:yes'>&nbsp; </span>=3D (Double_t)nbins;<o:p></o:p></s=
pan></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span></span><span
class=3DSpellE><span lang=3DEN-US style=3D'font-size:10.0pt'>Float_t</span>=
</span><span
lang=3DEN-US style=3D'font-size:10.0pt'> * source =3D new <span class=3DGra=
mE>float[</span><span
class=3DSpellE>nbins</span>];<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><span class=3DSpellE>Float_t=
</span>
* <span class=3DSpellE>dest</span> =3D new <span class=3DGramE>float[</span=
><span
class=3DSpellE>nbins</span>];<span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; =
</span><o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span>TH1F *h =3D new <span class=
=3DGramE>TH1F(</span>&quot;<span
class=3DSpellE>h&quot;,&quot;Transformed</span> spectrum using Cosine <span
class=3DSpellE>transform&quot;,nbins,xmin,xmax</span>);<o:p></o:p></span></=
p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><span class=3DSpellE>TFile</=
span> *f
=3D new <span class=3DSpellE><span class=3DGramE>TFile</span></span><span
class=3DGramE>(</span>&quot;spectra\\<span class=3DSpellE>TSpectrum.root</s=
pan>&quot;);<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span>h<span class=3DGramE>=3D(</s=
pan>TH1F*)
f-&gt;Get(&quot;transform1;1&quot;);<span style=3D'mso-spacerun:yes'>&nbsp;=
&nbsp;
</span><o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><span class=3DGramE>for</spa=
n> (<span
class=3DSpellE>i</span> =3D 0; <span class=3DSpellE>i</span> &lt; <span cla=
ss=3DSpellE>nbins</span>;
<span class=3DSpellE>i</span>++) source[<span class=3DSpellE>i</span>]=3Dh-=
&gt;<span
class=3DSpellE>GetBinContent</span>(<span class=3DSpellE>i</span> + 1);<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span><o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><span class=3DSpellE>TCanvas=
</span>
*Transform1 =3D <span class=3DSpellE>gROOT</span>-&gt;<span class=3DSpellE>=
<span
class=3DGramE>GetListOfCanvases</span></span><span class=3DGramE>(</span>)-=
&gt;<span
class=3DSpellE>FindObject</span>(&quot;Transform1&quot;);<o:p></o:p></span>=
</p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><span class=3DGramE>if</span>
(!Transform1) Transform1 =3D new <span class=3DSpellE>TCanvas</span>(&quot;=
Transform&quot;,&quot;Transform1&quot;,10,10,1000,700);<o:p></o:p></span></=
p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><span class=3DSpellE>TSpectr=
um</span>
*s =3D new <span class=3DSpellE><span class=3DGramE>TSpectrum</span></span>=
<span
class=3DGramE>(</span>);<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><span class=3DSpellE>TSpectr=
umTransform</span>
*t =3D new <span class=3DSpellE><span class=3DGramE>TSpectrumTransform</spa=
n></span><span
class=3DGramE>(</span>4096);<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span></span><span lang=3DFR
style=3D'font-size:10.0pt;mso-ansi-language:FR'>t-&gt;SetTransformType(t-&g=
t;kTransformCos,0);<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp;
</span>t-&gt;SetDirection(t-&gt;kTransformForward);<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span></span><span
class=3DGramE><span lang=3DEN-US style=3D'font-size:10.0pt'>t</span></span>=
<span
lang=3DEN-US style=3D'font-size:10.0pt'>-&gt;Transform(<span class=3DSpellE=
>source,dest</span>);<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><span class=3DGramE>for</spa=
n> (<span
class=3DSpellE>i</span> =3D 0; <span class=3DSpellE>i</span> &lt; <span cla=
ss=3DSpellE>nbins</span>;
<span class=3DSpellE>i</span>++) h-&gt;<span class=3DSpellE>SetBinContent</=
span>(<span
class=3DSpellE>i</span> + 1,dest[<span class=3DSpellE>i</span>]);<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><span class=3DGramE>h</span>=
-&gt;<span
class=3DSpellE>SetLineColor</span>(<span class=3DSpellE>kRed</span>);<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; </span><o:p></o:p=
></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><span class=3DGramE>h</span>=
-&gt;Draw(&quot;L&quot;);<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'>}<o:p></=
o:p></span></p>

</div>

<!-- */
// --> End_Html
   int i, j=0, k = 1, m, l;
   float val;
   double a, b, pi = 3.14159265358979323846;
   float *working_space = 0;
   if (fTransformType >= kTransformFourierWalsh && fTransformType <= kTransformSinHaar) {
      if (fTransformType >= kTransformCosWalsh)
         fDegree += 1;
      k = (int) TMath::Power(2, fDegree);
      j = fSize / k;
   }
   switch (fTransformType) {
   case kTransformHaar:
   case kTransformWalsh:
      working_space = new float[2 * fSize];
      break;
   case kTransformCos:
   case kTransformSin:
   case kTransformFourier:
   case kTransformHartley:
   case kTransformFourierWalsh:
   case kTransformFourierHaar:
   case kTransformWalshHaar:
      working_space = new float[4 * fSize];
      break;
   case kTransformCosWalsh:
   case kTransformCosHaar:
   case kTransformSinWalsh:
   case kTransformSinHaar:
      working_space = new float[8 * fSize];
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
            a = pi * (double) i / (double) fSize;
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
            a = pi * (double) i / (double) fSize;
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
               j = (int) TMath::Power(2, fDegree) / 2;
               k = i / j;
               k = 2 * k * j;
               working_space[k + i % j] = val;
               working_space[k + 2 * j - 1 - i % j] = val;
            }
            
            else if (fTransformType == kTransformSinWalsh
                     || fTransformType == kTransformSinHaar) {
               j = (int) TMath::Power(2, fDegree) / 2;
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
            m = (int) TMath::Power(2, fDegree);
            l = 2 * fSize / m;
            for (i = 0; i < l; i++)
               BitReverseHaar(working_space, 2 * fSize, m, i * m);
            GeneralExe(working_space, 0, 2 * fSize, fDegree, fTransformType);
            for (i = 0; i < fSize; i++) {
               k = i / j;
               k = 2 * k * j;
               a = pi * (double) (i % j) / (double) (2 * j);
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
            m = (int) TMath::Power(2, fDegree);
            l = 2 * fSize / m;
            for (i = 0; i < l; i++)
               BitReverseHaar(working_space, 2 * fSize, m, i * m);
            GeneralExe(working_space, 0, 2 * fSize, fDegree, fTransformType);
            for (i = 0; i < fSize; i++) {
               k = i / j;
               k = 2 * k * j;
               a = pi * (double) (i % j) / (double) (2 * j);
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
            k = (int) TMath::Power(2, fDegree - 1);
         
         else
            k = (int) TMath::Power(2, fDegree);
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
            a = pi * (double) i / (double) fSize;
            b = TMath::Sin(a);
            a = TMath::Cos(a);
            working_space[i + fSize] = (double) working_space[i] * b;
            working_space[i] = (double) working_space[i] * a;
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
            a = pi * (double) i / (double) fSize;
            working_space[i + fSize] =
                -(double) working_space[i - 1] * TMath::Cos(a);
            working_space[i] =
                (double) working_space[i - 1] * TMath::Sin(a);
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
            k = (int) TMath::Power(2, fDegree - 1);
         
         else
            k = (int) TMath::Power(2, fDegree);
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
            j = (int) TMath::Power(2, fDegree) / 2;
            m = (int) TMath::Power(2, fDegree);
            l = 2 * fSize / m;
            for (i = 0; i < fSize; i++) {
               k = i / j;
               k = 2 * k * j;
               a = pi * (double) (i % j) / (double) (2 * j);
               if (i % j == 0) {
                  working_space[2 * fSize + k + i % j] =
                      working_space[i] * TMath::Sqrt(2.0);
                  working_space[4 * fSize + 2 * fSize + k + i % j] = 0;
               }
               
               else {
                  b = TMath::Sin(a);
                  a = TMath::Cos(a);
                  working_space[4 * fSize + 2 * fSize + k + i % j] =
                      -(double) working_space[i] * b;
                  working_space[2 * fSize + k + i % j] =
                      (double) working_space[i] * a;
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
            m = (int) TMath::Power(2, fDegree);
            l = 2 * fSize / m;
            for (i = 0; i < l; i++)
               BitReverseHaar(working_space, 2 * fSize, m, i * m);
         }
         
         else if (fTransformType == kTransformSinWalsh
                  || fTransformType == kTransformSinHaar) {
            j = (int) TMath::Power(2, fDegree) / 2;
            m = (int) TMath::Power(2, fDegree);
            l = 2 * fSize / m;
            for (i = 0; i < fSize; i++) {
               k = i / j;
               k = 2 * k * j;
               a = pi * (double) (i % j) / (double) (2 * j);
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
                      -(double) working_space[j + k / 2 - i % j - 1] * b;
                  working_space[2 * fSize + k + j + i % j] =
                      (double) working_space[j + k / 2 - i % j - 1] * a;
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
void TSpectrumTransform::FilterZonal(const float *source, float *destVector)
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
<div class=3DSection2>

<p class=3DMsoNormal><b><span lang=3DEN-US style=3D'font-size:14.0pt'>Examp=
le of
zonal filtering</span></b><span lang=3DEN-US style=3D'font-size:14.0pt'><o:=
p></o:p></span></p>

<p class=3DMsoNormal><i><span lang=3DEN-US><o:p>&nbsp;</o:p></span></i></p>

<p class=3DMsoNormal><i><span lang=3DEN-US>Function:</span></i></p>

<p class=3DMsoNormal><span class=3DGramE><b style=3D'mso-bidi-font-weight:n=
ormal'><span
lang=3DEN-US>void</span></b></span><b style=3D'mso-bidi-font-weight:normal'=
><span
lang=3DEN-US> <span class=3DSpellE>TSpectrumTransform::FilterZonal</span>(c=
onst <a
href=3D"http://root.cern.ch/root/html/ListOfTypes.html#float">float</a> *<s=
pan
class=3DSpellE>fSource</span>, <a
href=3D"http://root.cern.ch/root/html/ListOfTypes.html#float">float</a> *<s=
pan
class=3DSpellE>fDest</span>)<o:p></o:p></span></b></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US><o:p>&=
nbsp;</o:p></span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US>This f=
unction
transforms the source spectrum (for details see Transform function). Before=
 the
<span class=3DSpellE>FilterZonal</span> function is called the class must be
created by constructor and the type of the transform as well as some other
parameters must be set using a set of setter <span class=3DSpellE>funcions<=
/span>.
The <span class=3DSpellE>FilterZonal</span> function sets transformed
coefficients in the given region (<span class=3DSpellE>fXmin</span>, <span
class=3DSpellE>fXmax</span>) to the given <span class=3DSpellE>fFilterCoeff=
</span>
and transforms it back. Filtered data are written into <span class=3DSpellE=
>dest</span>
spectrum. </span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><i><span lang=3DEN-US
style=3D'font-size:16.0pt'><o:p>&nbsp;</o:p></span></i></p>

<p class=3DMsoNormal style=3D'text-align:justify'><i><span lang=3DEN-US>Exa=
mple
&#8211; script <span class=3DSpellE>Filter.c</span>:<o:p></o:p></span></i><=
/p>

<p class=3DMsoNormal style=3D'text-align:justify'><i><span lang=3DEN-US
style=3D'font-size:16.0pt'><!--[if gte vml 1]><v:shapetype id=3D"_x0000_t75"
 coordsize=3D"21600,21600" o:spt=3D"75" o:preferrelative=3D"t" path=3D"m@4@=
5l@4@11@9@11@9@5xe"
 filled=3D"f" stroked=3D"f">
 <v:stroke joinstyle=3D"miter"/>
 <v:formulas>
  <v:f eqn=3D"if lineDrawn pixelLineWidth 0"/>
  <v:f eqn=3D"sum @0 1 0"/>
  <v:f eqn=3D"sum 0 0 @1"/>
  <v:f eqn=3D"prod @2 1 2"/>
  <v:f eqn=3D"prod @3 21600 pixelWidth"/>
  <v:f eqn=3D"prod @3 21600 pixelHeight"/>
  <v:f eqn=3D"sum @0 0 1"/>
  <v:f eqn=3D"prod @6 1 2"/>
  <v:f eqn=3D"prod @7 21600 pixelWidth"/>
  <v:f eqn=3D"sum @8 21600 0"/>
  <v:f eqn=3D"prod @7 21600 pixelHeight"/>
  <v:f eqn=3D"sum @10 21600 0"/>
 </v:formulas>
 <v:path o:extrusionok=3D"f" gradientshapeok=3D"t" o:connecttype=3D"rect"/>
 <o:lock v:ext=3D"edit" aspectratio=3D"t"/>
</v:shapetype><v:shape id=3D"_x0000_i1025" type=3D"#_x0000_t75" style=3D'wi=
dth:450.75pt;
 height:301.5pt'>
 <v:imagedata src=3D"Filter_files/image001.jpg" o:title=3D"Filter"/>
</v:shape><![endif]--><![if !vml]><img border=3D0 width=3D601 height=3D402
src=3D"Filter_files/image002.jpg" v:shapes=3D"_x0000_i1025"><![endif]><o:p>=
</o:p></span></i></p>

<p class=3DMsoNormal style=3D'text-align:justify'><b><span lang=3DEN-US>Fig=
. 1
Original spectrum (black line) and filtered spectrum (red line) using Cosine
transform and zonal filtration (channels 2048-4095 were set to 0) <o:p></o:=
p></span></b></p>

<p class=3DMsoNormal><b style=3D'mso-bidi-font-weight:normal'><span lang=3D=
EN-US
style=3D'color:#339966'><o:p>&nbsp;</o:p></span></b></p>

<p class=3DMsoNormal><b style=3D'mso-bidi-font-weight:normal'><span lang=3D=
EN-US
style=3D'color:#339966'>Script:<o:p></o:p></span></b></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'>// Examp=
le to
illustrate <span class=3DSpellE>FilterZonal</span> function (class <span
class=3DSpellE>TSpectrumTransform</span>).<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'>// <span
class=3DGramE>To</span> execute this example, do<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'>// root =
&gt; .x <span
class=3DSpellE>Filter.C</span><o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'>#include=
 &lt;<span
class=3DSpellE>TSpectrum</span>&gt;<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'>#include=
 &lt;<span
class=3DSpellE>TSpectrumTransform</span>&gt;<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><o:p>&nb=
sp;</o:p></span></p>

<p class=3DMsoNormal><span class=3DGramE><span lang=3DEN-US style=3D'font-s=
ize:10.0pt'>void</span></span><span
lang=3DEN-US style=3D'font-size:10.0pt'> Filter() {<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span>Int_t i;<o:p></o:p=
></span></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span>Double_t nbins =3D=
 4096;<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span>Double_t xmin<span
style=3D'mso-spacerun:yes'>&nbsp; </span>=3D 0;<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span>Double_t xmax<span
style=3D'mso-spacerun:yes'>&nbsp; </span>=3D (Double_t)nbins;<o:p></o:p></s=
pan></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span></span><span
class=3DSpellE><span lang=3DEN-US style=3D'font-size:10.0pt'>Float_t</span>=
</span><span
lang=3DEN-US style=3D'font-size:10.0pt'> * source =3D new <span class=3DGra=
mE>float[</span><span
class=3DSpellE>nbins</span>];<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><span class=3DSpellE>Float_t=
</span>
* <span class=3DSpellE>dest</span> =3D new <span class=3DGramE>float[</span=
><span
class=3DSpellE>nbins</span>];<span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; =
</span><o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span>TH1F *h =3D new <span class=
=3DGramE>TH1F(</span>&quot;<span
class=3DSpellE>h&quot;,&quot;Zonal</span> filtering using Cosine <span
class=3DSpellE>transform&quot;,nbins,xmin,xmax</span>);<o:p></o:p></span></=
p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span>TH1F *d =3D new <span class=
=3DGramE>TH1F(</span>&quot;<span
class=3DSpellE>d&quot;,&quot;&quot;,nbins,xmin,xmax</span>);<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span><o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><span class=3DSpellE>TFile</=
span> *f
=3D new <span class=3DSpellE><span class=3DGramE>TFile</span></span><span
class=3DGramE>(</span>&quot;spectra\\<span class=3DSpellE>TSpectrum.root</s=
pan>&quot;);<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span>h<span class=3DGramE>=3D(</s=
pan>TH1F*)
f-&gt;Get(&quot;transform1;1&quot;);<span style=3D'mso-spacerun:yes'>&nbsp;=
&nbsp;
</span><o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><span class=3DGramE>for</spa=
n> (<span
class=3DSpellE>i</span> =3D 0; <span class=3DSpellE>i</span> &lt; <span cla=
ss=3DSpellE>nbins</span>;
<span class=3DSpellE>i</span>++) source[<span class=3DSpellE>i</span>]=3Dh-=
&gt;<span
class=3DSpellE>GetBinContent</span>(<span class=3DSpellE>i</span> + 1);<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp; </span><o:p></o:p></spa=
n></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;</span><span style=3D'mso-spacerun:yes'>&n=
bsp;
</span><span class=3DSpellE>TCanvas</span> *Transform1 =3D <span class=3DSp=
ellE>gROOT</span>-&gt;<span
class=3DSpellE><span class=3DGramE>GetListOfCanvases</span></span><span
class=3DGramE>(</span>)-&gt;<span class=3DSpellE>FindObject</span>(&quot;Tr=
ansform1&quot;);<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><span class=3DGramE>if</span>
(!Transform1) Transform1 =3D new <span class=3DSpellE>TCanvas</span>(&quot;=
Transform&quot;,&quot;Transform1&quot;,10,10,1000,700);<o:p></o:p></span></=
p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><span class=3DGramE>h</span>=
-&gt;<span
class=3DSpellE>SetAxisRange</span>(700,1024);<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><span class=3DGramE>h</span>=
-&gt;Draw(&quot;L&quot;);<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><span class=3DSpellE>TSpectr=
um</span>
*s =3D new <span class=3DSpellE><span class=3DGramE>TSpectrum</span></span>=
<span
class=3DGramE>(</span>);<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><span class=3DSpellE>TSpectr=
umTransform</span>
*t =3D new <span class=3DSpellE><span class=3DGramE>TSpectrumTransform</spa=
n></span><span
class=3DGramE>(</span>4096);<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span></span><span lang=3DFR
style=3D'font-size:10.0pt;mso-ansi-language:FR'>t-&gt;SetTransformType(t-&g=
t;kTransformCos,0);<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span>t-&gt;SetRegion(20=
48,
4095);<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp;
</span>t-&gt;FilterZonal(source,dest);<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp; </span><o:p></o:p></spa=
n></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span></span><span
class=3DGramE><span lang=3DEN-US style=3D'font-size:10.0pt'>for</span></spa=
n><span
lang=3DEN-US style=3D'font-size:10.0pt'> (<span class=3DSpellE>i</span> =3D=
 0; <span
class=3DSpellE>i</span> &lt; <span class=3DSpellE>nbins</span>; <span class=
=3DSpellE>i</span>++)
d-&gt;<span class=3DSpellE>SetBinContent</span>(<span class=3DSpellE>i</spa=
n> +
1,dest[<span class=3DSpellE>i</span>]);<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><span class=3DGramE>d</span>=
-&gt;<span
class=3DSpellE>SetLineColor</span>(<span class=3DSpellE>kRed</span>);<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><span class=3DGramE>d</span>=
-&gt;Draw(&quot;SAME
L&quot;);<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'>}<o:p></=
o:p></span></p>

</div>

<!-- */
// --> End_Html
   int i, j=0, k = 1, m, l;
   float val;
   float *working_space = 0;
   double a, b, pi = 3.14159265358979323846, old_area, new_area;
   if (fTransformType >= kTransformFourierWalsh && fTransformType <= kTransformSinHaar) {
      if (fTransformType >= kTransformCosWalsh)
         fDegree += 1;
      k = (int) TMath::Power(2, fDegree);
      j = fSize / k;
   }
   switch (fTransformType) {
   case kTransformHaar:
   case kTransformWalsh:
      working_space = new float[2 * fSize];
      break;
   case kTransformCos:
   case kTransformSin:
   case kTransformFourier:
   case kTransformHartley:
   case kTransformFourierWalsh:
   case kTransformFourierHaar:
   case kTransformWalshHaar:
      working_space = new float[4 * fSize];
      break;
   case kTransformCosWalsh:
   case kTransformCosHaar:
   case kTransformSinWalsh:
   case kTransformSinHaar:
      working_space = new float[8 * fSize];
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
         a = pi * (double) i / (double) fSize;
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
         a = pi * (double) i / (double) fSize;
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
            j = (int) TMath::Power(2, fDegree) / 2;
            k = i / j;
            k = 2 * k * j;
            working_space[k + i % j] = val;
            working_space[k + 2 * j - 1 - i % j] = val;
         }
         
         else if (fTransformType == kTransformSinWalsh
                  || fTransformType == kTransformSinHaar) {
            j = (int) TMath::Power(2, fDegree) / 2;
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
         m = (int) TMath::Power(2, fDegree);
         l = 2 * fSize / m;
         for (i = 0; i < l; i++)
            BitReverseHaar(working_space, 2 * fSize, m, i * m);
         GeneralExe(working_space, 0, 2 * fSize, fDegree, fTransformType);
         for (i = 0; i < fSize; i++) {
            k = i / j;
            k = 2 * k * j;
            a = pi * (double) (i % j) / (double) (2 * j);
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
         m = (int) TMath::Power(2, fDegree);
         l = 2 * fSize / m;
         for (i = 0; i < l; i++)
            BitReverseHaar(working_space, 2 * fSize, m, i * m);
         GeneralExe(working_space, 0, 2 * fSize, fDegree, fTransformType);
         for (i = 0; i < fSize; i++) {
            k = i / j;
            k = 2 * k * j;
            a = pi * (double) (i % j) / (double) (2 * j);
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
         k = (int) TMath::Power(2, fDegree - 1);
      
      else
         k = (int) TMath::Power(2, fDegree);
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
         a = pi * (double) i / (double) fSize;
         b = TMath::Sin(a);
         a = TMath::Cos(a);
         working_space[i + fSize] = (double) working_space[i] * b;
         working_space[i] = (double) working_space[i] * a;
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
         a = pi * (double) i / (double) fSize;
         working_space[i + fSize] =
             -(double) working_space[i - 1] * TMath::Cos(a);
         working_space[i] = (double) working_space[i - 1] * TMath::Sin(a);
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
         k = (int) TMath::Power(2, fDegree - 1);
      
      else
         k = (int) TMath::Power(2, fDegree);
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
         j = (int) TMath::Power(2, fDegree) / 2;
         m = (int) TMath::Power(2, fDegree);
         l = 2 * fSize / m;
         for (i = 0; i < fSize; i++) {
            k = i / j;
            k = 2 * k * j;
            a = pi * (double) (i % j) / (double) (2 * j);
            if (i % j == 0) {
               working_space[2 * fSize + k + i % j] =
                   working_space[i] * TMath::Sqrt(2.0);
               working_space[4 * fSize + 2 * fSize + k + i % j] = 0;
            }
            
            else {
               b = TMath::Sin(a);
               a = TMath::Cos(a);
               working_space[4 * fSize + 2 * fSize + k + i % j] =
                   -(double) working_space[i] * b;
               working_space[2 * fSize + k + i % j] =
                   (double) working_space[i] * a;
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
         m = (int) TMath::Power(2, fDegree);
         l = 2 * fSize / m;
         for (i = 0; i < l; i++)
            BitReverseHaar(working_space, 2 * fSize, m, i * m);
      }
      
      else if (fTransformType == kTransformSinWalsh || fTransformType == kTransformSinHaar) {
         j = (int) TMath::Power(2, fDegree) / 2;
         m = (int) TMath::Power(2, fDegree);
         l = 2 * fSize / m;
         for (i = 0; i < fSize; i++) {
            k = i / j;
            k = 2 * k * j;
            a = pi * (double) (i % j) / (double) (2 * j);
            if (i % j == 0) {
               working_space[2 * fSize + k + j + i % j] =
                   working_space[j + k / 2 - i % j - 1] * TMath::Sqrt(2.0);
               working_space[4 * fSize + 2 * fSize + k + j + i % j] = 0;
            }
            
            else {
               b = TMath::Sin(a);
               a = TMath::Cos(a);
               working_space[4 * fSize + 2 * fSize + k + j + i % j] =
                   -(double) working_space[j + k / 2 - i % j - 1] * b;
               working_space[2 * fSize + k + j + i % j] =
                   (double) working_space[j + k / 2 - i % j - 1] * a;
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
void TSpectrumTransform::Enhance(const float *source, float *destVector) 
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
<div class=3DSection3>

<p class=3DMsoNormal><b><span lang=3DEN-US style=3D'font-size:14.0pt'>Examp=
le of enhancement</span></b><span
lang=3DEN-US style=3D'font-size:14.0pt'><o:p></o:p></span></p>

<p class=3DMsoNormal><i><span lang=3DEN-US><o:p>&nbsp;</o:p></span></i></p>

<p class=3DMsoNormal><i><span lang=3DEN-US>Function:</span></i></p>

<p class=3DMsoNormal><span class=3DGramE><b style=3D'mso-bidi-font-weight:n=
ormal'><span
lang=3DEN-US>void</span></b></span><b style=3D'mso-bidi-font-weight:normal'=
><span
lang=3DEN-US> <span class=3DSpellE>TSpectrumTransform::Enhance</span>(const=
 <a
href=3D"http://root.cern.ch/root/html/ListOfTypes.html#float">float</a> *<s=
pan
class=3DSpellE>fSource</span>, <a
href=3D"http://root.cern.ch/root/html/ListOfTypes.html#float">float</a> *<s=
pan
class=3DSpellE>fDest</span>)<o:p></o:p></span></b></p>

<p class=3DMsoNormal><b style=3D'mso-bidi-font-weight:normal'><span lang=3D=
EN-US><o:p>&nbsp;</o:p></span></b></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US>This f=
unction
transforms the source spectrum (for details see Transform function). Before=
 the
Enhance function is called the class must be created by constructor and the
type of the transform as well as some other parameters must be set using a =
set
of setter <span class=3DSpellE>funcions</span>. The Enhance function multip=
lies
transformed coefficients in the given region (<span class=3DSpellE>fXmin</s=
pan>, <span
class=3DSpellE>fXmax</span>) by the given <span class=3DSpellE>fEnhancCoeff=
</span>
and transforms it back. Enhanced data are written into <span class=3DSpellE=
>dest</span>
spectrum.<b style=3D'mso-bidi-font-weight:normal'><o:p></o:p></b></span></p>

<p class=3DMsoNormal><span lang=3DEN-US><o:p>&nbsp;</o:p></span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span class=3DGramE><i><s=
pan
lang=3DEN-US>Example <span style=3D'mso-spacerun:yes'>&nbsp;</span>&#8211;<=
/span></i></span><i><span
lang=3DEN-US> script <span class=3DSpellE>Enhance.c</span>:<o:p></o:p></spa=
n></i></p>

<p class=3DMsoNormal style=3D'text-align:justify'><i><span lang=3DEN-US
style=3D'font-size:16.0pt'><!--[if gte vml 1]><v:shapetype id=3D"_x0000_t75"
 coordsize=3D"21600,21600" o:spt=3D"75" o:preferrelative=3D"t" path=3D"m@4@=
5l@4@11@9@11@9@5xe"
 filled=3D"f" stroked=3D"f">
 <v:stroke joinstyle=3D"miter"/>
 <v:formulas>
  <v:f eqn=3D"if lineDrawn pixelLineWidth 0"/>
  <v:f eqn=3D"sum @0 1 0"/>
  <v:f eqn=3D"sum 0 0 @1"/>
  <v:f eqn=3D"prod @2 1 2"/>
  <v:f eqn=3D"prod @3 21600 pixelWidth"/>
  <v:f eqn=3D"prod @3 21600 pixelHeight"/>
  <v:f eqn=3D"sum @0 0 1"/>
  <v:f eqn=3D"prod @6 1 2"/>
  <v:f eqn=3D"prod @7 21600 pixelWidth"/>
  <v:f eqn=3D"sum @8 21600 0"/>
  <v:f eqn=3D"prod @7 21600 pixelHeight"/>
  <v:f eqn=3D"sum @10 21600 0"/>
 </v:formulas>
 <v:path o:extrusionok=3D"f" gradientshapeok=3D"t" o:connecttype=3D"rect"/>
 <o:lock v:ext=3D"edit" aspectratio=3D"t"/>
</v:shapetype><v:shape id=3D"_x0000_i1025" type=3D"#_x0000_t75" style=3D'wi=
dth:450.75pt;
 height:301.5pt'>
 <v:imagedata src=3D"Enhance_files/image001.jpg" o:title=3D"Enhance"/>
</v:shape><![endif]--><![if !vml]><img border=3D0 width=3D601 height=3D402
src=3D"Enhance_files/image002.jpg" v:shapes=3D"_x0000_i1025"><![endif]><o:p=
></o:p></span></i></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US
style=3D'font-size:18.0pt'><o:p>&nbsp;</o:p></span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><b><span lang=3DEN-US>Fig=
. 1
Original spectrum (black line) and enhanced spectrum (red line) using Cosine
transform (channels 0-1024 were multiplied by 2) </span></b></p>

<p class=3DMsoNormal><b style=3D'mso-bidi-font-weight:normal'><span lang=3D=
EN-US
style=3D'color:#339966'><o:p>&nbsp;</o:p></span></b></p>

<p class=3DMsoNormal><b style=3D'mso-bidi-font-weight:normal'><span lang=3D=
EN-US
style=3D'color:#339966'>Script:<o:p></o:p></span></b></p>

<p class=3DMsoNormal><span lang=3DEN-US><o:p>&nbsp;</o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'>// Examp=
le to
illustrate Enhance function (class <span class=3DSpellE>TSpectrumTransform<=
/span>).<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'>// <span
class=3DGramE>To</span> execute this example, do<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'>// root =
&gt; .x <span
class=3DSpellE>Enhance.C</span><o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><o:p>&nb=
sp;</o:p></span></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'>void Enhance() {<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span>Int_t i;<o:p></o:p=
></span></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span>Double_t nbins =3D=
 4096;<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span>Double_t xmin<span
style=3D'mso-spacerun:yes'>&nbsp; </span>=3D 0;<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span>Double_t xmax<span
style=3D'mso-spacerun:yes'>&nbsp; </span>=3D (Double_t)nbins;<o:p></o:p></s=
pan></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span></span><span
class=3DSpellE><span lang=3DEN-US style=3D'font-size:10.0pt'>Float_t</span>=
</span><span
lang=3DEN-US style=3D'font-size:10.0pt'> * source =3D new <span class=3DGra=
mE>float[</span><span
class=3DSpellE>nbins</span>];<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><span class=3DSpellE>Float_t=
</span>
* <span class=3DSpellE>dest</span> =3D new <span class=3DGramE>float[</span=
><span
class=3DSpellE>nbins</span>];<span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; =
</span><o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span>TH1F *h =3D new <span class=
=3DGramE>TH1F(</span>&quot;<span
class=3DSpellE>h&quot;,&quot;Enhancement</span> using Cosine <span class=3D=
SpellE>transform&quot;,nbins,xmin,xmax</span>);<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span>TH1F *d =3D new <span class=
=3DGramE>TH1F(</span>&quot;<span
class=3DSpellE>d&quot;,&quot;&quot;,nbins,xmin,xmax</span>);<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span><o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><span class=3DSpellE>TFile</=
span> *f
=3D new <span class=3DSpellE><span class=3DGramE>TFile</span></span><span
class=3DGramE>(</span>&quot;spectra\\<span class=3DSpellE>TSpectrum.root</s=
pan>&quot;);<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span>h<span class=3DGramE>=3D(</s=
pan>TH1F*)
f-&gt;Get(&quot;transform1;1&quot;);<span style=3D'mso-spacerun:yes'>&nbsp;=
&nbsp;
</span><o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><span class=3DGramE>for</spa=
n> (<span
class=3DSpellE>i</span> =3D 0; <span class=3DSpellE>i</span> &lt; <span cla=
ss=3DSpellE>nbins</span>;
<span class=3DSpellE>i</span>++) source[<span class=3DSpellE>i</span>]=3Dh-=
&gt;<span
class=3DSpellE>GetBinContent</span>(<span class=3DSpellE>i</span> + 1);<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp; </span><o:p></o:p></spa=
n></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><span class=3DSpellE>TCanvas=
</span>
*Transform1 =3D <span class=3DSpellE>gROOT</span>-&gt;<span class=3DSpellE>=
<span
class=3DGramE>GetListOfCanvases</span></span><span class=3DGramE>(</span>)-=
&gt;<span
class=3DSpellE>FindObject</span>(&quot;Transform1&quot;);<o:p></o:p></span>=
</p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><span class=3DGramE>if</span>
(!Transform1) Transform1 =3D new <span class=3DSpellE>TCanvas</span>(&quot;=
Transform&quot;,&quot;Transform1&quot;,10,10,1000,700);<o:p></o:p></span></=
p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><span class=3DGramE>h</span>=
-&gt;<span
class=3DSpellE>SetAxisRange</span>(700,1024);<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><span class=3DGramE>h</span>=
-&gt;Draw(&quot;L&quot;);<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><span class=3DSpellE>TSpectr=
um</span>
*s =3D new <span class=3DSpellE><span class=3DGramE>TSpectrum</span></span>=
<span
class=3DGramE>(</span>);<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><span class=3DSpellE>TSpectr=
umTransform</span>
*t =3D new <span class=3DSpellE><span class=3DGramE>TSpectrumTransform</spa=
n></span><span
class=3DGramE>(</span>4096);<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span></span><span lang=3DFR
style=3D'font-size:10.0pt;mso-ansi-language:FR'>t-&gt;SetTransformType(t-&g=
t;kTransformCos,0);<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span>t-&gt;SetRegion(0,
1024);<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp;
</span>t-&gt;SetEnhanceCoeff(2);<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp;
</span>t-&gt;Enhance(source,dest);<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; </spa=
n><o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span></span><span
class=3DGramE><span lang=3DEN-US style=3D'font-size:10.0pt'>for</span></spa=
n><span
lang=3DEN-US style=3D'font-size:10.0pt'> (<span class=3DSpellE>i</span> =3D=
 0; <span
class=3DSpellE>i</span> &lt; <span class=3DSpellE>nbins</span>; <span class=
=3DSpellE>i</span>++)
d-&gt;<span class=3DSpellE>SetBinContent</span>(<span class=3DSpellE>i</spa=
n> +
1,dest[<span class=3DSpellE>i</span>]);<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><span class=3DGramE>d</span>=
-&gt;<span
class=3DSpellE>SetLineColor</span>(<span class=3DSpellE>kRed</span>);<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><span class=3DGramE>d</span>=
-&gt;Draw(&quot;SAME
L&quot;);<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'>}<o:p></=
o:p></span></p>

</div>

<!-- */
// --> End_Html
   int i, j=0, k = 1, m, l;
   float val;
   float *working_space = 0;
   double a, b, pi = 3.14159265358979323846, old_area, new_area;
   if (fTransformType >= kTransformFourierWalsh && fTransformType <= kTransformSinHaar) {
      if (fTransformType >= kTransformCosWalsh)
         fDegree += 1;
      k = (int) TMath::Power(2, fDegree);
      j = fSize / k;
   }
   switch (fTransformType) {
   case kTransformHaar:
   case kTransformWalsh:
      working_space = new float[2 * fSize];
      break;
   case kTransformCos:
   case kTransformSin:
   case kTransformFourier:
   case kTransformHartley:
   case kTransformFourierWalsh:
   case kTransformFourierHaar:
   case kTransformWalshHaar:
      working_space = new float[4 * fSize];
      break;
   case kTransformCosWalsh:
   case kTransformCosHaar:
   case kTransformSinWalsh:
   case kTransformSinHaar:
      working_space = new float[8 * fSize];
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
         a = pi * (double) i / (double) fSize;
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
         a = pi * (double) i / (double) fSize;
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
            j = (int) TMath::Power(2, fDegree) / 2;
            k = i / j;
            k = 2 * k * j;
            working_space[k + i % j] = val;
            working_space[k + 2 * j - 1 - i % j] = val;
         }
         
         else if (fTransformType == kTransformSinWalsh
                  || fTransformType == kTransformSinHaar) {
            j = (int) TMath::Power(2, fDegree) / 2;
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
         m = (int) TMath::Power(2, fDegree);
         l = 2 * fSize / m;
         for (i = 0; i < l; i++)
            BitReverseHaar(working_space, 2 * fSize, m, i * m);
         GeneralExe(working_space, 0, 2 * fSize, fDegree, fTransformType);
         for (i = 0; i < fSize; i++) {
            k = i / j;
            k = 2 * k * j;
            a = pi * (double) (i % j) / (double) (2 * j);
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
         m = (int) TMath::Power(2, fDegree);
         l = 2 * fSize / m;
         for (i = 0; i < l; i++)
            BitReverseHaar(working_space, 2 * fSize, m, i * m);
         GeneralExe(working_space, 0, 2 * fSize, fDegree, fTransformType);
         for (i = 0; i < fSize; i++) {
            k = i / j;
            k = 2 * k * j;
            a = pi * (double) (i % j) / (double) (2 * j);
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
         k = (int) TMath::Power(2, fDegree - 1);
      
      else
         k = (int) TMath::Power(2, fDegree);
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
         a = pi * (double) i / (double) fSize;
         b = TMath::Sin(a);
         a = TMath::Cos(a);
         working_space[i + fSize] = (double) working_space[i] * b;
         working_space[i] = (double) working_space[i] * a;
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
         a = pi * (double) i / (double) fSize;
         working_space[i + fSize] =
             -(double) working_space[i - 1] * TMath::Cos(a);
         working_space[i] = (double) working_space[i - 1] * TMath::Sin(a);
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
         k = (int) TMath::Power(2, fDegree - 1);
      
      else
         k = (int) TMath::Power(2, fDegree);
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
         j = (int) TMath::Power(2, fDegree) / 2;
         m = (int) TMath::Power(2, fDegree);
         l = 2 * fSize / m;
         for (i = 0; i < fSize; i++) {
            k = i / j;
            k = 2 * k * j;
            a = pi * (double) (i % j) / (double) (2 * j);
            if (i % j == 0) {
               working_space[2 * fSize + k + i % j] =
                   working_space[i] * TMath::Sqrt(2.0);
               working_space[4 * fSize + 2 * fSize + k + i % j] = 0;
            }
            
            else {
               b = TMath::Sin(a);
               a = TMath::Cos(a);
               working_space[4 * fSize + 2 * fSize + k + i % j] =
                   -(double) working_space[i] * b;
               working_space[2 * fSize + k + i % j] =
                   (double) working_space[i] * a;
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
         m = (int) TMath::Power(2, fDegree);
         l = 2 * fSize / m;
         for (i = 0; i < l; i++)
            BitReverseHaar(working_space, 2 * fSize, m, i * m);
      }
      
      else if (fTransformType == kTransformSinWalsh || fTransformType == kTransformSinHaar) {
         j = (int) TMath::Power(2, fDegree) / 2;
         m = (int) TMath::Power(2, fDegree);
         l = 2 * fSize / m;
         for (i = 0; i < fSize; i++) {
            k = i / j;
            k = 2 * k * j;
            a = pi * (double) (i % j) / (double) (2 * j);
            if (i % j == 0) {
               working_space[2 * fSize + k + j + i % j] =
                   working_space[j + k / 2 - i % j - 1] * TMath::Sqrt(2.0);
               working_space[4 * fSize + 2 * fSize + k + j + i % j] = 0;
            }
            
            else {
               b = TMath::Sin(a);
               a = TMath::Cos(a);
               working_space[4 * fSize + 2 * fSize + k + j + i % j] =
                   -(double) working_space[j + k / 2 - i % j - 1] * b;
               working_space[2 * fSize + k + j + i % j] =
                   (double) working_space[j + k / 2 - i % j - 1] * a;
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
//   This funcion sets the following parameters for transform:
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
//   This funcion sets the filtering or enhancement region:
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
//   This funcion sets the direction of the transform:
//         -direction (forward or inverse)
//////////////////////////////////////////////////////////////////////////////   
   if(direction != kTransformForward && direction != kTransformInverse){ 
      Error("TSpectrumTransform", "Wrong direction");      
      return;
   }         
   fDirection = direction;
}

//___________________________________________________________________________
void TSpectrumTransform::SetFilterCoeff(Float_t filterCoeff)
{
//////////////////////////////////////////////////////////////////////////////
//   SETTER FUNCION                                                      
//                                                     
//   This funcion sets the filter coefficient:
//         -filterCoeff - after the transform the filtered region (xmin, xmax) is replaced by this coefficient. Applies only for filtereng operation.
//////////////////////////////////////////////////////////////////////////////   
   fFilterCoeff = filterCoeff;
}

//___________________________________________________________________________
void TSpectrumTransform::SetEnhanceCoeff(Float_t enhanceCoeff)
{
//////////////////////////////////////////////////////////////////////////////
//   SETTER FUNCION                                                      
//                                                     
//   This funcion sets the enhancement coefficient:
//         -enhanceCoeff - after the transform the enhanced region (xmin, xmax) is multiplied by this coefficient. Applies only for enhancement operation.
//////////////////////////////////////////////////////////////////////////////   
   fEnhanceCoeff = enhanceCoeff;
}
