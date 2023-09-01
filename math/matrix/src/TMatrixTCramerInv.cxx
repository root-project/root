// @(#)root/base:$Id$
// Authors: Fons Rademakers, Eddy Offermann  Jan 2004

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TMatrixTCramerInv
    \ingroup Matrix

TMatrixTCramerInv

Encapsulate templates of Cramer Inversion routines.

The 4x4, 5x5 and 6x6 are adapted from routines written by
Mark Fischler and Steven Haywood as part of the CLHEP package

Although for sizes <= 6x6 the Cramer Inversion has a gain in speed
compared to factorization schemes (like LU) , one pays a price in
accuracy  .

For Example:
~~~
  H * H^-1 = U, where H is a 5x5 Hilbert matrix
                      U is a 5x5 Unity matrix

 LU    : |U_jk| < 10e-13 for  j!=k
 Cramer: |U_jk| < 10e-7  for  j!=k
~~~

however Cramer algorithm is about 10 (!) times faster
*/

#include "TMatrixTCramerInv.h"
#include "TError.h"

#if !defined(R__SOLARIS) && !defined(R__ACC) && !defined(R__FBSD)
NamespaceImp(TMatrixTCramerInv);
#endif

////////////////////////////////////////////////////////////////////////////////

template<class Element>
Bool_t TMatrixTCramerInv::Inv2x2(TMatrixT<Element> &m,Double_t *determ)
{
   if (m.GetNrows() != 2 || m.GetNcols() != 2 || m.GetRowLwb() != m.GetColLwb()) {
      Error("Inv2x2","matrix should be square 2x2");
      return kFALSE;
   }

   Element *pM = m.GetMatrixArray();

   const Double_t det = pM[0] * pM[3] - pM[2] * pM[1];

   if (determ)
      *determ = det;

   const Double_t s = 1./det;
   if ( det == 0 ) {
      Error("Inv2x2","matrix is singular");
      return kFALSE;
   }

   const Double_t tmp = s*pM[3];
   pM[1] *= -s;
   pM[2] *= -s;
   pM[3] = s*pM[0];
   pM[0] = tmp;

   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////

template<class Element>
Bool_t TMatrixTCramerInv::Inv3x3(TMatrixT<Element> &m,Double_t *determ)
{
   if (m.GetNrows() != 3 || m.GetNcols() != 3 || m.GetRowLwb() != m.GetColLwb()) {
      Error("Inv3x3","matrix should be square 3x3");
      return kFALSE;
   }

   Element *pM = m.GetMatrixArray();

   const Double_t c00 = pM[4] * pM[8] - pM[5] * pM[7];
   const Double_t c01 = pM[5] * pM[6] - pM[3] * pM[8];
   const Double_t c02 = pM[3] * pM[7] - pM[4] * pM[6];
   const Double_t c10 = pM[7] * pM[2] - pM[8] * pM[1];
   const Double_t c11 = pM[8] * pM[0] - pM[6] * pM[2];
   const Double_t c12 = pM[6] * pM[1] - pM[7] * pM[0];
   const Double_t c20 = pM[1] * pM[5] - pM[2] * pM[4];
   const Double_t c21 = pM[2] * pM[3] - pM[0] * pM[5];
   const Double_t c22 = pM[0] * pM[4] - pM[1] * pM[3];

   const Double_t t0 = TMath::Abs(pM[0]);
   const Double_t t1 = TMath::Abs(pM[3]);
   const Double_t t2 = TMath::Abs(pM[6]);
   Double_t det;
   Double_t tmp;
   if (t0 >= t1) {
      if (t2 >= t0) {
      tmp = pM[6];
      det = c12*c01-c11*c02;
      } else {
         tmp = pM[0];
         det = c11*c22-c12*c21;
      }
   } else if (t2 >= t1) {
      tmp = pM[6];
      det = c12*c01-c11*c02;
   } else {
      tmp = pM[3];
      det = c02*c21-c01*c22;
   }

   if ( det == 0 || tmp == 0) {
      Error("Inv3x3","matrix is singular");
      return kFALSE;
   }

   const Double_t s = tmp/det;
   if (determ)
      *determ = 1./s;

   pM[0] = s*c00;
   pM[1] = s*c10;
   pM[2] = s*c20;
   pM[3] = s*c01;
   pM[4] = s*c11;
   pM[5] = s*c21;
   pM[6] = s*c02;
   pM[7] = s*c12;
   pM[8] = s*c22;

   return kTRUE;
}

// GFij are indices for a 4x4 matrix.

#define GF00 0
#define GF01 1
#define GF02 2
#define GF03 3

#define GF10 4
#define GF11 5
#define GF12 6
#define GF13 7

#define GF20 8
#define GF21 9
#define GF22 10
#define GF23 11

#define GF30 12
#define GF31 13
#define GF32 14
#define GF33 15

////////////////////////////////////////////////////////////////////////////////

template<class Element>
Bool_t TMatrixTCramerInv::Inv4x4(TMatrixT<Element> &m,Double_t *determ)
{
   if (m.GetNrows() != 4 || m.GetNcols() != 4 || m.GetRowLwb() != m.GetColLwb()) {
      Error("Inv4x4","matrix should be square 4x4");
      return kFALSE;
   }

   Element *pM = m.GetMatrixArray();

  // Find all NECESSARY 2x2 dets:  (18 of them)

   const Double_t det2_12_01 = pM[GF10]*pM[GF21] - pM[GF11]*pM[GF20];
   const Double_t det2_12_02 = pM[GF10]*pM[GF22] - pM[GF12]*pM[GF20];
   const Double_t det2_12_03 = pM[GF10]*pM[GF23] - pM[GF13]*pM[GF20];
   const Double_t det2_12_13 = pM[GF11]*pM[GF23] - pM[GF13]*pM[GF21];
   const Double_t det2_12_23 = pM[GF12]*pM[GF23] - pM[GF13]*pM[GF22];
   const Double_t det2_12_12 = pM[GF11]*pM[GF22] - pM[GF12]*pM[GF21];
   const Double_t det2_13_01 = pM[GF10]*pM[GF31] - pM[GF11]*pM[GF30];
   const Double_t det2_13_02 = pM[GF10]*pM[GF32] - pM[GF12]*pM[GF30];
   const Double_t det2_13_03 = pM[GF10]*pM[GF33] - pM[GF13]*pM[GF30];
   const Double_t det2_13_12 = pM[GF11]*pM[GF32] - pM[GF12]*pM[GF31];
   const Double_t det2_13_13 = pM[GF11]*pM[GF33] - pM[GF13]*pM[GF31];
   const Double_t det2_13_23 = pM[GF12]*pM[GF33] - pM[GF13]*pM[GF32];
   const Double_t det2_23_01 = pM[GF20]*pM[GF31] - pM[GF21]*pM[GF30];
   const Double_t det2_23_02 = pM[GF20]*pM[GF32] - pM[GF22]*pM[GF30];
   const Double_t det2_23_03 = pM[GF20]*pM[GF33] - pM[GF23]*pM[GF30];
   const Double_t det2_23_12 = pM[GF21]*pM[GF32] - pM[GF22]*pM[GF31];
   const Double_t det2_23_13 = pM[GF21]*pM[GF33] - pM[GF23]*pM[GF31];
   const Double_t det2_23_23 = pM[GF22]*pM[GF33] - pM[GF23]*pM[GF32];

  // Find all NECESSARY 3x3 dets:   (16 of them)

   const Double_t det3_012_012 = pM[GF00]*det2_12_12 - pM[GF01]*det2_12_02
                                 + pM[GF02]*det2_12_01;
   const Double_t det3_012_013 = pM[GF00]*det2_12_13 - pM[GF01]*det2_12_03
                                 + pM[GF03]*det2_12_01;
   const Double_t det3_012_023 = pM[GF00]*det2_12_23 - pM[GF02]*det2_12_03
                                 + pM[GF03]*det2_12_02;
   const Double_t det3_012_123 = pM[GF01]*det2_12_23 - pM[GF02]*det2_12_13
                                 + pM[GF03]*det2_12_12;
   const Double_t det3_013_012 = pM[GF00]*det2_13_12 - pM[GF01]*det2_13_02
                                 + pM[GF02]*det2_13_01;
   const Double_t det3_013_013 = pM[GF00]*det2_13_13 - pM[GF01]*det2_13_03
                                 + pM[GF03]*det2_13_01;
   const Double_t det3_013_023 = pM[GF00]*det2_13_23 - pM[GF02]*det2_13_03
                                 + pM[GF03]*det2_13_02;
   const Double_t det3_013_123 = pM[GF01]*det2_13_23 - pM[GF02]*det2_13_13
                                 + pM[GF03]*det2_13_12;
   const Double_t det3_023_012 = pM[GF00]*det2_23_12 - pM[GF01]*det2_23_02
                                 + pM[GF02]*det2_23_01;
   const Double_t det3_023_013 = pM[GF00]*det2_23_13 - pM[GF01]*det2_23_03
                                 + pM[GF03]*det2_23_01;
   const Double_t det3_023_023 = pM[GF00]*det2_23_23 - pM[GF02]*det2_23_03
                                 + pM[GF03]*det2_23_02;
   const Double_t det3_023_123 = pM[GF01]*det2_23_23 - pM[GF02]*det2_23_13
                                 + pM[GF03]*det2_23_12;
   const Double_t det3_123_012 = pM[GF10]*det2_23_12 - pM[GF11]*det2_23_02
                                 + pM[GF12]*det2_23_01;
   const Double_t det3_123_013 = pM[GF10]*det2_23_13 - pM[GF11]*det2_23_03
                                  + pM[GF13]*det2_23_01;
   const Double_t det3_123_023 = pM[GF10]*det2_23_23 - pM[GF12]*det2_23_03
                                 + pM[GF13]*det2_23_02;
   const Double_t det3_123_123 = pM[GF11]*det2_23_23 - pM[GF12]*det2_23_13
                                 + pM[GF13]*det2_23_12;

  // Find the 4x4 det:

   const Double_t det = pM[GF00]*det3_123_123 - pM[GF01]*det3_123_023
                        + pM[GF02]*det3_123_013 - pM[GF03]*det3_123_012;
   if (determ)
      *determ = det;

   if ( det == 0 ) {
      Error("Inv4x4","matrix is singular");
      return kFALSE;
   }

   const Double_t oneOverDet = 1.0/det;
   const Double_t mn1OverDet = - oneOverDet;

   pM[GF00] =  det3_123_123 * oneOverDet;
   pM[GF01] =  det3_023_123 * mn1OverDet;
   pM[GF02] =  det3_013_123 * oneOverDet;
   pM[GF03] =  det3_012_123 * mn1OverDet;

   pM[GF10] =  det3_123_023 * mn1OverDet;
   pM[GF11] =  det3_023_023 * oneOverDet;
   pM[GF12] =  det3_013_023 * mn1OverDet;
   pM[GF13] =  det3_012_023 * oneOverDet;

   pM[GF20] =  det3_123_013 * oneOverDet;
   pM[GF21] =  det3_023_013 * mn1OverDet;
   pM[GF22] =  det3_013_013 * oneOverDet;
   pM[GF23] =  det3_012_013 * mn1OverDet;

   pM[GF30] =  det3_123_012 * mn1OverDet;
   pM[GF31] =  det3_023_012 * oneOverDet;
   pM[GF32] =  det3_013_012 * mn1OverDet;
   pM[GF33] =  det3_012_012 * oneOverDet;

   return kTRUE;
}

// GMij are indices for a 5x5 matrix.

#define GM00 0
#define GM01 1
#define GM02 2
#define GM03 3
#define GM04 4

#define GM10 5
#define GM11 6
#define GM12 7
#define GM13 8
#define GM14 9

#define GM20 10
#define GM21 11
#define GM22 12
#define GM23 13
#define GM24 14

#define GM30 15
#define GM31 16
#define GM32 17
#define GM33 18
#define GM34 19

#define GM40 20
#define GM41 21
#define GM42 22
#define GM43 23
#define GM44 24

////////////////////////////////////////////////////////////////////////////////

template<class Element>
Bool_t TMatrixTCramerInv::Inv5x5(TMatrixT<Element> &m,Double_t *determ)
{
   if (m.GetNrows() != 5 || m.GetNcols() != 5 || m.GetRowLwb() != m.GetColLwb()) {
      Error("Inv5x5","matrix should be square 5x5");
      return kFALSE;
   }

   Element *pM = m.GetMatrixArray();

  // Find all NECESSARY 2x2 dets:  (30 of them)

   const Double_t det2_23_01 = pM[GM20]*pM[GM31] - pM[GM21]*pM[GM30];
   const Double_t det2_23_02 = pM[GM20]*pM[GM32] - pM[GM22]*pM[GM30];
   const Double_t det2_23_03 = pM[GM20]*pM[GM33] - pM[GM23]*pM[GM30];
   const Double_t det2_23_04 = pM[GM20]*pM[GM34] - pM[GM24]*pM[GM30];
   const Double_t det2_23_12 = pM[GM21]*pM[GM32] - pM[GM22]*pM[GM31];
   const Double_t det2_23_13 = pM[GM21]*pM[GM33] - pM[GM23]*pM[GM31];
   const Double_t det2_23_14 = pM[GM21]*pM[GM34] - pM[GM24]*pM[GM31];
   const Double_t det2_23_23 = pM[GM22]*pM[GM33] - pM[GM23]*pM[GM32];
   const Double_t det2_23_24 = pM[GM22]*pM[GM34] - pM[GM24]*pM[GM32];
   const Double_t det2_23_34 = pM[GM23]*pM[GM34] - pM[GM24]*pM[GM33];
   const Double_t det2_24_01 = pM[GM20]*pM[GM41] - pM[GM21]*pM[GM40];
   const Double_t det2_24_02 = pM[GM20]*pM[GM42] - pM[GM22]*pM[GM40];
   const Double_t det2_24_03 = pM[GM20]*pM[GM43] - pM[GM23]*pM[GM40];
   const Double_t det2_24_04 = pM[GM20]*pM[GM44] - pM[GM24]*pM[GM40];
   const Double_t det2_24_12 = pM[GM21]*pM[GM42] - pM[GM22]*pM[GM41];
   const Double_t det2_24_13 = pM[GM21]*pM[GM43] - pM[GM23]*pM[GM41];
   const Double_t det2_24_14 = pM[GM21]*pM[GM44] - pM[GM24]*pM[GM41];
   const Double_t det2_24_23 = pM[GM22]*pM[GM43] - pM[GM23]*pM[GM42];
   const Double_t det2_24_24 = pM[GM22]*pM[GM44] - pM[GM24]*pM[GM42];
   const Double_t det2_24_34 = pM[GM23]*pM[GM44] - pM[GM24]*pM[GM43];
   const Double_t det2_34_01 = pM[GM30]*pM[GM41] - pM[GM31]*pM[GM40];
   const Double_t det2_34_02 = pM[GM30]*pM[GM42] - pM[GM32]*pM[GM40];
   const Double_t det2_34_03 = pM[GM30]*pM[GM43] - pM[GM33]*pM[GM40];
   const Double_t det2_34_04 = pM[GM30]*pM[GM44] - pM[GM34]*pM[GM40];
   const Double_t det2_34_12 = pM[GM31]*pM[GM42] - pM[GM32]*pM[GM41];
   const Double_t det2_34_13 = pM[GM31]*pM[GM43] - pM[GM33]*pM[GM41];
   const Double_t det2_34_14 = pM[GM31]*pM[GM44] - pM[GM34]*pM[GM41];
   const Double_t det2_34_23 = pM[GM32]*pM[GM43] - pM[GM33]*pM[GM42];
   const Double_t det2_34_24 = pM[GM32]*pM[GM44] - pM[GM34]*pM[GM42];
   const Double_t det2_34_34 = pM[GM33]*pM[GM44] - pM[GM34]*pM[GM43];

  // Find all NECESSARY 3x3 dets:   (40 of them)

   const Double_t det3_123_012 = pM[GM10]*det2_23_12 - pM[GM11]*det2_23_02 + pM[GM12]*det2_23_01;
   const Double_t det3_123_013 = pM[GM10]*det2_23_13 - pM[GM11]*det2_23_03 + pM[GM13]*det2_23_01;
   const Double_t det3_123_014 = pM[GM10]*det2_23_14 - pM[GM11]*det2_23_04 + pM[GM14]*det2_23_01;
   const Double_t det3_123_023 = pM[GM10]*det2_23_23 - pM[GM12]*det2_23_03 + pM[GM13]*det2_23_02;
   const Double_t det3_123_024 = pM[GM10]*det2_23_24 - pM[GM12]*det2_23_04 + pM[GM14]*det2_23_02;
   const Double_t det3_123_034 = pM[GM10]*det2_23_34 - pM[GM13]*det2_23_04 + pM[GM14]*det2_23_03;
   const Double_t det3_123_123 = pM[GM11]*det2_23_23 - pM[GM12]*det2_23_13 + pM[GM13]*det2_23_12;
   const Double_t det3_123_124 = pM[GM11]*det2_23_24 - pM[GM12]*det2_23_14 + pM[GM14]*det2_23_12;
   const Double_t det3_123_134 = pM[GM11]*det2_23_34 - pM[GM13]*det2_23_14 + pM[GM14]*det2_23_13;
   const Double_t det3_123_234 = pM[GM12]*det2_23_34 - pM[GM13]*det2_23_24 + pM[GM14]*det2_23_23;
   const Double_t det3_124_012 = pM[GM10]*det2_24_12 - pM[GM11]*det2_24_02 + pM[GM12]*det2_24_01;
   const Double_t det3_124_013 = pM[GM10]*det2_24_13 - pM[GM11]*det2_24_03 + pM[GM13]*det2_24_01;
   const Double_t det3_124_014 = pM[GM10]*det2_24_14 - pM[GM11]*det2_24_04 + pM[GM14]*det2_24_01;
   const Double_t det3_124_023 = pM[GM10]*det2_24_23 - pM[GM12]*det2_24_03 + pM[GM13]*det2_24_02;
   const Double_t det3_124_024 = pM[GM10]*det2_24_24 - pM[GM12]*det2_24_04 + pM[GM14]*det2_24_02;
   const Double_t det3_124_034 = pM[GM10]*det2_24_34 - pM[GM13]*det2_24_04 + pM[GM14]*det2_24_03;
   const Double_t det3_124_123 = pM[GM11]*det2_24_23 - pM[GM12]*det2_24_13 + pM[GM13]*det2_24_12;
   const Double_t det3_124_124 = pM[GM11]*det2_24_24 - pM[GM12]*det2_24_14 + pM[GM14]*det2_24_12;
   const Double_t det3_124_134 = pM[GM11]*det2_24_34 - pM[GM13]*det2_24_14 + pM[GM14]*det2_24_13;
   const Double_t det3_124_234 = pM[GM12]*det2_24_34 - pM[GM13]*det2_24_24 + pM[GM14]*det2_24_23;
   const Double_t det3_134_012 = pM[GM10]*det2_34_12 - pM[GM11]*det2_34_02 + pM[GM12]*det2_34_01;
   const Double_t det3_134_013 = pM[GM10]*det2_34_13 - pM[GM11]*det2_34_03 + pM[GM13]*det2_34_01;
   const Double_t det3_134_014 = pM[GM10]*det2_34_14 - pM[GM11]*det2_34_04 + pM[GM14]*det2_34_01;
   const Double_t det3_134_023 = pM[GM10]*det2_34_23 - pM[GM12]*det2_34_03 + pM[GM13]*det2_34_02;
   const Double_t det3_134_024 = pM[GM10]*det2_34_24 - pM[GM12]*det2_34_04 + pM[GM14]*det2_34_02;
   const Double_t det3_134_034 = pM[GM10]*det2_34_34 - pM[GM13]*det2_34_04 + pM[GM14]*det2_34_03;
   const Double_t det3_134_123 = pM[GM11]*det2_34_23 - pM[GM12]*det2_34_13 + pM[GM13]*det2_34_12;
   const Double_t det3_134_124 = pM[GM11]*det2_34_24 - pM[GM12]*det2_34_14 + pM[GM14]*det2_34_12;
   const Double_t det3_134_134 = pM[GM11]*det2_34_34 - pM[GM13]*det2_34_14 + pM[GM14]*det2_34_13;
   const Double_t det3_134_234 = pM[GM12]*det2_34_34 - pM[GM13]*det2_34_24 + pM[GM14]*det2_34_23;
   const Double_t det3_234_012 = pM[GM20]*det2_34_12 - pM[GM21]*det2_34_02 + pM[GM22]*det2_34_01;
   const Double_t det3_234_013 = pM[GM20]*det2_34_13 - pM[GM21]*det2_34_03 + pM[GM23]*det2_34_01;
   const Double_t det3_234_014 = pM[GM20]*det2_34_14 - pM[GM21]*det2_34_04 + pM[GM24]*det2_34_01;
   const Double_t det3_234_023 = pM[GM20]*det2_34_23 - pM[GM22]*det2_34_03 + pM[GM23]*det2_34_02;
   const Double_t det3_234_024 = pM[GM20]*det2_34_24 - pM[GM22]*det2_34_04 + pM[GM24]*det2_34_02;
   const Double_t det3_234_034 = pM[GM20]*det2_34_34 - pM[GM23]*det2_34_04 + pM[GM24]*det2_34_03;
   const Double_t det3_234_123 = pM[GM21]*det2_34_23 - pM[GM22]*det2_34_13 + pM[GM23]*det2_34_12;
   const Double_t det3_234_124 = pM[GM21]*det2_34_24 - pM[GM22]*det2_34_14 + pM[GM24]*det2_34_12;
   const Double_t det3_234_134 = pM[GM21]*det2_34_34 - pM[GM23]*det2_34_14 + pM[GM24]*det2_34_13;
   const Double_t det3_234_234 = pM[GM22]*det2_34_34 - pM[GM23]*det2_34_24 + pM[GM24]*det2_34_23;

  // Find all NECESSARY 4x4 dets:   (25 of them)

   const Double_t det4_0123_0123 = pM[GM00]*det3_123_123 - pM[GM01]*det3_123_023
                                   + pM[GM02]*det3_123_013 - pM[GM03]*det3_123_012;
   const Double_t det4_0123_0124 = pM[GM00]*det3_123_124 - pM[GM01]*det3_123_024
                                   + pM[GM02]*det3_123_014 - pM[GM04]*det3_123_012;
   const Double_t det4_0123_0134 = pM[GM00]*det3_123_134 - pM[GM01]*det3_123_034
                                   + pM[GM03]*det3_123_014 - pM[GM04]*det3_123_013;
   const Double_t det4_0123_0234 = pM[GM00]*det3_123_234 - pM[GM02]*det3_123_034
                                   + pM[GM03]*det3_123_024 - pM[GM04]*det3_123_023;
   const Double_t det4_0123_1234 = pM[GM01]*det3_123_234 - pM[GM02]*det3_123_134
                                   + pM[GM03]*det3_123_124 - pM[GM04]*det3_123_123;
   const Double_t det4_0124_0123 = pM[GM00]*det3_124_123 - pM[GM01]*det3_124_023
                                   + pM[GM02]*det3_124_013 - pM[GM03]*det3_124_012;
   const Double_t det4_0124_0124 = pM[GM00]*det3_124_124 - pM[GM01]*det3_124_024
                                   + pM[GM02]*det3_124_014 - pM[GM04]*det3_124_012;
   const Double_t det4_0124_0134 = pM[GM00]*det3_124_134 - pM[GM01]*det3_124_034
                                   + pM[GM03]*det3_124_014 - pM[GM04]*det3_124_013;
   const Double_t det4_0124_0234 = pM[GM00]*det3_124_234 - pM[GM02]*det3_124_034
                                   + pM[GM03]*det3_124_024 - pM[GM04]*det3_124_023;
   const Double_t det4_0124_1234 = pM[GM01]*det3_124_234 - pM[GM02]*det3_124_134
                                   + pM[GM03]*det3_124_124 - pM[GM04]*det3_124_123;
   const Double_t det4_0134_0123 = pM[GM00]*det3_134_123 - pM[GM01]*det3_134_023
                                   + pM[GM02]*det3_134_013 - pM[GM03]*det3_134_012;
   const Double_t det4_0134_0124 = pM[GM00]*det3_134_124 - pM[GM01]*det3_134_024
                                   + pM[GM02]*det3_134_014 - pM[GM04]*det3_134_012;
   const Double_t det4_0134_0134 = pM[GM00]*det3_134_134 - pM[GM01]*det3_134_034
                                   + pM[GM03]*det3_134_014 - pM[GM04]*det3_134_013;
   const Double_t det4_0134_0234 = pM[GM00]*det3_134_234 - pM[GM02]*det3_134_034
                                   + pM[GM03]*det3_134_024 - pM[GM04]*det3_134_023;
   const Double_t det4_0134_1234 = pM[GM01]*det3_134_234 - pM[GM02]*det3_134_134
                                   + pM[GM03]*det3_134_124 - pM[GM04]*det3_134_123;
   const Double_t det4_0234_0123 = pM[GM00]*det3_234_123 - pM[GM01]*det3_234_023
                                   + pM[GM02]*det3_234_013 - pM[GM03]*det3_234_012;
   const Double_t det4_0234_0124 = pM[GM00]*det3_234_124 - pM[GM01]*det3_234_024
                                   + pM[GM02]*det3_234_014 - pM[GM04]*det3_234_012;
   const Double_t det4_0234_0134 = pM[GM00]*det3_234_134 - pM[GM01]*det3_234_034
                                   + pM[GM03]*det3_234_014 - pM[GM04]*det3_234_013;
   const Double_t det4_0234_0234 = pM[GM00]*det3_234_234 - pM[GM02]*det3_234_034
                                   + pM[GM03]*det3_234_024 - pM[GM04]*det3_234_023;
   const Double_t det4_0234_1234 = pM[GM01]*det3_234_234 - pM[GM02]*det3_234_134
                                   + pM[GM03]*det3_234_124 - pM[GM04]*det3_234_123;
   const Double_t det4_1234_0123 = pM[GM10]*det3_234_123 - pM[GM11]*det3_234_023
                                   + pM[GM12]*det3_234_013 - pM[GM13]*det3_234_012;
   const Double_t det4_1234_0124 = pM[GM10]*det3_234_124 - pM[GM11]*det3_234_024
                                    + pM[GM12]*det3_234_014 - pM[GM14]*det3_234_012;
   const Double_t det4_1234_0134 = pM[GM10]*det3_234_134 - pM[GM11]*det3_234_034
                                   + pM[GM13]*det3_234_014 - pM[GM14]*det3_234_013;
   const Double_t det4_1234_0234 = pM[GM10]*det3_234_234 - pM[GM12]*det3_234_034
                                   + pM[GM13]*det3_234_024 - pM[GM14]*det3_234_023;
   const Double_t det4_1234_1234 = pM[GM11]*det3_234_234 - pM[GM12]*det3_234_134
                                   + pM[GM13]*det3_234_124 - pM[GM14]*det3_234_123;

  // Find the 5x5 det:

   const Double_t det = pM[GM00]*det4_1234_1234 - pM[GM01]*det4_1234_0234 + pM[GM02]*det4_1234_0134
                        - pM[GM03]*det4_1234_0124 + pM[GM04]*det4_1234_0123;
   if (determ)
      *determ = det;

   if ( det == 0 ) {
      Error("Inv5x5","matrix is singular");
      return kFALSE;
   }

   const Double_t oneOverDet = 1.0/det;
   const Double_t mn1OverDet = - oneOverDet;

   pM[GM00] =  det4_1234_1234 * oneOverDet;
   pM[GM01] =  det4_0234_1234 * mn1OverDet;
   pM[GM02] =  det4_0134_1234 * oneOverDet;
   pM[GM03] =  det4_0124_1234 * mn1OverDet;
   pM[GM04] =  det4_0123_1234 * oneOverDet;

   pM[GM10] =  det4_1234_0234 * mn1OverDet;
   pM[GM11] =  det4_0234_0234 * oneOverDet;
   pM[GM12] =  det4_0134_0234 * mn1OverDet;
   pM[GM13] =  det4_0124_0234 * oneOverDet;
   pM[GM14] =  det4_0123_0234 * mn1OverDet;

   pM[GM20] =  det4_1234_0134 * oneOverDet;
   pM[GM21] =  det4_0234_0134 * mn1OverDet;
   pM[GM22] =  det4_0134_0134 * oneOverDet;
   pM[GM23] =  det4_0124_0134 * mn1OverDet;
   pM[GM24] =  det4_0123_0134 * oneOverDet;

   pM[GM30] =  det4_1234_0124 * mn1OverDet;
   pM[GM31] =  det4_0234_0124 * oneOverDet;
   pM[GM32] =  det4_0134_0124 * mn1OverDet;
   pM[GM33] =  det4_0124_0124 * oneOverDet;
   pM[GM34] =  det4_0123_0124 * mn1OverDet;

   pM[GM40] =  det4_1234_0123 * oneOverDet;
   pM[GM41] =  det4_0234_0123 * mn1OverDet;
   pM[GM42] =  det4_0134_0123 * oneOverDet;
   pM[GM43] =  det4_0124_0123 * mn1OverDet;
   pM[GM44] =  det4_0123_0123 * oneOverDet;

   return kTRUE;
}

// Aij are indices for a 6x6 matrix.

#define GA00 0
#define GA01 1
#define GA02 2
#define GA03 3
#define GA04 4
#define GA05 5

#define GA10 6
#define GA11 7
#define GA12 8
#define GA13 9
#define GA14 10
#define GA15 11

#define GA20 12
#define GA21 13
#define GA22 14
#define GA23 15
#define GA24 16
#define GA25 17

#define GA30 18
#define GA31 19
#define GA32 20
#define GA33 21
#define GA34 22
#define GA35 23

#define GA40 24
#define GA41 25
#define GA42 26
#define GA43 27
#define GA44 28
#define GA45 29

#define GA50 30
#define GA51 31
#define GA52 32
#define GA53 33
#define GA54 34
#define GA55 35

////////////////////////////////////////////////////////////////////////////////

template<class Element>
Bool_t TMatrixTCramerInv::Inv6x6(TMatrixT<Element> &m,Double_t *determ)
{
   if (m.GetNrows() != 6 || m.GetNcols() != 6 || m.GetRowLwb() != m.GetColLwb()) {
      Error("Inv6x6","matrix should be square 6x6");
      return kFALSE;
   }

   Element *pM = m.GetMatrixArray();

   // Find all NECESSGARY 2x2 dets:  (45 of them)

   const Double_t det2_34_01 = pM[GA30]*pM[GA41] - pM[GA31]*pM[GA40];
   const Double_t det2_34_02 = pM[GA30]*pM[GA42] - pM[GA32]*pM[GA40];
   const Double_t det2_34_03 = pM[GA30]*pM[GA43] - pM[GA33]*pM[GA40];
   const Double_t det2_34_04 = pM[GA30]*pM[GA44] - pM[GA34]*pM[GA40];
   const Double_t det2_34_05 = pM[GA30]*pM[GA45] - pM[GA35]*pM[GA40];
   const Double_t det2_34_12 = pM[GA31]*pM[GA42] - pM[GA32]*pM[GA41];
   const Double_t det2_34_13 = pM[GA31]*pM[GA43] - pM[GA33]*pM[GA41];
   const Double_t det2_34_14 = pM[GA31]*pM[GA44] - pM[GA34]*pM[GA41];
   const Double_t det2_34_15 = pM[GA31]*pM[GA45] - pM[GA35]*pM[GA41];
   const Double_t det2_34_23 = pM[GA32]*pM[GA43] - pM[GA33]*pM[GA42];
   const Double_t det2_34_24 = pM[GA32]*pM[GA44] - pM[GA34]*pM[GA42];
   const Double_t det2_34_25 = pM[GA32]*pM[GA45] - pM[GA35]*pM[GA42];
   const Double_t det2_34_34 = pM[GA33]*pM[GA44] - pM[GA34]*pM[GA43];
   const Double_t det2_34_35 = pM[GA33]*pM[GA45] - pM[GA35]*pM[GA43];
   const Double_t det2_34_45 = pM[GA34]*pM[GA45] - pM[GA35]*pM[GA44];
   const Double_t det2_35_01 = pM[GA30]*pM[GA51] - pM[GA31]*pM[GA50];
   const Double_t det2_35_02 = pM[GA30]*pM[GA52] - pM[GA32]*pM[GA50];
   const Double_t det2_35_03 = pM[GA30]*pM[GA53] - pM[GA33]*pM[GA50];
   const Double_t det2_35_04 = pM[GA30]*pM[GA54] - pM[GA34]*pM[GA50];
   const Double_t det2_35_05 = pM[GA30]*pM[GA55] - pM[GA35]*pM[GA50];
   const Double_t det2_35_12 = pM[GA31]*pM[GA52] - pM[GA32]*pM[GA51];
   const Double_t det2_35_13 = pM[GA31]*pM[GA53] - pM[GA33]*pM[GA51];
   const Double_t det2_35_14 = pM[GA31]*pM[GA54] - pM[GA34]*pM[GA51];
   const Double_t det2_35_15 = pM[GA31]*pM[GA55] - pM[GA35]*pM[GA51];
   const Double_t det2_35_23 = pM[GA32]*pM[GA53] - pM[GA33]*pM[GA52];
   const Double_t det2_35_24 = pM[GA32]*pM[GA54] - pM[GA34]*pM[GA52];
   const Double_t det2_35_25 = pM[GA32]*pM[GA55] - pM[GA35]*pM[GA52];
   const Double_t det2_35_34 = pM[GA33]*pM[GA54] - pM[GA34]*pM[GA53];
   const Double_t det2_35_35 = pM[GA33]*pM[GA55] - pM[GA35]*pM[GA53];
   const Double_t det2_35_45 = pM[GA34]*pM[GA55] - pM[GA35]*pM[GA54];
   const Double_t det2_45_01 = pM[GA40]*pM[GA51] - pM[GA41]*pM[GA50];
   const Double_t det2_45_02 = pM[GA40]*pM[GA52] - pM[GA42]*pM[GA50];
   const Double_t det2_45_03 = pM[GA40]*pM[GA53] - pM[GA43]*pM[GA50];
   const Double_t det2_45_04 = pM[GA40]*pM[GA54] - pM[GA44]*pM[GA50];
   const Double_t det2_45_05 = pM[GA40]*pM[GA55] - pM[GA45]*pM[GA50];
   const Double_t det2_45_12 = pM[GA41]*pM[GA52] - pM[GA42]*pM[GA51];
   const Double_t det2_45_13 = pM[GA41]*pM[GA53] - pM[GA43]*pM[GA51];
   const Double_t det2_45_14 = pM[GA41]*pM[GA54] - pM[GA44]*pM[GA51];
   const Double_t det2_45_15 = pM[GA41]*pM[GA55] - pM[GA45]*pM[GA51];
   const Double_t det2_45_23 = pM[GA42]*pM[GA53] - pM[GA43]*pM[GA52];
   const Double_t det2_45_24 = pM[GA42]*pM[GA54] - pM[GA44]*pM[GA52];
   const Double_t det2_45_25 = pM[GA42]*pM[GA55] - pM[GA45]*pM[GA52];
   const Double_t det2_45_34 = pM[GA43]*pM[GA54] - pM[GA44]*pM[GA53];
   const Double_t det2_45_35 = pM[GA43]*pM[GA55] - pM[GA45]*pM[GA53];
   const Double_t det2_45_45 = pM[GA44]*pM[GA55] - pM[GA45]*pM[GA54];

  // Find all NECESSGARY 3x3 dets:  (80 of them)

   const Double_t det3_234_012 = pM[GA20]*det2_34_12 - pM[GA21]*det2_34_02 + pM[GA22]*det2_34_01;
   const Double_t det3_234_013 = pM[GA20]*det2_34_13 - pM[GA21]*det2_34_03 + pM[GA23]*det2_34_01;
   const Double_t det3_234_014 = pM[GA20]*det2_34_14 - pM[GA21]*det2_34_04 + pM[GA24]*det2_34_01;
   const Double_t det3_234_015 = pM[GA20]*det2_34_15 - pM[GA21]*det2_34_05 + pM[GA25]*det2_34_01;
   const Double_t det3_234_023 = pM[GA20]*det2_34_23 - pM[GA22]*det2_34_03 + pM[GA23]*det2_34_02;
   const Double_t det3_234_024 = pM[GA20]*det2_34_24 - pM[GA22]*det2_34_04 + pM[GA24]*det2_34_02;
   const Double_t det3_234_025 = pM[GA20]*det2_34_25 - pM[GA22]*det2_34_05 + pM[GA25]*det2_34_02;
   const Double_t det3_234_034 = pM[GA20]*det2_34_34 - pM[GA23]*det2_34_04 + pM[GA24]*det2_34_03;
   const Double_t det3_234_035 = pM[GA20]*det2_34_35 - pM[GA23]*det2_34_05 + pM[GA25]*det2_34_03;
   const Double_t det3_234_045 = pM[GA20]*det2_34_45 - pM[GA24]*det2_34_05 + pM[GA25]*det2_34_04;
   const Double_t det3_234_123 = pM[GA21]*det2_34_23 - pM[GA22]*det2_34_13 + pM[GA23]*det2_34_12;
   const Double_t det3_234_124 = pM[GA21]*det2_34_24 - pM[GA22]*det2_34_14 + pM[GA24]*det2_34_12;
   const Double_t det3_234_125 = pM[GA21]*det2_34_25 - pM[GA22]*det2_34_15 + pM[GA25]*det2_34_12;
   const Double_t det3_234_134 = pM[GA21]*det2_34_34 - pM[GA23]*det2_34_14 + pM[GA24]*det2_34_13;
   const Double_t det3_234_135 = pM[GA21]*det2_34_35 - pM[GA23]*det2_34_15 + pM[GA25]*det2_34_13;
   const Double_t det3_234_145 = pM[GA21]*det2_34_45 - pM[GA24]*det2_34_15 + pM[GA25]*det2_34_14;
   const Double_t det3_234_234 = pM[GA22]*det2_34_34 - pM[GA23]*det2_34_24 + pM[GA24]*det2_34_23;
   const Double_t det3_234_235 = pM[GA22]*det2_34_35 - pM[GA23]*det2_34_25 + pM[GA25]*det2_34_23;
   const Double_t det3_234_245 = pM[GA22]*det2_34_45 - pM[GA24]*det2_34_25 + pM[GA25]*det2_34_24;
   const Double_t det3_234_345 = pM[GA23]*det2_34_45 - pM[GA24]*det2_34_35 + pM[GA25]*det2_34_34;
   const Double_t det3_235_012 = pM[GA20]*det2_35_12 - pM[GA21]*det2_35_02 + pM[GA22]*det2_35_01;
   const Double_t det3_235_013 = pM[GA20]*det2_35_13 - pM[GA21]*det2_35_03 + pM[GA23]*det2_35_01;
   const Double_t det3_235_014 = pM[GA20]*det2_35_14 - pM[GA21]*det2_35_04 + pM[GA24]*det2_35_01;
   const Double_t det3_235_015 = pM[GA20]*det2_35_15 - pM[GA21]*det2_35_05 + pM[GA25]*det2_35_01;
   const Double_t det3_235_023 = pM[GA20]*det2_35_23 - pM[GA22]*det2_35_03 + pM[GA23]*det2_35_02;
   const Double_t det3_235_024 = pM[GA20]*det2_35_24 - pM[GA22]*det2_35_04 + pM[GA24]*det2_35_02;
   const Double_t det3_235_025 = pM[GA20]*det2_35_25 - pM[GA22]*det2_35_05 + pM[GA25]*det2_35_02;
   const Double_t det3_235_034 = pM[GA20]*det2_35_34 - pM[GA23]*det2_35_04 + pM[GA24]*det2_35_03;
   const Double_t det3_235_035 = pM[GA20]*det2_35_35 - pM[GA23]*det2_35_05 + pM[GA25]*det2_35_03;
   const Double_t det3_235_045 = pM[GA20]*det2_35_45 - pM[GA24]*det2_35_05 + pM[GA25]*det2_35_04;
   const Double_t det3_235_123 = pM[GA21]*det2_35_23 - pM[GA22]*det2_35_13 + pM[GA23]*det2_35_12;
   const Double_t det3_235_124 = pM[GA21]*det2_35_24 - pM[GA22]*det2_35_14 + pM[GA24]*det2_35_12;
   const Double_t det3_235_125 = pM[GA21]*det2_35_25 - pM[GA22]*det2_35_15 + pM[GA25]*det2_35_12;
   const Double_t det3_235_134 = pM[GA21]*det2_35_34 - pM[GA23]*det2_35_14 + pM[GA24]*det2_35_13;
   const Double_t det3_235_135 = pM[GA21]*det2_35_35 - pM[GA23]*det2_35_15 + pM[GA25]*det2_35_13;
   const Double_t det3_235_145 = pM[GA21]*det2_35_45 - pM[GA24]*det2_35_15 + pM[GA25]*det2_35_14;
   const Double_t det3_235_234 = pM[GA22]*det2_35_34 - pM[GA23]*det2_35_24 + pM[GA24]*det2_35_23;
   const Double_t det3_235_235 = pM[GA22]*det2_35_35 - pM[GA23]*det2_35_25 + pM[GA25]*det2_35_23;
   const Double_t det3_235_245 = pM[GA22]*det2_35_45 - pM[GA24]*det2_35_25 + pM[GA25]*det2_35_24;
   const Double_t det3_235_345 = pM[GA23]*det2_35_45 - pM[GA24]*det2_35_35 + pM[GA25]*det2_35_34;
   const Double_t det3_245_012 = pM[GA20]*det2_45_12 - pM[GA21]*det2_45_02 + pM[GA22]*det2_45_01;
   const Double_t det3_245_013 = pM[GA20]*det2_45_13 - pM[GA21]*det2_45_03 + pM[GA23]*det2_45_01;
   const Double_t det3_245_014 = pM[GA20]*det2_45_14 - pM[GA21]*det2_45_04 + pM[GA24]*det2_45_01;
   const Double_t det3_245_015 = pM[GA20]*det2_45_15 - pM[GA21]*det2_45_05 + pM[GA25]*det2_45_01;
   const Double_t det3_245_023 = pM[GA20]*det2_45_23 - pM[GA22]*det2_45_03 + pM[GA23]*det2_45_02;
   const Double_t det3_245_024 = pM[GA20]*det2_45_24 - pM[GA22]*det2_45_04 + pM[GA24]*det2_45_02;
   const Double_t det3_245_025 = pM[GA20]*det2_45_25 - pM[GA22]*det2_45_05 + pM[GA25]*det2_45_02;
   const Double_t det3_245_034 = pM[GA20]*det2_45_34 - pM[GA23]*det2_45_04 + pM[GA24]*det2_45_03;
   const Double_t det3_245_035 = pM[GA20]*det2_45_35 - pM[GA23]*det2_45_05 + pM[GA25]*det2_45_03;
   const Double_t det3_245_045 = pM[GA20]*det2_45_45 - pM[GA24]*det2_45_05 + pM[GA25]*det2_45_04;
   const Double_t det3_245_123 = pM[GA21]*det2_45_23 - pM[GA22]*det2_45_13 + pM[GA23]*det2_45_12;
   const Double_t det3_245_124 = pM[GA21]*det2_45_24 - pM[GA22]*det2_45_14 + pM[GA24]*det2_45_12;
   const Double_t det3_245_125 = pM[GA21]*det2_45_25 - pM[GA22]*det2_45_15 + pM[GA25]*det2_45_12;
   const Double_t det3_245_134 = pM[GA21]*det2_45_34 - pM[GA23]*det2_45_14 + pM[GA24]*det2_45_13;
   const Double_t det3_245_135 = pM[GA21]*det2_45_35 - pM[GA23]*det2_45_15 + pM[GA25]*det2_45_13;
   const Double_t det3_245_145 = pM[GA21]*det2_45_45 - pM[GA24]*det2_45_15 + pM[GA25]*det2_45_14;
   const Double_t det3_245_234 = pM[GA22]*det2_45_34 - pM[GA23]*det2_45_24 + pM[GA24]*det2_45_23;
   const Double_t det3_245_235 = pM[GA22]*det2_45_35 - pM[GA23]*det2_45_25 + pM[GA25]*det2_45_23;
   const Double_t det3_245_245 = pM[GA22]*det2_45_45 - pM[GA24]*det2_45_25 + pM[GA25]*det2_45_24;
   const Double_t det3_245_345 = pM[GA23]*det2_45_45 - pM[GA24]*det2_45_35 + pM[GA25]*det2_45_34;
   const Double_t det3_345_012 = pM[GA30]*det2_45_12 - pM[GA31]*det2_45_02 + pM[GA32]*det2_45_01;
   const Double_t det3_345_013 = pM[GA30]*det2_45_13 - pM[GA31]*det2_45_03 + pM[GA33]*det2_45_01;
   const Double_t det3_345_014 = pM[GA30]*det2_45_14 - pM[GA31]*det2_45_04 + pM[GA34]*det2_45_01;
   const Double_t det3_345_015 = pM[GA30]*det2_45_15 - pM[GA31]*det2_45_05 + pM[GA35]*det2_45_01;
   const Double_t det3_345_023 = pM[GA30]*det2_45_23 - pM[GA32]*det2_45_03 + pM[GA33]*det2_45_02;
   const Double_t det3_345_024 = pM[GA30]*det2_45_24 - pM[GA32]*det2_45_04 + pM[GA34]*det2_45_02;
   const Double_t det3_345_025 = pM[GA30]*det2_45_25 - pM[GA32]*det2_45_05 + pM[GA35]*det2_45_02;
   const Double_t det3_345_034 = pM[GA30]*det2_45_34 - pM[GA33]*det2_45_04 + pM[GA34]*det2_45_03;
   const Double_t det3_345_035 = pM[GA30]*det2_45_35 - pM[GA33]*det2_45_05 + pM[GA35]*det2_45_03;
   const Double_t det3_345_045 = pM[GA30]*det2_45_45 - pM[GA34]*det2_45_05 + pM[GA35]*det2_45_04;
   const Double_t det3_345_123 = pM[GA31]*det2_45_23 - pM[GA32]*det2_45_13 + pM[GA33]*det2_45_12;
   const Double_t det3_345_124 = pM[GA31]*det2_45_24 - pM[GA32]*det2_45_14 + pM[GA34]*det2_45_12;
   const Double_t det3_345_125 = pM[GA31]*det2_45_25 - pM[GA32]*det2_45_15 + pM[GA35]*det2_45_12;
   const Double_t det3_345_134 = pM[GA31]*det2_45_34 - pM[GA33]*det2_45_14 + pM[GA34]*det2_45_13;
   const Double_t det3_345_135 = pM[GA31]*det2_45_35 - pM[GA33]*det2_45_15 + pM[GA35]*det2_45_13;
   const Double_t det3_345_145 = pM[GA31]*det2_45_45 - pM[GA34]*det2_45_15 + pM[GA35]*det2_45_14;
   const Double_t det3_345_234 = pM[GA32]*det2_45_34 - pM[GA33]*det2_45_24 + pM[GA34]*det2_45_23;
   const Double_t det3_345_235 = pM[GA32]*det2_45_35 - pM[GA33]*det2_45_25 + pM[GA35]*det2_45_23;
   const Double_t det3_345_245 = pM[GA32]*det2_45_45 - pM[GA34]*det2_45_25 + pM[GA35]*det2_45_24;
   const Double_t det3_345_345 = pM[GA33]*det2_45_45 - pM[GA34]*det2_45_35 + pM[GA35]*det2_45_34;

  // Find all NECESSGARY 4x4 dets:  (75 of them)

   const Double_t det4_1234_0123 = pM[GA10]*det3_234_123 - pM[GA11]*det3_234_023
                                   + pM[GA12]*det3_234_013 - pM[GA13]*det3_234_012;
   const Double_t det4_1234_0124 = pM[GA10]*det3_234_124 - pM[GA11]*det3_234_024
                                   + pM[GA12]*det3_234_014 - pM[GA14]*det3_234_012;
   const Double_t det4_1234_0125 = pM[GA10]*det3_234_125 - pM[GA11]*det3_234_025
                                   + pM[GA12]*det3_234_015 - pM[GA15]*det3_234_012;
   const Double_t det4_1234_0134 = pM[GA10]*det3_234_134 - pM[GA11]*det3_234_034
                                   + pM[GA13]*det3_234_014 - pM[GA14]*det3_234_013;
   const Double_t det4_1234_0135 = pM[GA10]*det3_234_135 - pM[GA11]*det3_234_035
                                   + pM[GA13]*det3_234_015 - pM[GA15]*det3_234_013;
   const Double_t det4_1234_0145 = pM[GA10]*det3_234_145 - pM[GA11]*det3_234_045
                                   + pM[GA14]*det3_234_015 - pM[GA15]*det3_234_014;
   const Double_t det4_1234_0234 = pM[GA10]*det3_234_234 - pM[GA12]*det3_234_034
                                   + pM[GA13]*det3_234_024 - pM[GA14]*det3_234_023;
   const Double_t det4_1234_0235 = pM[GA10]*det3_234_235 - pM[GA12]*det3_234_035
                                   + pM[GA13]*det3_234_025 - pM[GA15]*det3_234_023;
   const Double_t det4_1234_0245 = pM[GA10]*det3_234_245 - pM[GA12]*det3_234_045
                                   + pM[GA14]*det3_234_025 - pM[GA15]*det3_234_024;
   const Double_t det4_1234_0345 = pM[GA10]*det3_234_345 - pM[GA13]*det3_234_045
                                   + pM[GA14]*det3_234_035 - pM[GA15]*det3_234_034;
   const Double_t det4_1234_1234 = pM[GA11]*det3_234_234 - pM[GA12]*det3_234_134
                                   + pM[GA13]*det3_234_124 - pM[GA14]*det3_234_123;
   const Double_t det4_1234_1235 = pM[GA11]*det3_234_235 - pM[GA12]*det3_234_135
                                   + pM[GA13]*det3_234_125 - pM[GA15]*det3_234_123;
   const Double_t det4_1234_1245 = pM[GA11]*det3_234_245 - pM[GA12]*det3_234_145
                                   + pM[GA14]*det3_234_125 - pM[GA15]*det3_234_124;
   const Double_t det4_1234_1345 = pM[GA11]*det3_234_345 - pM[GA13]*det3_234_145
                                   + pM[GA14]*det3_234_135 - pM[GA15]*det3_234_134;
   const Double_t det4_1234_2345 = pM[GA12]*det3_234_345 - pM[GA13]*det3_234_245
                                   + pM[GA14]*det3_234_235 - pM[GA15]*det3_234_234;
   const Double_t det4_1235_0123 = pM[GA10]*det3_235_123 - pM[GA11]*det3_235_023
                                   + pM[GA12]*det3_235_013 - pM[GA13]*det3_235_012;
   const Double_t det4_1235_0124 = pM[GA10]*det3_235_124 - pM[GA11]*det3_235_024
                                   + pM[GA12]*det3_235_014 - pM[GA14]*det3_235_012;
   const Double_t det4_1235_0125 = pM[GA10]*det3_235_125 - pM[GA11]*det3_235_025
                                   + pM[GA12]*det3_235_015 - pM[GA15]*det3_235_012;
   const Double_t det4_1235_0134 = pM[GA10]*det3_235_134 - pM[GA11]*det3_235_034
                                   + pM[GA13]*det3_235_014 - pM[GA14]*det3_235_013;
   const Double_t det4_1235_0135 = pM[GA10]*det3_235_135 - pM[GA11]*det3_235_035
                                   + pM[GA13]*det3_235_015 - pM[GA15]*det3_235_013;
   const Double_t det4_1235_0145 = pM[GA10]*det3_235_145 - pM[GA11]*det3_235_045
                                   + pM[GA14]*det3_235_015 - pM[GA15]*det3_235_014;
   const Double_t det4_1235_0234 = pM[GA10]*det3_235_234 - pM[GA12]*det3_235_034
                                   + pM[GA13]*det3_235_024 - pM[GA14]*det3_235_023;
   const Double_t det4_1235_0235 = pM[GA10]*det3_235_235 - pM[GA12]*det3_235_035
                                   + pM[GA13]*det3_235_025 - pM[GA15]*det3_235_023;
   const Double_t det4_1235_0245 = pM[GA10]*det3_235_245 - pM[GA12]*det3_235_045
                                   + pM[GA14]*det3_235_025 - pM[GA15]*det3_235_024;
   const Double_t det4_1235_0345 = pM[GA10]*det3_235_345 - pM[GA13]*det3_235_045
                                   + pM[GA14]*det3_235_035 - pM[GA15]*det3_235_034;
   const Double_t det4_1235_1234 = pM[GA11]*det3_235_234 - pM[GA12]*det3_235_134
                                   + pM[GA13]*det3_235_124 - pM[GA14]*det3_235_123;
   const Double_t det4_1235_1235 = pM[GA11]*det3_235_235 - pM[GA12]*det3_235_135
                                   + pM[GA13]*det3_235_125 - pM[GA15]*det3_235_123;
   const Double_t det4_1235_1245 = pM[GA11]*det3_235_245 - pM[GA12]*det3_235_145
                                   + pM[GA14]*det3_235_125 - pM[GA15]*det3_235_124;
   const Double_t det4_1235_1345 = pM[GA11]*det3_235_345 - pM[GA13]*det3_235_145
                                   + pM[GA14]*det3_235_135 - pM[GA15]*det3_235_134;
   const Double_t det4_1235_2345 = pM[GA12]*det3_235_345 - pM[GA13]*det3_235_245
                                   + pM[GA14]*det3_235_235 - pM[GA15]*det3_235_234;
   const Double_t det4_1245_0123 = pM[GA10]*det3_245_123 - pM[GA11]*det3_245_023
                                   + pM[GA12]*det3_245_013 - pM[GA13]*det3_245_012;
   const Double_t det4_1245_0124 = pM[GA10]*det3_245_124 - pM[GA11]*det3_245_024
                                   + pM[GA12]*det3_245_014 - pM[GA14]*det3_245_012;
   const Double_t det4_1245_0125 = pM[GA10]*det3_245_125 - pM[GA11]*det3_245_025
                                   + pM[GA12]*det3_245_015 - pM[GA15]*det3_245_012;
   const Double_t det4_1245_0134 = pM[GA10]*det3_245_134 - pM[GA11]*det3_245_034
                                   + pM[GA13]*det3_245_014 - pM[GA14]*det3_245_013;
   const Double_t det4_1245_0135 = pM[GA10]*det3_245_135 - pM[GA11]*det3_245_035
                                   + pM[GA13]*det3_245_015 - pM[GA15]*det3_245_013;
   const Double_t det4_1245_0145 = pM[GA10]*det3_245_145 - pM[GA11]*det3_245_045
                                   + pM[GA14]*det3_245_015 - pM[GA15]*det3_245_014;
   const Double_t det4_1245_0234 = pM[GA10]*det3_245_234 - pM[GA12]*det3_245_034
                                   + pM[GA13]*det3_245_024 - pM[GA14]*det3_245_023;
   const Double_t det4_1245_0235 = pM[GA10]*det3_245_235 - pM[GA12]*det3_245_035
                                   + pM[GA13]*det3_245_025 - pM[GA15]*det3_245_023;
   const Double_t det4_1245_0245 = pM[GA10]*det3_245_245 - pM[GA12]*det3_245_045
                                   + pM[GA14]*det3_245_025 - pM[GA15]*det3_245_024;
   const Double_t det4_1245_0345 = pM[GA10]*det3_245_345 - pM[GA13]*det3_245_045
                                   + pM[GA14]*det3_245_035 - pM[GA15]*det3_245_034;
   const Double_t det4_1245_1234 = pM[GA11]*det3_245_234 - pM[GA12]*det3_245_134
                                   + pM[GA13]*det3_245_124 - pM[GA14]*det3_245_123;
   const Double_t det4_1245_1235 = pM[GA11]*det3_245_235 - pM[GA12]*det3_245_135
                                   + pM[GA13]*det3_245_125 - pM[GA15]*det3_245_123;
   const Double_t det4_1245_1245 = pM[GA11]*det3_245_245 - pM[GA12]*det3_245_145
                                   + pM[GA14]*det3_245_125 - pM[GA15]*det3_245_124;
   const Double_t det4_1245_1345 = pM[GA11]*det3_245_345 - pM[GA13]*det3_245_145
                                   + pM[GA14]*det3_245_135 - pM[GA15]*det3_245_134;
   const Double_t det4_1245_2345 = pM[GA12]*det3_245_345 - pM[GA13]*det3_245_245
                                   + pM[GA14]*det3_245_235 - pM[GA15]*det3_245_234;
   const Double_t det4_1345_0123 = pM[GA10]*det3_345_123 - pM[GA11]*det3_345_023
                                   + pM[GA12]*det3_345_013 - pM[GA13]*det3_345_012;
   const Double_t det4_1345_0124 = pM[GA10]*det3_345_124 - pM[GA11]*det3_345_024
                                   + pM[GA12]*det3_345_014 - pM[GA14]*det3_345_012;
   const Double_t det4_1345_0125 = pM[GA10]*det3_345_125 - pM[GA11]*det3_345_025
                                   + pM[GA12]*det3_345_015 - pM[GA15]*det3_345_012;
   const Double_t det4_1345_0134 = pM[GA10]*det3_345_134 - pM[GA11]*det3_345_034
                                   + pM[GA13]*det3_345_014 - pM[GA14]*det3_345_013;
   const Double_t det4_1345_0135 = pM[GA10]*det3_345_135 - pM[GA11]*det3_345_035
                                   + pM[GA13]*det3_345_015 - pM[GA15]*det3_345_013;
   const Double_t det4_1345_0145 = pM[GA10]*det3_345_145 - pM[GA11]*det3_345_045
                                   + pM[GA14]*det3_345_015 - pM[GA15]*det3_345_014;
   const Double_t det4_1345_0234 = pM[GA10]*det3_345_234 - pM[GA12]*det3_345_034
                                   + pM[GA13]*det3_345_024 - pM[GA14]*det3_345_023;
   const Double_t det4_1345_0235 = pM[GA10]*det3_345_235 - pM[GA12]*det3_345_035
                                   + pM[GA13]*det3_345_025 - pM[GA15]*det3_345_023;
   const Double_t det4_1345_0245 = pM[GA10]*det3_345_245 - pM[GA12]*det3_345_045
                                   + pM[GA14]*det3_345_025 - pM[GA15]*det3_345_024;
   const Double_t det4_1345_0345 = pM[GA10]*det3_345_345 - pM[GA13]*det3_345_045
                                   + pM[GA14]*det3_345_035 - pM[GA15]*det3_345_034;
   const Double_t det4_1345_1234 = pM[GA11]*det3_345_234 - pM[GA12]*det3_345_134
                                   + pM[GA13]*det3_345_124 - pM[GA14]*det3_345_123;
   const Double_t det4_1345_1235 = pM[GA11]*det3_345_235 - pM[GA12]*det3_345_135
                                   + pM[GA13]*det3_345_125 - pM[GA15]*det3_345_123;
   const Double_t det4_1345_1245 = pM[GA11]*det3_345_245 - pM[GA12]*det3_345_145
                                   + pM[GA14]*det3_345_125 - pM[GA15]*det3_345_124;
   const Double_t det4_1345_1345 = pM[GA11]*det3_345_345 - pM[GA13]*det3_345_145
                                   + pM[GA14]*det3_345_135 - pM[GA15]*det3_345_134;
   const Double_t det4_1345_2345 = pM[GA12]*det3_345_345 - pM[GA13]*det3_345_245
                                   + pM[GA14]*det3_345_235 - pM[GA15]*det3_345_234;
   const Double_t det4_2345_0123 = pM[GA20]*det3_345_123 - pM[GA21]*det3_345_023
                                   + pM[GA22]*det3_345_013 - pM[GA23]*det3_345_012;
   const Double_t det4_2345_0124 = pM[GA20]*det3_345_124 - pM[GA21]*det3_345_024
                                   + pM[GA22]*det3_345_014 - pM[GA24]*det3_345_012;
   const Double_t det4_2345_0125 = pM[GA20]*det3_345_125 - pM[GA21]*det3_345_025
                                   + pM[GA22]*det3_345_015 - pM[GA25]*det3_345_012;
   const Double_t det4_2345_0134 = pM[GA20]*det3_345_134 - pM[GA21]*det3_345_034
                                   + pM[GA23]*det3_345_014 - pM[GA24]*det3_345_013;
   const Double_t det4_2345_0135 = pM[GA20]*det3_345_135 - pM[GA21]*det3_345_035
                                   + pM[GA23]*det3_345_015 - pM[GA25]*det3_345_013;
   const Double_t det4_2345_0145 = pM[GA20]*det3_345_145 - pM[GA21]*det3_345_045
                                   + pM[GA24]*det3_345_015 - pM[GA25]*det3_345_014;
   const Double_t det4_2345_0234 = pM[GA20]*det3_345_234 - pM[GA22]*det3_345_034
                                   + pM[GA23]*det3_345_024 - pM[GA24]*det3_345_023;
   const Double_t det4_2345_0235 = pM[GA20]*det3_345_235 - pM[GA22]*det3_345_035
                                   + pM[GA23]*det3_345_025 - pM[GA25]*det3_345_023;
   const Double_t det4_2345_0245 = pM[GA20]*det3_345_245 - pM[GA22]*det3_345_045
                                   + pM[GA24]*det3_345_025 - pM[GA25]*det3_345_024;
   const Double_t det4_2345_0345 = pM[GA20]*det3_345_345 - pM[GA23]*det3_345_045
                                   + pM[GA24]*det3_345_035 - pM[GA25]*det3_345_034;
   const Double_t det4_2345_1234 = pM[GA21]*det3_345_234 - pM[GA22]*det3_345_134
                                   + pM[GA23]*det3_345_124 - pM[GA24]*det3_345_123;
   const Double_t det4_2345_1235 = pM[GA21]*det3_345_235 - pM[GA22]*det3_345_135
                                   + pM[GA23]*det3_345_125 - pM[GA25]*det3_345_123;
   const Double_t det4_2345_1245 = pM[GA21]*det3_345_245 - pM[GA22]*det3_345_145
                                   + pM[GA24]*det3_345_125 - pM[GA25]*det3_345_124;
   const Double_t det4_2345_1345 = pM[GA21]*det3_345_345 - pM[GA23]*det3_345_145
                                   + pM[GA24]*det3_345_135 - pM[GA25]*det3_345_134;
   const Double_t det4_2345_2345 = pM[GA22]*det3_345_345 - pM[GA23]*det3_345_245
                                   + pM[GA24]*det3_345_235 - pM[GA25]*det3_345_234;

  // Find all NECESSGARY 5x5 dets:  (36 of them)

   const Double_t det5_01234_01234 = pM[GA00]*det4_1234_1234 - pM[GA01]*det4_1234_0234
                                     + pM[GA02]*det4_1234_0134 - pM[GA03]*det4_1234_0124 + pM[GA04]*det4_1234_0123;
   const Double_t det5_01234_01235 = pM[GA00]*det4_1234_1235 - pM[GA01]*det4_1234_0235
                                     + pM[GA02]*det4_1234_0135 - pM[GA03]*det4_1234_0125 + pM[GA05]*det4_1234_0123;
   const Double_t det5_01234_01245 = pM[GA00]*det4_1234_1245 - pM[GA01]*det4_1234_0245
                                     + pM[GA02]*det4_1234_0145 - pM[GA04]*det4_1234_0125 + pM[GA05]*det4_1234_0124;
   const Double_t det5_01234_01345 = pM[GA00]*det4_1234_1345 - pM[GA01]*det4_1234_0345
                                     + pM[GA03]*det4_1234_0145 - pM[GA04]*det4_1234_0135 + pM[GA05]*det4_1234_0134;
   const Double_t det5_01234_02345 = pM[GA00]*det4_1234_2345 - pM[GA02]*det4_1234_0345
                                     + pM[GA03]*det4_1234_0245 - pM[GA04]*det4_1234_0235 + pM[GA05]*det4_1234_0234;
   const Double_t det5_01234_12345 = pM[GA01]*det4_1234_2345 - pM[GA02]*det4_1234_1345
                                     + pM[GA03]*det4_1234_1245 - pM[GA04]*det4_1234_1235 + pM[GA05]*det4_1234_1234;
   const Double_t det5_01235_01234 = pM[GA00]*det4_1235_1234 - pM[GA01]*det4_1235_0234
                                     + pM[GA02]*det4_1235_0134 - pM[GA03]*det4_1235_0124 + pM[GA04]*det4_1235_0123;
   const Double_t det5_01235_01235 = pM[GA00]*det4_1235_1235 - pM[GA01]*det4_1235_0235
                                     + pM[GA02]*det4_1235_0135 - pM[GA03]*det4_1235_0125 + pM[GA05]*det4_1235_0123;
   const Double_t det5_01235_01245 = pM[GA00]*det4_1235_1245 - pM[GA01]*det4_1235_0245
                                     + pM[GA02]*det4_1235_0145 - pM[GA04]*det4_1235_0125 + pM[GA05]*det4_1235_0124;
   const Double_t det5_01235_01345 = pM[GA00]*det4_1235_1345 - pM[GA01]*det4_1235_0345
                                     + pM[GA03]*det4_1235_0145 - pM[GA04]*det4_1235_0135 + pM[GA05]*det4_1235_0134;
   const Double_t det5_01235_02345 = pM[GA00]*det4_1235_2345 - pM[GA02]*det4_1235_0345
                                     + pM[GA03]*det4_1235_0245 - pM[GA04]*det4_1235_0235 + pM[GA05]*det4_1235_0234;
   const Double_t det5_01235_12345 = pM[GA01]*det4_1235_2345 - pM[GA02]*det4_1235_1345
                                     + pM[GA03]*det4_1235_1245 - pM[GA04]*det4_1235_1235 + pM[GA05]*det4_1235_1234;
   const Double_t det5_01245_01234 = pM[GA00]*det4_1245_1234 - pM[GA01]*det4_1245_0234
                                     + pM[GA02]*det4_1245_0134 - pM[GA03]*det4_1245_0124 + pM[GA04]*det4_1245_0123;
   const Double_t det5_01245_01235 = pM[GA00]*det4_1245_1235 - pM[GA01]*det4_1245_0235
                                     + pM[GA02]*det4_1245_0135 - pM[GA03]*det4_1245_0125 + pM[GA05]*det4_1245_0123;
   const Double_t det5_01245_01245 = pM[GA00]*det4_1245_1245 - pM[GA01]*det4_1245_0245
                                     + pM[GA02]*det4_1245_0145 - pM[GA04]*det4_1245_0125 + pM[GA05]*det4_1245_0124;
   const Double_t det5_01245_01345 = pM[GA00]*det4_1245_1345 - pM[GA01]*det4_1245_0345
                                     + pM[GA03]*det4_1245_0145 - pM[GA04]*det4_1245_0135 + pM[GA05]*det4_1245_0134;
   const Double_t det5_01245_02345 = pM[GA00]*det4_1245_2345 - pM[GA02]*det4_1245_0345
                                     + pM[GA03]*det4_1245_0245 - pM[GA04]*det4_1245_0235 + pM[GA05]*det4_1245_0234;
   const Double_t det5_01245_12345 = pM[GA01]*det4_1245_2345 - pM[GA02]*det4_1245_1345
                                     + pM[GA03]*det4_1245_1245 - pM[GA04]*det4_1245_1235 + pM[GA05]*det4_1245_1234;
   const Double_t det5_01345_01234 = pM[GA00]*det4_1345_1234 - pM[GA01]*det4_1345_0234
                                     + pM[GA02]*det4_1345_0134 - pM[GA03]*det4_1345_0124 + pM[GA04]*det4_1345_0123;
   const Double_t det5_01345_01235 = pM[GA00]*det4_1345_1235 - pM[GA01]*det4_1345_0235
                                     + pM[GA02]*det4_1345_0135 - pM[GA03]*det4_1345_0125 + pM[GA05]*det4_1345_0123;
   const Double_t det5_01345_01245 = pM[GA00]*det4_1345_1245 - pM[GA01]*det4_1345_0245
                                     + pM[GA02]*det4_1345_0145 - pM[GA04]*det4_1345_0125 + pM[GA05]*det4_1345_0124;
   const Double_t det5_01345_01345 = pM[GA00]*det4_1345_1345 - pM[GA01]*det4_1345_0345
                                     + pM[GA03]*det4_1345_0145 - pM[GA04]*det4_1345_0135 + pM[GA05]*det4_1345_0134;
   const Double_t det5_01345_02345 = pM[GA00]*det4_1345_2345 - pM[GA02]*det4_1345_0345
                                     + pM[GA03]*det4_1345_0245 - pM[GA04]*det4_1345_0235 + pM[GA05]*det4_1345_0234;
   const Double_t det5_01345_12345 = pM[GA01]*det4_1345_2345 - pM[GA02]*det4_1345_1345
                                     + pM[GA03]*det4_1345_1245 - pM[GA04]*det4_1345_1235 + pM[GA05]*det4_1345_1234;
   const Double_t det5_02345_01234 = pM[GA00]*det4_2345_1234 - pM[GA01]*det4_2345_0234
                                     + pM[GA02]*det4_2345_0134 - pM[GA03]*det4_2345_0124 + pM[GA04]*det4_2345_0123;
   const Double_t det5_02345_01235 = pM[GA00]*det4_2345_1235 - pM[GA01]*det4_2345_0235
                                     + pM[GA02]*det4_2345_0135 - pM[GA03]*det4_2345_0125 + pM[GA05]*det4_2345_0123;
   const Double_t det5_02345_01245 = pM[GA00]*det4_2345_1245 - pM[GA01]*det4_2345_0245
                                     + pM[GA02]*det4_2345_0145 - pM[GA04]*det4_2345_0125 + pM[GA05]*det4_2345_0124;
   const Double_t det5_02345_01345 = pM[GA00]*det4_2345_1345 - pM[GA01]*det4_2345_0345
                                     + pM[GA03]*det4_2345_0145 - pM[GA04]*det4_2345_0135 + pM[GA05]*det4_2345_0134;
   const Double_t det5_02345_02345 = pM[GA00]*det4_2345_2345 - pM[GA02]*det4_2345_0345
                                     + pM[GA03]*det4_2345_0245 - pM[GA04]*det4_2345_0235 + pM[GA05]*det4_2345_0234;
   const Double_t det5_02345_12345 = pM[GA01]*det4_2345_2345 - pM[GA02]*det4_2345_1345
                                     + pM[GA03]*det4_2345_1245 - pM[GA04]*det4_2345_1235 + pM[GA05]*det4_2345_1234;
   const Double_t det5_12345_01234 = pM[GA10]*det4_2345_1234 - pM[GA11]*det4_2345_0234
                                     + pM[GA12]*det4_2345_0134 - pM[GA13]*det4_2345_0124 + pM[GA14]*det4_2345_0123;
   const Double_t det5_12345_01235 = pM[GA10]*det4_2345_1235 - pM[GA11]*det4_2345_0235
                                     + pM[GA12]*det4_2345_0135 - pM[GA13]*det4_2345_0125 + pM[GA15]*det4_2345_0123;
   const Double_t det5_12345_01245 = pM[GA10]*det4_2345_1245 - pM[GA11]*det4_2345_0245
                                     + pM[GA12]*det4_2345_0145 - pM[GA14]*det4_2345_0125 + pM[GA15]*det4_2345_0124;
   const Double_t det5_12345_01345 = pM[GA10]*det4_2345_1345 - pM[GA11]*det4_2345_0345
                                     + pM[GA13]*det4_2345_0145 - pM[GA14]*det4_2345_0135 + pM[GA15]*det4_2345_0134;
   const Double_t det5_12345_02345 = pM[GA10]*det4_2345_2345 - pM[GA12]*det4_2345_0345
                                     + pM[GA13]*det4_2345_0245 - pM[GA14]*det4_2345_0235 + pM[GA15]*det4_2345_0234;
   const Double_t det5_12345_12345 = pM[GA11]*det4_2345_2345 - pM[GA12]*det4_2345_1345
                                     + pM[GA13]*det4_2345_1245 - pM[GA14]*det4_2345_1235 + pM[GA15]*det4_2345_1234;

  // Find the determinant

   const Double_t det = pM[GA00]*det5_12345_12345 - pM[GA01]*det5_12345_02345 + pM[GA02]*det5_12345_01345
                        - pM[GA03]*det5_12345_01245 + pM[GA04]*det5_12345_01235 - pM[GA05]*det5_12345_01234;
   if (determ)
      *determ = det;

   if ( det == 0 ) {
      Error("Inv6x6","matrix is singular");
      return kFALSE;
   }

   const Double_t oneOverDet = 1.0/det;
   const Double_t mn1OverDet = - oneOverDet;

   pM[GA00] =  det5_12345_12345*oneOverDet;
   pM[GA01] =  det5_02345_12345*mn1OverDet;
   pM[GA02] =  det5_01345_12345*oneOverDet;
   pM[GA03] =  det5_01245_12345*mn1OverDet;
   pM[GA04] =  det5_01235_12345*oneOverDet;
   pM[GA05] =  det5_01234_12345*mn1OverDet;

   pM[GA10] =  det5_12345_02345*mn1OverDet;
   pM[GA11] =  det5_02345_02345*oneOverDet;
   pM[GA12] =  det5_01345_02345*mn1OverDet;
   pM[GA13] =  det5_01245_02345*oneOverDet;
   pM[GA14] =  det5_01235_02345*mn1OverDet;
   pM[GA15] =  det5_01234_02345*oneOverDet;

   pM[GA20] =  det5_12345_01345*oneOverDet;
   pM[GA21] =  det5_02345_01345*mn1OverDet;
   pM[GA22] =  det5_01345_01345*oneOverDet;
   pM[GA23] =  det5_01245_01345*mn1OverDet;
   pM[GA24] =  det5_01235_01345*oneOverDet;
   pM[GA25] =  det5_01234_01345*mn1OverDet;

   pM[GA30] =  det5_12345_01245*mn1OverDet;
   pM[GA31] =  det5_02345_01245*oneOverDet;
   pM[GA32] =  det5_01345_01245*mn1OverDet;
   pM[GA33] =  det5_01245_01245*oneOverDet;
   pM[GA34] =  det5_01235_01245*mn1OverDet;
   pM[GA35] =  det5_01234_01245*oneOverDet;

   pM[GA40] =  det5_12345_01235*oneOverDet;
   pM[GA41] =  det5_02345_01235*mn1OverDet;
   pM[GA42] =  det5_01345_01235*oneOverDet;
   pM[GA43] =  det5_01245_01235*mn1OverDet;
   pM[GA44] =  det5_01235_01235*oneOverDet;
   pM[GA45] =  det5_01234_01235*mn1OverDet;

   pM[GA50] =  det5_12345_01234*mn1OverDet;
   pM[GA51] =  det5_02345_01234*oneOverDet;
   pM[GA52] =  det5_01345_01234*mn1OverDet;
   pM[GA53] =  det5_01245_01234*oneOverDet;
   pM[GA54] =  det5_01235_01234*mn1OverDet;
   pM[GA55] =  det5_01234_01234*oneOverDet;

   return kTRUE;
}

#include "TMatrixFfwd.h"

template Bool_t TMatrixTCramerInv::Inv2x2<Float_t>(TMatrixF&,Double_t*);
template Bool_t TMatrixTCramerInv::Inv3x3<Float_t>(TMatrixF&,Double_t*);
template Bool_t TMatrixTCramerInv::Inv4x4<Float_t>(TMatrixF&,Double_t*);
template Bool_t TMatrixTCramerInv::Inv5x5<Float_t>(TMatrixF&,Double_t*);
template Bool_t TMatrixTCramerInv::Inv6x6<Float_t>(TMatrixF&,Double_t*);

#include "TMatrixDfwd.h"

template Bool_t TMatrixTCramerInv::Inv2x2<Double_t>(TMatrixD&,Double_t*);
template Bool_t TMatrixTCramerInv::Inv3x3<Double_t>(TMatrixD&,Double_t*);
template Bool_t TMatrixTCramerInv::Inv4x4<Double_t>(TMatrixD&,Double_t*);
template Bool_t TMatrixTCramerInv::Inv5x5<Double_t>(TMatrixD&,Double_t*);
template Bool_t TMatrixTCramerInv::Inv6x6<Double_t>(TMatrixD&,Double_t*);
