// @(#)root/base:$Name:  $:$Id: TMatrixDCramerInv.cxx,v 1.1 2004/01/25 20:33:32 brun Exp $
// Authors: Fons Rademakers, Eddy Offermann  Jan 2004

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMatrixDCramerInv                                                    //
//                                                                      //
// Encapsulate Cramer Inversion routines.                               //
//                                                                      //
// The 4x4, 5x5 and 6x6 are adapted from routines written by            //
// Mark Fischler and Steven Haywood as part of the CLHEP package        //
//                                                                      //
// Although for sizes <= 6x6 the Cramer Inversion has a gain in speed   //
// compared to factorization schemes (like LU) , one pays a price in    //
// accuracy  .                                                          //
//                                                                      //
// For Example:                                                         //
//  H * H^-1 = U, where H is a 5x5 Hilbert matrix                       //
//                      U is a 5x5 Unity matrix                         //
//                                                                      //
// LU    : |U_jk| < 10e-13 for  j!=k                                    //
// Cramer: |U_jk| < 10e-7  for  j!=k                                    //
//                                                                      //
//  however Cramer algorithm is about 10 (!) times faster               //
//////////////////////////////////////////////////////////////////////////

#include "TMatrixDCramerInv.h"

//______________________________________________________________________________
Int_t TMatrixDCramerInv::Inv2x2(TMatrixD &m,Double_t *determ)
{
  Double_t *pM = m.GetMatrixArray();

  const Double_t det = pM[0] * pM[3] - pM[2] * pM[1];

  if (determ)
    *determ = det;

  if ( det == 0 )
    return kFALSE;

  const Double_t tmp1 = -pM[2] / det;
  const Double_t tmp2 =  pM[3] / det;
  pM[2] = -pM[1] / det;
  pM[3] =  pM[0] / det;
  pM[1] = tmp1;
  pM[0] = tmp2;

  return kTRUE;
}

//______________________________________________________________________________
Int_t TMatrixDCramerInv::Inv3x3(TMatrixD &m,Double_t *determ)
{
  Double_t *pM = m.GetMatrixArray();

  const Double_t c00 = pM[4] * pM[8] - pM[5] * pM[7];
  const Double_t c01 = pM[5] * pM[6] - pM[3] * pM[8];
  const Double_t c02 = pM[3] * pM[7] - pM[4] * pM[6];
  const Double_t c10 = pM[7] * pM[2] - pM[8] * pM[1];
  const Double_t c11 = pM[8] * pM[0] - pM[6] * pM[2];
  const Double_t c12 = pM[6] * pM[1] - pM[7] * pM[0];
  const Double_t c20 = pM[1] * pM[5] - pM[2] * pM[4];
  const Double_t c21 = pM[2] * pM[3] - pM[0] * pM[5];
  const Double_t c22 = pM[0] * pM[4] - pM[1] * pM[3];

  const Double_t det = pM[0] * c00 + pM[3] * c10 + pM[6] * c20;

  if (determ)
    *determ = det;

  if ( det == 0 )
    return kFALSE;

  pM[0] = c00 / det;
  pM[1] = c01 / det;
  pM[2] = c02 / det;
  pM[3] = c10 / det;
  pM[4] = c11 / det;
  pM[5] = c12 / det;
  pM[6] = c20 / det;
  pM[7] = c21 / det;
  pM[8] = c22 / det;

  return kTRUE;
}

// Fij are indices for a 4x4 matrix.

#define F00 0
#define F01 1
#define F02 2
#define F03 3

#define F10 4
#define F11 5
#define F12 6
#define F13 7

#define F20 8
#define F21 9
#define F22 10
#define F23 11

#define F30 12
#define F31 13
#define F32 14
#define F33 15

//______________________________________________________________________________
Int_t TMatrixDCramerInv::Inv4x4(TMatrixD &m,Double_t *determ)
{
  Double_t *pM = m.GetMatrixArray();

  // Find all NECESSARY 2x2 dets:  (18 of them)

  const Double_t Det2_12_01 = pM[F10]*pM[F21] - pM[F11]*pM[F20];
  const Double_t Det2_12_02 = pM[F10]*pM[F22] - pM[F12]*pM[F20];
  const Double_t Det2_12_03 = pM[F10]*pM[F23] - pM[F13]*pM[F20];
  const Double_t Det2_12_13 = pM[F11]*pM[F23] - pM[F13]*pM[F21];
  const Double_t Det2_12_23 = pM[F12]*pM[F23] - pM[F13]*pM[F22];
  const Double_t Det2_12_12 = pM[F11]*pM[F22] - pM[F12]*pM[F21];
  const Double_t Det2_13_01 = pM[F10]*pM[F31] - pM[F11]*pM[F30];
  const Double_t Det2_13_02 = pM[F10]*pM[F32] - pM[F12]*pM[F30];
  const Double_t Det2_13_03 = pM[F10]*pM[F33] - pM[F13]*pM[F30];
  const Double_t Det2_13_12 = pM[F11]*pM[F32] - pM[F12]*pM[F31];
  const Double_t Det2_13_13 = pM[F11]*pM[F33] - pM[F13]*pM[F31];
  const Double_t Det2_13_23 = pM[F12]*pM[F33] - pM[F13]*pM[F32];
  const Double_t Det2_23_01 = pM[F20]*pM[F31] - pM[F21]*pM[F30];
  const Double_t Det2_23_02 = pM[F20]*pM[F32] - pM[F22]*pM[F30];
  const Double_t Det2_23_03 = pM[F20]*pM[F33] - pM[F23]*pM[F30];
  const Double_t Det2_23_12 = pM[F21]*pM[F32] - pM[F22]*pM[F31];
  const Double_t Det2_23_13 = pM[F21]*pM[F33] - pM[F23]*pM[F31];
  const Double_t Det2_23_23 = pM[F22]*pM[F33] - pM[F23]*pM[F32];

  // Find all NECESSARY 3x3 dets:   (16 of them)

  const Double_t Det3_012_012 = pM[F00]*Det2_12_12 - pM[F01]*Det2_12_02 
                                + pM[F02]*Det2_12_01;
  const Double_t Det3_012_013 = pM[F00]*Det2_12_13 - pM[F01]*Det2_12_03 
                                + pM[F03]*Det2_12_01;
  const Double_t Det3_012_023 = pM[F00]*Det2_12_23 - pM[F02]*Det2_12_03 
                                + pM[F03]*Det2_12_02;
  const Double_t Det3_012_123 = pM[F01]*Det2_12_23 - pM[F02]*Det2_12_13 
                                + pM[F03]*Det2_12_12;
  const Double_t Det3_013_012 = pM[F00]*Det2_13_12 - pM[F01]*Det2_13_02 
                                + pM[F02]*Det2_13_01;
  const Double_t Det3_013_013 = pM[F00]*Det2_13_13 - pM[F01]*Det2_13_03
                                + pM[F03]*Det2_13_01;
  const Double_t Det3_013_023 = pM[F00]*Det2_13_23 - pM[F02]*Det2_13_03
                                + pM[F03]*Det2_13_02;
  const Double_t Det3_013_123 = pM[F01]*Det2_13_23 - pM[F02]*Det2_13_13
                                + pM[F03]*Det2_13_12;
  const Double_t Det3_023_012 = pM[F00]*Det2_23_12 - pM[F01]*Det2_23_02 
                                + pM[F02]*Det2_23_01;
  const Double_t Det3_023_013 = pM[F00]*Det2_23_13 - pM[F01]*Det2_23_03
                                + pM[F03]*Det2_23_01;
  const Double_t Det3_023_023 = pM[F00]*Det2_23_23 - pM[F02]*Det2_23_03
                                + pM[F03]*Det2_23_02;
  const Double_t Det3_023_123 = pM[F01]*Det2_23_23 - pM[F02]*Det2_23_13
                                + pM[F03]*Det2_23_12;
  const Double_t Det3_123_012 = pM[F10]*Det2_23_12 - pM[F11]*Det2_23_02 
                                + pM[F12]*Det2_23_01;
  const Double_t Det3_123_013 = pM[F10]*Det2_23_13 - pM[F11]*Det2_23_03 
                                + pM[F13]*Det2_23_01;
  const Double_t Det3_123_023 = pM[F10]*Det2_23_23 - pM[F12]*Det2_23_03 
                                + pM[F13]*Det2_23_02;
  const Double_t Det3_123_123 = pM[F11]*Det2_23_23 - pM[F12]*Det2_23_13 
                                + pM[F13]*Det2_23_12;

  // Find the 4x4 det:

  const Double_t det = pM[F00]*Det3_123_123 - pM[F01]*Det3_123_023 
                       + pM[F02]*Det3_123_013 - pM[F03]*Det3_123_012;
  if (determ)
    *determ = det;

  if ( det == 0 )
    return kFALSE;

  const Double_t oneOverDet = 1.0/det;
  const Double_t mn1OverDet = - oneOverDet;

  pM[F00] =  Det3_123_123 * oneOverDet;
  pM[F01] =  Det3_023_123 * mn1OverDet;
  pM[F02] =  Det3_013_123 * oneOverDet;
  pM[F03] =  Det3_012_123 * mn1OverDet;

  pM[F10] =  Det3_123_023 * mn1OverDet;
  pM[F11] =  Det3_023_023 * oneOverDet;
  pM[F12] =  Det3_013_023 * mn1OverDet;
  pM[F13] =  Det3_012_023 * oneOverDet;

  pM[F20] =  Det3_123_013 * oneOverDet;
  pM[F21] =  Det3_023_013 * mn1OverDet;
  pM[F22] =  Det3_013_013 * oneOverDet;
  pM[F23] =  Det3_012_013 * mn1OverDet;

  pM[F30] =  Det3_123_012 * mn1OverDet;
  pM[F31] =  Det3_023_012 * oneOverDet;
  pM[F32] =  Det3_013_012 * mn1OverDet;
  pM[F33] =  Det3_012_012 * oneOverDet;

  return kTRUE;
}

// Mij are indices for a 5x5 matrix.

#define M00 0
#define M01 1
#define M02 2
#define M03 3
#define M04 4

#define M10 5
#define M11 6
#define M12 7
#define M13 8
#define M14 9

#define M20 10
#define M21 11
#define M22 12
#define M23 13
#define M24 14

#define M30 15
#define M31 16
#define M32 17
#define M33 18
#define M34 19

#define M40 20
#define M41 21
#define M42 22
#define M43 23
#define M44 24

//______________________________________________________________________________
Int_t TMatrixDCramerInv::Inv5x5(TMatrixD &m,Double_t *determ)
{
  Double_t *pM = m.GetMatrixArray();

  // Find all NECESSARY 2x2 dets:  (30 of them)

  const Double_t Det2_23_01 = pM[M20]*pM[M31] - pM[M21]*pM[M30];
  const Double_t Det2_23_02 = pM[M20]*pM[M32] - pM[M22]*pM[M30];
  const Double_t Det2_23_03 = pM[M20]*pM[M33] - pM[M23]*pM[M30];
  const Double_t Det2_23_04 = pM[M20]*pM[M34] - pM[M24]*pM[M30];
  const Double_t Det2_23_12 = pM[M21]*pM[M32] - pM[M22]*pM[M31];
  const Double_t Det2_23_13 = pM[M21]*pM[M33] - pM[M23]*pM[M31];
  const Double_t Det2_23_14 = pM[M21]*pM[M34] - pM[M24]*pM[M31];
  const Double_t Det2_23_23 = pM[M22]*pM[M33] - pM[M23]*pM[M32];
  const Double_t Det2_23_24 = pM[M22]*pM[M34] - pM[M24]*pM[M32];
  const Double_t Det2_23_34 = pM[M23]*pM[M34] - pM[M24]*pM[M33];
  const Double_t Det2_24_01 = pM[M20]*pM[M41] - pM[M21]*pM[M40];
  const Double_t Det2_24_02 = pM[M20]*pM[M42] - pM[M22]*pM[M40];
  const Double_t Det2_24_03 = pM[M20]*pM[M43] - pM[M23]*pM[M40];
  const Double_t Det2_24_04 = pM[M20]*pM[M44] - pM[M24]*pM[M40];
  const Double_t Det2_24_12 = pM[M21]*pM[M42] - pM[M22]*pM[M41];
  const Double_t Det2_24_13 = pM[M21]*pM[M43] - pM[M23]*pM[M41];
  const Double_t Det2_24_14 = pM[M21]*pM[M44] - pM[M24]*pM[M41];
  const Double_t Det2_24_23 = pM[M22]*pM[M43] - pM[M23]*pM[M42];
  const Double_t Det2_24_24 = pM[M22]*pM[M44] - pM[M24]*pM[M42];
  const Double_t Det2_24_34 = pM[M23]*pM[M44] - pM[M24]*pM[M43];
  const Double_t Det2_34_01 = pM[M30]*pM[M41] - pM[M31]*pM[M40];
  const Double_t Det2_34_02 = pM[M30]*pM[M42] - pM[M32]*pM[M40];
  const Double_t Det2_34_03 = pM[M30]*pM[M43] - pM[M33]*pM[M40];
  const Double_t Det2_34_04 = pM[M30]*pM[M44] - pM[M34]*pM[M40];
  const Double_t Det2_34_12 = pM[M31]*pM[M42] - pM[M32]*pM[M41];
  const Double_t Det2_34_13 = pM[M31]*pM[M43] - pM[M33]*pM[M41];
  const Double_t Det2_34_14 = pM[M31]*pM[M44] - pM[M34]*pM[M41];
  const Double_t Det2_34_23 = pM[M32]*pM[M43] - pM[M33]*pM[M42];
  const Double_t Det2_34_24 = pM[M32]*pM[M44] - pM[M34]*pM[M42];
  const Double_t Det2_34_34 = pM[M33]*pM[M44] - pM[M34]*pM[M43];

  // Find all NECESSARY 3x3 dets:   (40 of them)

  const Double_t Det3_123_012 = pM[M10]*Det2_23_12 - pM[M11]*Det2_23_02 + pM[M12]*Det2_23_01;
  const Double_t Det3_123_013 = pM[M10]*Det2_23_13 - pM[M11]*Det2_23_03 + pM[M13]*Det2_23_01;
  const Double_t Det3_123_014 = pM[M10]*Det2_23_14 - pM[M11]*Det2_23_04 + pM[M14]*Det2_23_01;
  const Double_t Det3_123_023 = pM[M10]*Det2_23_23 - pM[M12]*Det2_23_03 + pM[M13]*Det2_23_02;
  const Double_t Det3_123_024 = pM[M10]*Det2_23_24 - pM[M12]*Det2_23_04 + pM[M14]*Det2_23_02;
  const Double_t Det3_123_034 = pM[M10]*Det2_23_34 - pM[M13]*Det2_23_04 + pM[M14]*Det2_23_03;
  const Double_t Det3_123_123 = pM[M11]*Det2_23_23 - pM[M12]*Det2_23_13 + pM[M13]*Det2_23_12;
  const Double_t Det3_123_124 = pM[M11]*Det2_23_24 - pM[M12]*Det2_23_14 + pM[M14]*Det2_23_12;
  const Double_t Det3_123_134 = pM[M11]*Det2_23_34 - pM[M13]*Det2_23_14 + pM[M14]*Det2_23_13;
  const Double_t Det3_123_234 = pM[M12]*Det2_23_34 - pM[M13]*Det2_23_24 + pM[M14]*Det2_23_23;
  const Double_t Det3_124_012 = pM[M10]*Det2_24_12 - pM[M11]*Det2_24_02 + pM[M12]*Det2_24_01;
  const Double_t Det3_124_013 = pM[M10]*Det2_24_13 - pM[M11]*Det2_24_03 + pM[M13]*Det2_24_01;
  const Double_t Det3_124_014 = pM[M10]*Det2_24_14 - pM[M11]*Det2_24_04 + pM[M14]*Det2_24_01;
  const Double_t Det3_124_023 = pM[M10]*Det2_24_23 - pM[M12]*Det2_24_03 + pM[M13]*Det2_24_02;
  const Double_t Det3_124_024 = pM[M10]*Det2_24_24 - pM[M12]*Det2_24_04 + pM[M14]*Det2_24_02;
  const Double_t Det3_124_034 = pM[M10]*Det2_24_34 - pM[M13]*Det2_24_04 + pM[M14]*Det2_24_03;
  const Double_t Det3_124_123 = pM[M11]*Det2_24_23 - pM[M12]*Det2_24_13 + pM[M13]*Det2_24_12;
  const Double_t Det3_124_124 = pM[M11]*Det2_24_24 - pM[M12]*Det2_24_14 + pM[M14]*Det2_24_12;
  const Double_t Det3_124_134 = pM[M11]*Det2_24_34 - pM[M13]*Det2_24_14 + pM[M14]*Det2_24_13;
  const Double_t Det3_124_234 = pM[M12]*Det2_24_34 - pM[M13]*Det2_24_24 + pM[M14]*Det2_24_23;
  const Double_t Det3_134_012 = pM[M10]*Det2_34_12 - pM[M11]*Det2_34_02 + pM[M12]*Det2_34_01;
  const Double_t Det3_134_013 = pM[M10]*Det2_34_13 - pM[M11]*Det2_34_03 + pM[M13]*Det2_34_01;
  const Double_t Det3_134_014 = pM[M10]*Det2_34_14 - pM[M11]*Det2_34_04 + pM[M14]*Det2_34_01;
  const Double_t Det3_134_023 = pM[M10]*Det2_34_23 - pM[M12]*Det2_34_03 + pM[M13]*Det2_34_02;
  const Double_t Det3_134_024 = pM[M10]*Det2_34_24 - pM[M12]*Det2_34_04 + pM[M14]*Det2_34_02;
  const Double_t Det3_134_034 = pM[M10]*Det2_34_34 - pM[M13]*Det2_34_04 + pM[M14]*Det2_34_03;
  const Double_t Det3_134_123 = pM[M11]*Det2_34_23 - pM[M12]*Det2_34_13 + pM[M13]*Det2_34_12;
  const Double_t Det3_134_124 = pM[M11]*Det2_34_24 - pM[M12]*Det2_34_14 + pM[M14]*Det2_34_12;
  const Double_t Det3_134_134 = pM[M11]*Det2_34_34 - pM[M13]*Det2_34_14 + pM[M14]*Det2_34_13;
  const Double_t Det3_134_234 = pM[M12]*Det2_34_34 - pM[M13]*Det2_34_24 + pM[M14]*Det2_34_23;
  const Double_t Det3_234_012 = pM[M20]*Det2_34_12 - pM[M21]*Det2_34_02 + pM[M22]*Det2_34_01;
  const Double_t Det3_234_013 = pM[M20]*Det2_34_13 - pM[M21]*Det2_34_03 + pM[M23]*Det2_34_01;
  const Double_t Det3_234_014 = pM[M20]*Det2_34_14 - pM[M21]*Det2_34_04 + pM[M24]*Det2_34_01;
  const Double_t Det3_234_023 = pM[M20]*Det2_34_23 - pM[M22]*Det2_34_03 + pM[M23]*Det2_34_02;
  const Double_t Det3_234_024 = pM[M20]*Det2_34_24 - pM[M22]*Det2_34_04 + pM[M24]*Det2_34_02;
  const Double_t Det3_234_034 = pM[M20]*Det2_34_34 - pM[M23]*Det2_34_04 + pM[M24]*Det2_34_03;
  const Double_t Det3_234_123 = pM[M21]*Det2_34_23 - pM[M22]*Det2_34_13 + pM[M23]*Det2_34_12;
  const Double_t Det3_234_124 = pM[M21]*Det2_34_24 - pM[M22]*Det2_34_14 + pM[M24]*Det2_34_12;
  const Double_t Det3_234_134 = pM[M21]*Det2_34_34 - pM[M23]*Det2_34_14 + pM[M24]*Det2_34_13;
  const Double_t Det3_234_234 = pM[M22]*Det2_34_34 - pM[M23]*Det2_34_24 + pM[M24]*Det2_34_23;

  // Find all NECESSARY 4x4 dets:   (25 of them)

  const Double_t Det4_0123_0123 = pM[M00]*Det3_123_123 - pM[M01]*Det3_123_023 
                                  + pM[M02]*Det3_123_013 - pM[M03]*Det3_123_012;
  const Double_t Det4_0123_0124 = pM[M00]*Det3_123_124 - pM[M01]*Det3_123_024 
                                  + pM[M02]*Det3_123_014 - pM[M04]*Det3_123_012;
  const Double_t Det4_0123_0134 = pM[M00]*Det3_123_134 - pM[M01]*Det3_123_034 
                                  + pM[M03]*Det3_123_014 - pM[M04]*Det3_123_013;
  const Double_t Det4_0123_0234 = pM[M00]*Det3_123_234 - pM[M02]*Det3_123_034 
                                  + pM[M03]*Det3_123_024 - pM[M04]*Det3_123_023;
  const Double_t Det4_0123_1234 = pM[M01]*Det3_123_234 - pM[M02]*Det3_123_134 
                                  + pM[M03]*Det3_123_124 - pM[M04]*Det3_123_123;
  const Double_t Det4_0124_0123 = pM[M00]*Det3_124_123 - pM[M01]*Det3_124_023 
                                  + pM[M02]*Det3_124_013 - pM[M03]*Det3_124_012;
  const Double_t Det4_0124_0124 = pM[M00]*Det3_124_124 - pM[M01]*Det3_124_024 
                                  + pM[M02]*Det3_124_014 - pM[M04]*Det3_124_012;
  const Double_t Det4_0124_0134 = pM[M00]*Det3_124_134 - pM[M01]*Det3_124_034 
                                  + pM[M03]*Det3_124_014 - pM[M04]*Det3_124_013;
  const Double_t Det4_0124_0234 = pM[M00]*Det3_124_234 - pM[M02]*Det3_124_034 
                                  + pM[M03]*Det3_124_024 - pM[M04]*Det3_124_023;
  const Double_t Det4_0124_1234 = pM[M01]*Det3_124_234 - pM[M02]*Det3_124_134 
                                  + pM[M03]*Det3_124_124 - pM[M04]*Det3_124_123;
  const Double_t Det4_0134_0123 = pM[M00]*Det3_134_123 - pM[M01]*Det3_134_023 
                                  + pM[M02]*Det3_134_013 - pM[M03]*Det3_134_012;
  const Double_t Det4_0134_0124 = pM[M00]*Det3_134_124 - pM[M01]*Det3_134_024 
                                  + pM[M02]*Det3_134_014 - pM[M04]*Det3_134_012;
  const Double_t Det4_0134_0134 = pM[M00]*Det3_134_134 - pM[M01]*Det3_134_034 
                                  + pM[M03]*Det3_134_014 - pM[M04]*Det3_134_013;
  const Double_t Det4_0134_0234 = pM[M00]*Det3_134_234 - pM[M02]*Det3_134_034 
                                  + pM[M03]*Det3_134_024 - pM[M04]*Det3_134_023;
  const Double_t Det4_0134_1234 = pM[M01]*Det3_134_234 - pM[M02]*Det3_134_134 
                                  + pM[M03]*Det3_134_124 - pM[M04]*Det3_134_123;
  const Double_t Det4_0234_0123 = pM[M00]*Det3_234_123 - pM[M01]*Det3_234_023 
                                  + pM[M02]*Det3_234_013 - pM[M03]*Det3_234_012;
  const Double_t Det4_0234_0124 = pM[M00]*Det3_234_124 - pM[M01]*Det3_234_024 
                                  + pM[M02]*Det3_234_014 - pM[M04]*Det3_234_012;
  const Double_t Det4_0234_0134 = pM[M00]*Det3_234_134 - pM[M01]*Det3_234_034 
                                  + pM[M03]*Det3_234_014 - pM[M04]*Det3_234_013;
  const Double_t Det4_0234_0234 = pM[M00]*Det3_234_234 - pM[M02]*Det3_234_034 
                                  + pM[M03]*Det3_234_024 - pM[M04]*Det3_234_023;
  const Double_t Det4_0234_1234 = pM[M01]*Det3_234_234 - pM[M02]*Det3_234_134 
                                  + pM[M03]*Det3_234_124 - pM[M04]*Det3_234_123;
  const Double_t Det4_1234_0123 = pM[M10]*Det3_234_123 - pM[M11]*Det3_234_023 
                                  + pM[M12]*Det3_234_013 - pM[M13]*Det3_234_012;
  const Double_t Det4_1234_0124 = pM[M10]*Det3_234_124 - pM[M11]*Det3_234_024 
                                  + pM[M12]*Det3_234_014 - pM[M14]*Det3_234_012;
  const Double_t Det4_1234_0134 = pM[M10]*Det3_234_134 - pM[M11]*Det3_234_034 
                                  + pM[M13]*Det3_234_014 - pM[M14]*Det3_234_013;
  const Double_t Det4_1234_0234 = pM[M10]*Det3_234_234 - pM[M12]*Det3_234_034 
                                  + pM[M13]*Det3_234_024 - pM[M14]*Det3_234_023;
  const Double_t Det4_1234_1234 = pM[M11]*Det3_234_234 - pM[M12]*Det3_234_134 
                                  + pM[M13]*Det3_234_124 - pM[M14]*Det3_234_123;

  // Find the 5x5 det:

  const Double_t det = pM[M00]*Det4_1234_1234 - pM[M01]*Det4_1234_0234 + pM[M02]*Det4_1234_0134 
                       - pM[M03]*Det4_1234_0124 + pM[M04]*Det4_1234_0123;
  if (determ)
    *determ = det;

  if ( det == 0 )
    return kFALSE;

  const Double_t oneOverDet = 1.0/det;
  const Double_t mn1OverDet = - oneOverDet;

  pM[M00] =  Det4_1234_1234 * oneOverDet;
  pM[M01] =  Det4_0234_1234 * mn1OverDet;
  pM[M02] =  Det4_0134_1234 * oneOverDet;
  pM[M03] =  Det4_0124_1234 * mn1OverDet;
  pM[M04] =  Det4_0123_1234 * oneOverDet;

  pM[M10] =  Det4_1234_0234 * mn1OverDet;
  pM[M11] =  Det4_0234_0234 * oneOverDet;
  pM[M12] =  Det4_0134_0234 * mn1OverDet;
  pM[M13] =  Det4_0124_0234 * oneOverDet;
  pM[M14] =  Det4_0123_0234 * mn1OverDet;

  pM[M20] =  Det4_1234_0134 * oneOverDet;
  pM[M21] =  Det4_0234_0134 * mn1OverDet;
  pM[M22] =  Det4_0134_0134 * oneOverDet;
  pM[M23] =  Det4_0124_0134 * mn1OverDet;
  pM[M24] =  Det4_0123_0134 * oneOverDet;

  pM[M30] =  Det4_1234_0124 * mn1OverDet;
  pM[M31] =  Det4_0234_0124 * oneOverDet;
  pM[M32] =  Det4_0134_0124 * mn1OverDet;
  pM[M33] =  Det4_0124_0124 * oneOverDet;
  pM[M34] =  Det4_0123_0124 * mn1OverDet;

  pM[M40] =  Det4_1234_0123 * oneOverDet;
  pM[M41] =  Det4_0234_0123 * mn1OverDet;
  pM[M42] =  Det4_0134_0123 * oneOverDet;
  pM[M43] =  Det4_0124_0123 * mn1OverDet;
  pM[M44] =  Det4_0123_0123 * oneOverDet;

  return kTRUE;
}

// Aij are indices for a 6x6 matrix.

#define A00 0
#define A01 1
#define A02 2
#define A03 3
#define A04 4
#define A05 5

#define A10 6
#define A11 7
#define A12 8
#define A13 9
#define A14 10
#define A15 11

#define A20 12
#define A21 13
#define A22 14
#define A23 15
#define A24 16
#define A25 17

#define A30 18
#define A31 19
#define A32 20
#define A33 21
#define A34 22
#define A35 23

#define A40 24
#define A41 25
#define A42 26
#define A43 27
#define A44 28
#define A45 29

#define A50 30
#define A51 31
#define A52 32
#define A53 33
#define A54 34
#define A55 35

//______________________________________________________________________________
Int_t TMatrixDCramerInv::Inv6x6(TMatrixD &m,Double_t *determ)
{
  Double_t *pM = m.GetMatrixArray();

  // Find all NECESSARY 2x2 dets:  (45 of them)

  const Double_t Det2_34_01 = pM[A30]*pM[A41] - pM[A31]*pM[A40];
  const Double_t Det2_34_02 = pM[A30]*pM[A42] - pM[A32]*pM[A40];
  const Double_t Det2_34_03 = pM[A30]*pM[A43] - pM[A33]*pM[A40];
  const Double_t Det2_34_04 = pM[A30]*pM[A44] - pM[A34]*pM[A40];
  const Double_t Det2_34_05 = pM[A30]*pM[A45] - pM[A35]*pM[A40];
  const Double_t Det2_34_12 = pM[A31]*pM[A42] - pM[A32]*pM[A41];
  const Double_t Det2_34_13 = pM[A31]*pM[A43] - pM[A33]*pM[A41];
  const Double_t Det2_34_14 = pM[A31]*pM[A44] - pM[A34]*pM[A41];
  const Double_t Det2_34_15 = pM[A31]*pM[A45] - pM[A35]*pM[A41];
  const Double_t Det2_34_23 = pM[A32]*pM[A43] - pM[A33]*pM[A42];
  const Double_t Det2_34_24 = pM[A32]*pM[A44] - pM[A34]*pM[A42];
  const Double_t Det2_34_25 = pM[A32]*pM[A45] - pM[A35]*pM[A42];
  const Double_t Det2_34_34 = pM[A33]*pM[A44] - pM[A34]*pM[A43];
  const Double_t Det2_34_35 = pM[A33]*pM[A45] - pM[A35]*pM[A43];
  const Double_t Det2_34_45 = pM[A34]*pM[A45] - pM[A35]*pM[A44];
  const Double_t Det2_35_01 = pM[A30]*pM[A51] - pM[A31]*pM[A50];
  const Double_t Det2_35_02 = pM[A30]*pM[A52] - pM[A32]*pM[A50];
  const Double_t Det2_35_03 = pM[A30]*pM[A53] - pM[A33]*pM[A50];
  const Double_t Det2_35_04 = pM[A30]*pM[A54] - pM[A34]*pM[A50];
  const Double_t Det2_35_05 = pM[A30]*pM[A55] - pM[A35]*pM[A50];
  const Double_t Det2_35_12 = pM[A31]*pM[A52] - pM[A32]*pM[A51];
  const Double_t Det2_35_13 = pM[A31]*pM[A53] - pM[A33]*pM[A51];
  const Double_t Det2_35_14 = pM[A31]*pM[A54] - pM[A34]*pM[A51];
  const Double_t Det2_35_15 = pM[A31]*pM[A55] - pM[A35]*pM[A51];
  const Double_t Det2_35_23 = pM[A32]*pM[A53] - pM[A33]*pM[A52];
  const Double_t Det2_35_24 = pM[A32]*pM[A54] - pM[A34]*pM[A52];
  const Double_t Det2_35_25 = pM[A32]*pM[A55] - pM[A35]*pM[A52];
  const Double_t Det2_35_34 = pM[A33]*pM[A54] - pM[A34]*pM[A53];
  const Double_t Det2_35_35 = pM[A33]*pM[A55] - pM[A35]*pM[A53];
  const Double_t Det2_35_45 = pM[A34]*pM[A55] - pM[A35]*pM[A54];
  const Double_t Det2_45_01 = pM[A40]*pM[A51] - pM[A41]*pM[A50];
  const Double_t Det2_45_02 = pM[A40]*pM[A52] - pM[A42]*pM[A50];
  const Double_t Det2_45_03 = pM[A40]*pM[A53] - pM[A43]*pM[A50];
  const Double_t Det2_45_04 = pM[A40]*pM[A54] - pM[A44]*pM[A50];
  const Double_t Det2_45_05 = pM[A40]*pM[A55] - pM[A45]*pM[A50];
  const Double_t Det2_45_12 = pM[A41]*pM[A52] - pM[A42]*pM[A51];
  const Double_t Det2_45_13 = pM[A41]*pM[A53] - pM[A43]*pM[A51];
  const Double_t Det2_45_14 = pM[A41]*pM[A54] - pM[A44]*pM[A51];
  const Double_t Det2_45_15 = pM[A41]*pM[A55] - pM[A45]*pM[A51];
  const Double_t Det2_45_23 = pM[A42]*pM[A53] - pM[A43]*pM[A52];
  const Double_t Det2_45_24 = pM[A42]*pM[A54] - pM[A44]*pM[A52];
  const Double_t Det2_45_25 = pM[A42]*pM[A55] - pM[A45]*pM[A52];
  const Double_t Det2_45_34 = pM[A43]*pM[A54] - pM[A44]*pM[A53];
  const Double_t Det2_45_35 = pM[A43]*pM[A55] - pM[A45]*pM[A53];
  const Double_t Det2_45_45 = pM[A44]*pM[A55] - pM[A45]*pM[A54];

  // Find all NECESSARY 3x3 dets:  (80 of them)

  const Double_t Det3_234_012 = pM[A20]*Det2_34_12 - pM[A21]*Det2_34_02 + pM[A22]*Det2_34_01;
  const Double_t Det3_234_013 = pM[A20]*Det2_34_13 - pM[A21]*Det2_34_03 + pM[A23]*Det2_34_01;
  const Double_t Det3_234_014 = pM[A20]*Det2_34_14 - pM[A21]*Det2_34_04 + pM[A24]*Det2_34_01;
  const Double_t Det3_234_015 = pM[A20]*Det2_34_15 - pM[A21]*Det2_34_05 + pM[A25]*Det2_34_01;
  const Double_t Det3_234_023 = pM[A20]*Det2_34_23 - pM[A22]*Det2_34_03 + pM[A23]*Det2_34_02;
  const Double_t Det3_234_024 = pM[A20]*Det2_34_24 - pM[A22]*Det2_34_04 + pM[A24]*Det2_34_02;
  const Double_t Det3_234_025 = pM[A20]*Det2_34_25 - pM[A22]*Det2_34_05 + pM[A25]*Det2_34_02;
  const Double_t Det3_234_034 = pM[A20]*Det2_34_34 - pM[A23]*Det2_34_04 + pM[A24]*Det2_34_03;
  const Double_t Det3_234_035 = pM[A20]*Det2_34_35 - pM[A23]*Det2_34_05 + pM[A25]*Det2_34_03;
  const Double_t Det3_234_045 = pM[A20]*Det2_34_45 - pM[A24]*Det2_34_05 + pM[A25]*Det2_34_04;
  const Double_t Det3_234_123 = pM[A21]*Det2_34_23 - pM[A22]*Det2_34_13 + pM[A23]*Det2_34_12;
  const Double_t Det3_234_124 = pM[A21]*Det2_34_24 - pM[A22]*Det2_34_14 + pM[A24]*Det2_34_12;
  const Double_t Det3_234_125 = pM[A21]*Det2_34_25 - pM[A22]*Det2_34_15 + pM[A25]*Det2_34_12;
  const Double_t Det3_234_134 = pM[A21]*Det2_34_34 - pM[A23]*Det2_34_14 + pM[A24]*Det2_34_13;
  const Double_t Det3_234_135 = pM[A21]*Det2_34_35 - pM[A23]*Det2_34_15 + pM[A25]*Det2_34_13;
  const Double_t Det3_234_145 = pM[A21]*Det2_34_45 - pM[A24]*Det2_34_15 + pM[A25]*Det2_34_14;
  const Double_t Det3_234_234 = pM[A22]*Det2_34_34 - pM[A23]*Det2_34_24 + pM[A24]*Det2_34_23;
  const Double_t Det3_234_235 = pM[A22]*Det2_34_35 - pM[A23]*Det2_34_25 + pM[A25]*Det2_34_23;
  const Double_t Det3_234_245 = pM[A22]*Det2_34_45 - pM[A24]*Det2_34_25 + pM[A25]*Det2_34_24;
  const Double_t Det3_234_345 = pM[A23]*Det2_34_45 - pM[A24]*Det2_34_35 + pM[A25]*Det2_34_34;
  const Double_t Det3_235_012 = pM[A20]*Det2_35_12 - pM[A21]*Det2_35_02 + pM[A22]*Det2_35_01;
  const Double_t Det3_235_013 = pM[A20]*Det2_35_13 - pM[A21]*Det2_35_03 + pM[A23]*Det2_35_01;
  const Double_t Det3_235_014 = pM[A20]*Det2_35_14 - pM[A21]*Det2_35_04 + pM[A24]*Det2_35_01;
  const Double_t Det3_235_015 = pM[A20]*Det2_35_15 - pM[A21]*Det2_35_05 + pM[A25]*Det2_35_01;
  const Double_t Det3_235_023 = pM[A20]*Det2_35_23 - pM[A22]*Det2_35_03 + pM[A23]*Det2_35_02;
  const Double_t Det3_235_024 = pM[A20]*Det2_35_24 - pM[A22]*Det2_35_04 + pM[A24]*Det2_35_02;
  const Double_t Det3_235_025 = pM[A20]*Det2_35_25 - pM[A22]*Det2_35_05 + pM[A25]*Det2_35_02;
  const Double_t Det3_235_034 = pM[A20]*Det2_35_34 - pM[A23]*Det2_35_04 + pM[A24]*Det2_35_03;
  const Double_t Det3_235_035 = pM[A20]*Det2_35_35 - pM[A23]*Det2_35_05 + pM[A25]*Det2_35_03;
  const Double_t Det3_235_045 = pM[A20]*Det2_35_45 - pM[A24]*Det2_35_05 + pM[A25]*Det2_35_04;
  const Double_t Det3_235_123 = pM[A21]*Det2_35_23 - pM[A22]*Det2_35_13 + pM[A23]*Det2_35_12;
  const Double_t Det3_235_124 = pM[A21]*Det2_35_24 - pM[A22]*Det2_35_14 + pM[A24]*Det2_35_12;
  const Double_t Det3_235_125 = pM[A21]*Det2_35_25 - pM[A22]*Det2_35_15 + pM[A25]*Det2_35_12;
  const Double_t Det3_235_134 = pM[A21]*Det2_35_34 - pM[A23]*Det2_35_14 + pM[A24]*Det2_35_13;
  const Double_t Det3_235_135 = pM[A21]*Det2_35_35 - pM[A23]*Det2_35_15 + pM[A25]*Det2_35_13;
  const Double_t Det3_235_145 = pM[A21]*Det2_35_45 - pM[A24]*Det2_35_15 + pM[A25]*Det2_35_14;
  const Double_t Det3_235_234 = pM[A22]*Det2_35_34 - pM[A23]*Det2_35_24 + pM[A24]*Det2_35_23;
  const Double_t Det3_235_235 = pM[A22]*Det2_35_35 - pM[A23]*Det2_35_25 + pM[A25]*Det2_35_23;
  const Double_t Det3_235_245 = pM[A22]*Det2_35_45 - pM[A24]*Det2_35_25 + pM[A25]*Det2_35_24;
  const Double_t Det3_235_345 = pM[A23]*Det2_35_45 - pM[A24]*Det2_35_35 + pM[A25]*Det2_35_34;
  const Double_t Det3_245_012 = pM[A20]*Det2_45_12 - pM[A21]*Det2_45_02 + pM[A22]*Det2_45_01;
  const Double_t Det3_245_013 = pM[A20]*Det2_45_13 - pM[A21]*Det2_45_03 + pM[A23]*Det2_45_01;
  const Double_t Det3_245_014 = pM[A20]*Det2_45_14 - pM[A21]*Det2_45_04 + pM[A24]*Det2_45_01;
  const Double_t Det3_245_015 = pM[A20]*Det2_45_15 - pM[A21]*Det2_45_05 + pM[A25]*Det2_45_01;
  const Double_t Det3_245_023 = pM[A20]*Det2_45_23 - pM[A22]*Det2_45_03 + pM[A23]*Det2_45_02;
  const Double_t Det3_245_024 = pM[A20]*Det2_45_24 - pM[A22]*Det2_45_04 + pM[A24]*Det2_45_02;
  const Double_t Det3_245_025 = pM[A20]*Det2_45_25 - pM[A22]*Det2_45_05 + pM[A25]*Det2_45_02;
  const Double_t Det3_245_034 = pM[A20]*Det2_45_34 - pM[A23]*Det2_45_04 + pM[A24]*Det2_45_03;
  const Double_t Det3_245_035 = pM[A20]*Det2_45_35 - pM[A23]*Det2_45_05 + pM[A25]*Det2_45_03;
  const Double_t Det3_245_045 = pM[A20]*Det2_45_45 - pM[A24]*Det2_45_05 + pM[A25]*Det2_45_04;
  const Double_t Det3_245_123 = pM[A21]*Det2_45_23 - pM[A22]*Det2_45_13 + pM[A23]*Det2_45_12;
  const Double_t Det3_245_124 = pM[A21]*Det2_45_24 - pM[A22]*Det2_45_14 + pM[A24]*Det2_45_12;
  const Double_t Det3_245_125 = pM[A21]*Det2_45_25 - pM[A22]*Det2_45_15 + pM[A25]*Det2_45_12;
  const Double_t Det3_245_134 = pM[A21]*Det2_45_34 - pM[A23]*Det2_45_14 + pM[A24]*Det2_45_13;
  const Double_t Det3_245_135 = pM[A21]*Det2_45_35 - pM[A23]*Det2_45_15 + pM[A25]*Det2_45_13;
  const Double_t Det3_245_145 = pM[A21]*Det2_45_45 - pM[A24]*Det2_45_15 + pM[A25]*Det2_45_14;
  const Double_t Det3_245_234 = pM[A22]*Det2_45_34 - pM[A23]*Det2_45_24 + pM[A24]*Det2_45_23;
  const Double_t Det3_245_235 = pM[A22]*Det2_45_35 - pM[A23]*Det2_45_25 + pM[A25]*Det2_45_23;
  const Double_t Det3_245_245 = pM[A22]*Det2_45_45 - pM[A24]*Det2_45_25 + pM[A25]*Det2_45_24;
  const Double_t Det3_245_345 = pM[A23]*Det2_45_45 - pM[A24]*Det2_45_35 + pM[A25]*Det2_45_34;
  const Double_t Det3_345_012 = pM[A30]*Det2_45_12 - pM[A31]*Det2_45_02 + pM[A32]*Det2_45_01;
  const Double_t Det3_345_013 = pM[A30]*Det2_45_13 - pM[A31]*Det2_45_03 + pM[A33]*Det2_45_01;
  const Double_t Det3_345_014 = pM[A30]*Det2_45_14 - pM[A31]*Det2_45_04 + pM[A34]*Det2_45_01;
  const Double_t Det3_345_015 = pM[A30]*Det2_45_15 - pM[A31]*Det2_45_05 + pM[A35]*Det2_45_01;
  const Double_t Det3_345_023 = pM[A30]*Det2_45_23 - pM[A32]*Det2_45_03 + pM[A33]*Det2_45_02;
  const Double_t Det3_345_024 = pM[A30]*Det2_45_24 - pM[A32]*Det2_45_04 + pM[A34]*Det2_45_02;
  const Double_t Det3_345_025 = pM[A30]*Det2_45_25 - pM[A32]*Det2_45_05 + pM[A35]*Det2_45_02;
  const Double_t Det3_345_034 = pM[A30]*Det2_45_34 - pM[A33]*Det2_45_04 + pM[A34]*Det2_45_03;
  const Double_t Det3_345_035 = pM[A30]*Det2_45_35 - pM[A33]*Det2_45_05 + pM[A35]*Det2_45_03;
  const Double_t Det3_345_045 = pM[A30]*Det2_45_45 - pM[A34]*Det2_45_05 + pM[A35]*Det2_45_04;
  const Double_t Det3_345_123 = pM[A31]*Det2_45_23 - pM[A32]*Det2_45_13 + pM[A33]*Det2_45_12;
  const Double_t Det3_345_124 = pM[A31]*Det2_45_24 - pM[A32]*Det2_45_14 + pM[A34]*Det2_45_12;
  const Double_t Det3_345_125 = pM[A31]*Det2_45_25 - pM[A32]*Det2_45_15 + pM[A35]*Det2_45_12;
  const Double_t Det3_345_134 = pM[A31]*Det2_45_34 - pM[A33]*Det2_45_14 + pM[A34]*Det2_45_13;
  const Double_t Det3_345_135 = pM[A31]*Det2_45_35 - pM[A33]*Det2_45_15 + pM[A35]*Det2_45_13;
  const Double_t Det3_345_145 = pM[A31]*Det2_45_45 - pM[A34]*Det2_45_15 + pM[A35]*Det2_45_14;
  const Double_t Det3_345_234 = pM[A32]*Det2_45_34 - pM[A33]*Det2_45_24 + pM[A34]*Det2_45_23;
  const Double_t Det3_345_235 = pM[A32]*Det2_45_35 - pM[A33]*Det2_45_25 + pM[A35]*Det2_45_23;
  const Double_t Det3_345_245 = pM[A32]*Det2_45_45 - pM[A34]*Det2_45_25 + pM[A35]*Det2_45_24;
  const Double_t Det3_345_345 = pM[A33]*Det2_45_45 - pM[A34]*Det2_45_35 + pM[A35]*Det2_45_34;

  // Find all NECESSARY 4x4 dets:  (75 of them)

  const Double_t Det4_1234_0123 = pM[A10]*Det3_234_123 - pM[A11]*Det3_234_023 
                                  + pM[A12]*Det3_234_013 - pM[A13]*Det3_234_012;
  const Double_t Det4_1234_0124 = pM[A10]*Det3_234_124 - pM[A11]*Det3_234_024 
                                  + pM[A12]*Det3_234_014 - pM[A14]*Det3_234_012;
  const Double_t Det4_1234_0125 = pM[A10]*Det3_234_125 - pM[A11]*Det3_234_025 
                                  + pM[A12]*Det3_234_015 - pM[A15]*Det3_234_012;
  const Double_t Det4_1234_0134 = pM[A10]*Det3_234_134 - pM[A11]*Det3_234_034 
                                  + pM[A13]*Det3_234_014 - pM[A14]*Det3_234_013;
  const Double_t Det4_1234_0135 = pM[A10]*Det3_234_135 - pM[A11]*Det3_234_035
                                  + pM[A13]*Det3_234_015 - pM[A15]*Det3_234_013;
  const Double_t Det4_1234_0145 = pM[A10]*Det3_234_145 - pM[A11]*Det3_234_045
                                  + pM[A14]*Det3_234_015 - pM[A15]*Det3_234_014;
  const Double_t Det4_1234_0234 = pM[A10]*Det3_234_234 - pM[A12]*Det3_234_034 
                                  + pM[A13]*Det3_234_024 - pM[A14]*Det3_234_023;
  const Double_t Det4_1234_0235 = pM[A10]*Det3_234_235 - pM[A12]*Det3_234_035 
                                  + pM[A13]*Det3_234_025 - pM[A15]*Det3_234_023;
  const Double_t Det4_1234_0245 = pM[A10]*Det3_234_245 - pM[A12]*Det3_234_045 
                                  + pM[A14]*Det3_234_025 - pM[A15]*Det3_234_024;
  const Double_t Det4_1234_0345 = pM[A10]*Det3_234_345 - pM[A13]*Det3_234_045 
                                  + pM[A14]*Det3_234_035 - pM[A15]*Det3_234_034;
  const Double_t Det4_1234_1234 = pM[A11]*Det3_234_234 - pM[A12]*Det3_234_134 
                                  + pM[A13]*Det3_234_124 - pM[A14]*Det3_234_123;
  const Double_t Det4_1234_1235 = pM[A11]*Det3_234_235 - pM[A12]*Det3_234_135 
                                  + pM[A13]*Det3_234_125 - pM[A15]*Det3_234_123;
  const Double_t Det4_1234_1245 = pM[A11]*Det3_234_245 - pM[A12]*Det3_234_145 
                                  + pM[A14]*Det3_234_125 - pM[A15]*Det3_234_124;
  const Double_t Det4_1234_1345 = pM[A11]*Det3_234_345 - pM[A13]*Det3_234_145 
                                  + pM[A14]*Det3_234_135 - pM[A15]*Det3_234_134;
  const Double_t Det4_1234_2345 = pM[A12]*Det3_234_345 - pM[A13]*Det3_234_245 
                                  + pM[A14]*Det3_234_235 - pM[A15]*Det3_234_234;
  const Double_t Det4_1235_0123 = pM[A10]*Det3_235_123 - pM[A11]*Det3_235_023 
                                  + pM[A12]*Det3_235_013 - pM[A13]*Det3_235_012;
  const Double_t Det4_1235_0124 = pM[A10]*Det3_235_124 - pM[A11]*Det3_235_024 
                                  + pM[A12]*Det3_235_014 - pM[A14]*Det3_235_012;
  const Double_t Det4_1235_0125 = pM[A10]*Det3_235_125 - pM[A11]*Det3_235_025 
                                  + pM[A12]*Det3_235_015 - pM[A15]*Det3_235_012;
  const Double_t Det4_1235_0134 = pM[A10]*Det3_235_134 - pM[A11]*Det3_235_034 
                                  + pM[A13]*Det3_235_014 - pM[A14]*Det3_235_013;
  const Double_t Det4_1235_0135 = pM[A10]*Det3_235_135 - pM[A11]*Det3_235_035 
                                  + pM[A13]*Det3_235_015 - pM[A15]*Det3_235_013;
  const Double_t Det4_1235_0145 = pM[A10]*Det3_235_145 - pM[A11]*Det3_235_045 
                                  + pM[A14]*Det3_235_015 - pM[A15]*Det3_235_014;
  const Double_t Det4_1235_0234 = pM[A10]*Det3_235_234 - pM[A12]*Det3_235_034 
                                  + pM[A13]*Det3_235_024 - pM[A14]*Det3_235_023;
  const Double_t Det4_1235_0235 = pM[A10]*Det3_235_235 - pM[A12]*Det3_235_035 
                                  + pM[A13]*Det3_235_025 - pM[A15]*Det3_235_023;
  const Double_t Det4_1235_0245 = pM[A10]*Det3_235_245 - pM[A12]*Det3_235_045 
                                  + pM[A14]*Det3_235_025 - pM[A15]*Det3_235_024;
  const Double_t Det4_1235_0345 = pM[A10]*Det3_235_345 - pM[A13]*Det3_235_045 
                                  + pM[A14]*Det3_235_035 - pM[A15]*Det3_235_034;
  const Double_t Det4_1235_1234 = pM[A11]*Det3_235_234 - pM[A12]*Det3_235_134 
                                  + pM[A13]*Det3_235_124 - pM[A14]*Det3_235_123;
  const Double_t Det4_1235_1235 = pM[A11]*Det3_235_235 - pM[A12]*Det3_235_135 
                                  + pM[A13]*Det3_235_125 - pM[A15]*Det3_235_123;
  const Double_t Det4_1235_1245 = pM[A11]*Det3_235_245 - pM[A12]*Det3_235_145 
                                  + pM[A14]*Det3_235_125 - pM[A15]*Det3_235_124;
  const Double_t Det4_1235_1345 = pM[A11]*Det3_235_345 - pM[A13]*Det3_235_145 
                                  + pM[A14]*Det3_235_135 - pM[A15]*Det3_235_134;
  const Double_t Det4_1235_2345 = pM[A12]*Det3_235_345 - pM[A13]*Det3_235_245 
                                  + pM[A14]*Det3_235_235 - pM[A15]*Det3_235_234;
  const Double_t Det4_1245_0123 = pM[A10]*Det3_245_123 - pM[A11]*Det3_245_023 
                                  + pM[A12]*Det3_245_013 - pM[A13]*Det3_245_012;
  const Double_t Det4_1245_0124 = pM[A10]*Det3_245_124 - pM[A11]*Det3_245_024 
                                  + pM[A12]*Det3_245_014 - pM[A14]*Det3_245_012;
  const Double_t Det4_1245_0125 = pM[A10]*Det3_245_125 - pM[A11]*Det3_245_025 
                                  + pM[A12]*Det3_245_015 - pM[A15]*Det3_245_012;
  const Double_t Det4_1245_0134 = pM[A10]*Det3_245_134 - pM[A11]*Det3_245_034 
                                  + pM[A13]*Det3_245_014 - pM[A14]*Det3_245_013;
  const Double_t Det4_1245_0135 = pM[A10]*Det3_245_135 - pM[A11]*Det3_245_035 
                                  + pM[A13]*Det3_245_015 - pM[A15]*Det3_245_013;
  const Double_t Det4_1245_0145 = pM[A10]*Det3_245_145 - pM[A11]*Det3_245_045 
                                  + pM[A14]*Det3_245_015 - pM[A15]*Det3_245_014;
  const Double_t Det4_1245_0234 = pM[A10]*Det3_245_234 - pM[A12]*Det3_245_034 
                                  + pM[A13]*Det3_245_024 - pM[A14]*Det3_245_023;
  const Double_t Det4_1245_0235 = pM[A10]*Det3_245_235 - pM[A12]*Det3_245_035 
                                  + pM[A13]*Det3_245_025 - pM[A15]*Det3_245_023;
  const Double_t Det4_1245_0245 = pM[A10]*Det3_245_245 - pM[A12]*Det3_245_045 
                                  + pM[A14]*Det3_245_025 - pM[A15]*Det3_245_024;
  const Double_t Det4_1245_0345 = pM[A10]*Det3_245_345 - pM[A13]*Det3_245_045 
                                  + pM[A14]*Det3_245_035 - pM[A15]*Det3_245_034;
  const Double_t Det4_1245_1234 = pM[A11]*Det3_245_234 - pM[A12]*Det3_245_134 
                                  + pM[A13]*Det3_245_124 - pM[A14]*Det3_245_123;
  const Double_t Det4_1245_1235 = pM[A11]*Det3_245_235 - pM[A12]*Det3_245_135 
                                  + pM[A13]*Det3_245_125 - pM[A15]*Det3_245_123;
  const Double_t Det4_1245_1245 = pM[A11]*Det3_245_245 - pM[A12]*Det3_245_145 
                                  + pM[A14]*Det3_245_125 - pM[A15]*Det3_245_124;
  const Double_t Det4_1245_1345 = pM[A11]*Det3_245_345 - pM[A13]*Det3_245_145 
                                  + pM[A14]*Det3_245_135 - pM[A15]*Det3_245_134;
  const Double_t Det4_1245_2345 = pM[A12]*Det3_245_345 - pM[A13]*Det3_245_245 
                                  + pM[A14]*Det3_245_235 - pM[A15]*Det3_245_234;
  const Double_t Det4_1345_0123 = pM[A10]*Det3_345_123 - pM[A11]*Det3_345_023 
                                  + pM[A12]*Det3_345_013 - pM[A13]*Det3_345_012;
  const Double_t Det4_1345_0124 = pM[A10]*Det3_345_124 - pM[A11]*Det3_345_024 
                                  + pM[A12]*Det3_345_014 - pM[A14]*Det3_345_012;
  const Double_t Det4_1345_0125 = pM[A10]*Det3_345_125 - pM[A11]*Det3_345_025 
                                  + pM[A12]*Det3_345_015 - pM[A15]*Det3_345_012;
  const Double_t Det4_1345_0134 = pM[A10]*Det3_345_134 - pM[A11]*Det3_345_034 
                                  + pM[A13]*Det3_345_014 - pM[A14]*Det3_345_013;
  const Double_t Det4_1345_0135 = pM[A10]*Det3_345_135 - pM[A11]*Det3_345_035 
                                  + pM[A13]*Det3_345_015 - pM[A15]*Det3_345_013;
  const Double_t Det4_1345_0145 = pM[A10]*Det3_345_145 - pM[A11]*Det3_345_045 
                                  + pM[A14]*Det3_345_015 - pM[A15]*Det3_345_014;
  const Double_t Det4_1345_0234 = pM[A10]*Det3_345_234 - pM[A12]*Det3_345_034 
                                  + pM[A13]*Det3_345_024 - pM[A14]*Det3_345_023;
  const Double_t Det4_1345_0235 = pM[A10]*Det3_345_235 - pM[A12]*Det3_345_035 
                                  + pM[A13]*Det3_345_025 - pM[A15]*Det3_345_023;
  const Double_t Det4_1345_0245 = pM[A10]*Det3_345_245 - pM[A12]*Det3_345_045 
                                  + pM[A14]*Det3_345_025 - pM[A15]*Det3_345_024;
  const Double_t Det4_1345_0345 = pM[A10]*Det3_345_345 - pM[A13]*Det3_345_045 
                                  + pM[A14]*Det3_345_035 - pM[A15]*Det3_345_034;
  const Double_t Det4_1345_1234 = pM[A11]*Det3_345_234 - pM[A12]*Det3_345_134 
                                  + pM[A13]*Det3_345_124 - pM[A14]*Det3_345_123;
  const Double_t Det4_1345_1235 = pM[A11]*Det3_345_235 - pM[A12]*Det3_345_135 
                                  + pM[A13]*Det3_345_125 - pM[A15]*Det3_345_123;
  const Double_t Det4_1345_1245 = pM[A11]*Det3_345_245 - pM[A12]*Det3_345_145 
                                  + pM[A14]*Det3_345_125 - pM[A15]*Det3_345_124;
  const Double_t Det4_1345_1345 = pM[A11]*Det3_345_345 - pM[A13]*Det3_345_145 
                                  + pM[A14]*Det3_345_135 - pM[A15]*Det3_345_134;
  const Double_t Det4_1345_2345 = pM[A12]*Det3_345_345 - pM[A13]*Det3_345_245 
                                  + pM[A14]*Det3_345_235 - pM[A15]*Det3_345_234;
  const Double_t Det4_2345_0123 = pM[A20]*Det3_345_123 - pM[A21]*Det3_345_023 
                                  + pM[A22]*Det3_345_013 - pM[A23]*Det3_345_012;
  const Double_t Det4_2345_0124 = pM[A20]*Det3_345_124 - pM[A21]*Det3_345_024 
                                  + pM[A22]*Det3_345_014 - pM[A24]*Det3_345_012;
  const Double_t Det4_2345_0125 = pM[A20]*Det3_345_125 - pM[A21]*Det3_345_025 
                                  + pM[A22]*Det3_345_015 - pM[A25]*Det3_345_012;
  const Double_t Det4_2345_0134 = pM[A20]*Det3_345_134 - pM[A21]*Det3_345_034 
                                  + pM[A23]*Det3_345_014 - pM[A24]*Det3_345_013;
  const Double_t Det4_2345_0135 = pM[A20]*Det3_345_135 - pM[A21]*Det3_345_035 
                                  + pM[A23]*Det3_345_015 - pM[A25]*Det3_345_013;
  const Double_t Det4_2345_0145 = pM[A20]*Det3_345_145 - pM[A21]*Det3_345_045 
                                  + pM[A24]*Det3_345_015 - pM[A25]*Det3_345_014;
  const Double_t Det4_2345_0234 = pM[A20]*Det3_345_234 - pM[A22]*Det3_345_034 
                                  + pM[A23]*Det3_345_024 - pM[A24]*Det3_345_023;
  const Double_t Det4_2345_0235 = pM[A20]*Det3_345_235 - pM[A22]*Det3_345_035 
                                  + pM[A23]*Det3_345_025 - pM[A25]*Det3_345_023;
  const Double_t Det4_2345_0245 = pM[A20]*Det3_345_245 - pM[A22]*Det3_345_045 
                                  + pM[A24]*Det3_345_025 - pM[A25]*Det3_345_024;
  const Double_t Det4_2345_0345 = pM[A20]*Det3_345_345 - pM[A23]*Det3_345_045 
                                  + pM[A24]*Det3_345_035 - pM[A25]*Det3_345_034;
  const Double_t Det4_2345_1234 = pM[A21]*Det3_345_234 - pM[A22]*Det3_345_134 
                                  + pM[A23]*Det3_345_124 - pM[A24]*Det3_345_123;
  const Double_t Det4_2345_1235 = pM[A21]*Det3_345_235 - pM[A22]*Det3_345_135 
                                  + pM[A23]*Det3_345_125 - pM[A25]*Det3_345_123;
  const Double_t Det4_2345_1245 = pM[A21]*Det3_345_245 - pM[A22]*Det3_345_145 
                                  + pM[A24]*Det3_345_125 - pM[A25]*Det3_345_124;
  const Double_t Det4_2345_1345 = pM[A21]*Det3_345_345 - pM[A23]*Det3_345_145 
                                  + pM[A24]*Det3_345_135 - pM[A25]*Det3_345_134;
  const Double_t Det4_2345_2345 = pM[A22]*Det3_345_345 - pM[A23]*Det3_345_245 
                                  + pM[A24]*Det3_345_235 - pM[A25]*Det3_345_234;

  // Find all NECESSARY 5x5 dets:  (36 of them)

  const Double_t Det5_01234_01234 = pM[A00]*Det4_1234_1234 - pM[A01]*Det4_1234_0234 
                                    + pM[A02]*Det4_1234_0134 - pM[A03]*Det4_1234_0124 + pM[A04]*Det4_1234_0123;
  const Double_t Det5_01234_01235 = pM[A00]*Det4_1234_1235 - pM[A01]*Det4_1234_0235
                                    + pM[A02]*Det4_1234_0135 - pM[A03]*Det4_1234_0125 + pM[A05]*Det4_1234_0123;
  const Double_t Det5_01234_01245 = pM[A00]*Det4_1234_1245 - pM[A01]*Det4_1234_0245
                                    + pM[A02]*Det4_1234_0145 - pM[A04]*Det4_1234_0125 + pM[A05]*Det4_1234_0124;
  const Double_t Det5_01234_01345 = pM[A00]*Det4_1234_1345 - pM[A01]*Det4_1234_0345
                                    + pM[A03]*Det4_1234_0145 - pM[A04]*Det4_1234_0135 + pM[A05]*Det4_1234_0134;
  const Double_t Det5_01234_02345 = pM[A00]*Det4_1234_2345 - pM[A02]*Det4_1234_0345
                                    + pM[A03]*Det4_1234_0245 - pM[A04]*Det4_1234_0235 + pM[A05]*Det4_1234_0234;
  const Double_t Det5_01234_12345 = pM[A01]*Det4_1234_2345 - pM[A02]*Det4_1234_1345
                                    + pM[A03]*Det4_1234_1245 - pM[A04]*Det4_1234_1235 + pM[A05]*Det4_1234_1234;
  const Double_t Det5_01235_01234 = pM[A00]*Det4_1235_1234 - pM[A01]*Det4_1235_0234 
                                    + pM[A02]*Det4_1235_0134 - pM[A03]*Det4_1235_0124 + pM[A04]*Det4_1235_0123;
  const Double_t Det5_01235_01235 = pM[A00]*Det4_1235_1235 - pM[A01]*Det4_1235_0235 
                                    + pM[A02]*Det4_1235_0135 - pM[A03]*Det4_1235_0125 + pM[A05]*Det4_1235_0123;
  const Double_t Det5_01235_01245 = pM[A00]*Det4_1235_1245 - pM[A01]*Det4_1235_0245 
                                    + pM[A02]*Det4_1235_0145 - pM[A04]*Det4_1235_0125 + pM[A05]*Det4_1235_0124;
  const Double_t Det5_01235_01345 = pM[A00]*Det4_1235_1345 - pM[A01]*Det4_1235_0345 
                                    + pM[A03]*Det4_1235_0145 - pM[A04]*Det4_1235_0135 + pM[A05]*Det4_1235_0134;
  const Double_t Det5_01235_02345 = pM[A00]*Det4_1235_2345 - pM[A02]*Det4_1235_0345 
                                    + pM[A03]*Det4_1235_0245 - pM[A04]*Det4_1235_0235 + pM[A05]*Det4_1235_0234;
  const Double_t Det5_01235_12345 = pM[A01]*Det4_1235_2345 - pM[A02]*Det4_1235_1345 
                                    + pM[A03]*Det4_1235_1245 - pM[A04]*Det4_1235_1235 + pM[A05]*Det4_1235_1234;
  const Double_t Det5_01245_01234 = pM[A00]*Det4_1245_1234 - pM[A01]*Det4_1245_0234 
                                    + pM[A02]*Det4_1245_0134 - pM[A03]*Det4_1245_0124 + pM[A04]*Det4_1245_0123;
  const Double_t Det5_01245_01235 = pM[A00]*Det4_1245_1235 - pM[A01]*Det4_1245_0235 
                                    + pM[A02]*Det4_1245_0135 - pM[A03]*Det4_1245_0125 + pM[A05]*Det4_1245_0123;
  const Double_t Det5_01245_01245 = pM[A00]*Det4_1245_1245 - pM[A01]*Det4_1245_0245 
                                    + pM[A02]*Det4_1245_0145 - pM[A04]*Det4_1245_0125 + pM[A05]*Det4_1245_0124;
  const Double_t Det5_01245_01345 = pM[A00]*Det4_1245_1345 - pM[A01]*Det4_1245_0345 
                                    + pM[A03]*Det4_1245_0145 - pM[A04]*Det4_1245_0135 + pM[A05]*Det4_1245_0134;
  const Double_t Det5_01245_02345 = pM[A00]*Det4_1245_2345 - pM[A02]*Det4_1245_0345 
                                    + pM[A03]*Det4_1245_0245 - pM[A04]*Det4_1245_0235 + pM[A05]*Det4_1245_0234;
  const Double_t Det5_01245_12345 = pM[A01]*Det4_1245_2345 - pM[A02]*Det4_1245_1345 
                                    + pM[A03]*Det4_1245_1245 - pM[A04]*Det4_1245_1235 + pM[A05]*Det4_1245_1234;
  const Double_t Det5_01345_01234 = pM[A00]*Det4_1345_1234 - pM[A01]*Det4_1345_0234 
                                    + pM[A02]*Det4_1345_0134 - pM[A03]*Det4_1345_0124 + pM[A04]*Det4_1345_0123;
  const Double_t Det5_01345_01235 = pM[A00]*Det4_1345_1235 - pM[A01]*Det4_1345_0235 
                                    + pM[A02]*Det4_1345_0135 - pM[A03]*Det4_1345_0125 + pM[A05]*Det4_1345_0123;
  const Double_t Det5_01345_01245 = pM[A00]*Det4_1345_1245 - pM[A01]*Det4_1345_0245 
                                    + pM[A02]*Det4_1345_0145 - pM[A04]*Det4_1345_0125 + pM[A05]*Det4_1345_0124;
  const Double_t Det5_01345_01345 = pM[A00]*Det4_1345_1345 - pM[A01]*Det4_1345_0345 
                                    + pM[A03]*Det4_1345_0145 - pM[A04]*Det4_1345_0135 + pM[A05]*Det4_1345_0134;
  const Double_t Det5_01345_02345 = pM[A00]*Det4_1345_2345 - pM[A02]*Det4_1345_0345 
                                    + pM[A03]*Det4_1345_0245 - pM[A04]*Det4_1345_0235 + pM[A05]*Det4_1345_0234;
  const Double_t Det5_01345_12345 = pM[A01]*Det4_1345_2345 - pM[A02]*Det4_1345_1345 
                                    + pM[A03]*Det4_1345_1245 - pM[A04]*Det4_1345_1235 + pM[A05]*Det4_1345_1234;
  const Double_t Det5_02345_01234 = pM[A00]*Det4_2345_1234 - pM[A01]*Det4_2345_0234 
                                    + pM[A02]*Det4_2345_0134 - pM[A03]*Det4_2345_0124 + pM[A04]*Det4_2345_0123;
  const Double_t Det5_02345_01235 = pM[A00]*Det4_2345_1235 - pM[A01]*Det4_2345_0235 
                                    + pM[A02]*Det4_2345_0135 - pM[A03]*Det4_2345_0125 + pM[A05]*Det4_2345_0123;
  const Double_t Det5_02345_01245 = pM[A00]*Det4_2345_1245 - pM[A01]*Det4_2345_0245 
                                    + pM[A02]*Det4_2345_0145 - pM[A04]*Det4_2345_0125 + pM[A05]*Det4_2345_0124;
  const Double_t Det5_02345_01345 = pM[A00]*Det4_2345_1345 - pM[A01]*Det4_2345_0345 
                                    + pM[A03]*Det4_2345_0145 - pM[A04]*Det4_2345_0135 + pM[A05]*Det4_2345_0134;
  const Double_t Det5_02345_02345 = pM[A00]*Det4_2345_2345 - pM[A02]*Det4_2345_0345 
                                    + pM[A03]*Det4_2345_0245 - pM[A04]*Det4_2345_0235 + pM[A05]*Det4_2345_0234;
  const Double_t Det5_02345_12345 = pM[A01]*Det4_2345_2345 - pM[A02]*Det4_2345_1345 
                                    + pM[A03]*Det4_2345_1245 - pM[A04]*Det4_2345_1235 + pM[A05]*Det4_2345_1234;
  const Double_t Det5_12345_01234 = pM[A10]*Det4_2345_1234 - pM[A11]*Det4_2345_0234 
                                    + pM[A12]*Det4_2345_0134 - pM[A13]*Det4_2345_0124 + pM[A14]*Det4_2345_0123;
  const Double_t Det5_12345_01235 = pM[A10]*Det4_2345_1235 - pM[A11]*Det4_2345_0235 
                                    + pM[A12]*Det4_2345_0135 - pM[A13]*Det4_2345_0125 + pM[A15]*Det4_2345_0123;
  const Double_t Det5_12345_01245 = pM[A10]*Det4_2345_1245 - pM[A11]*Det4_2345_0245 
                                    + pM[A12]*Det4_2345_0145 - pM[A14]*Det4_2345_0125 + pM[A15]*Det4_2345_0124;
  const Double_t Det5_12345_01345 = pM[A10]*Det4_2345_1345 - pM[A11]*Det4_2345_0345 
                                    + pM[A13]*Det4_2345_0145 - pM[A14]*Det4_2345_0135 + pM[A15]*Det4_2345_0134;
  const Double_t Det5_12345_02345 = pM[A10]*Det4_2345_2345 - pM[A12]*Det4_2345_0345 
                                    + pM[A13]*Det4_2345_0245 - pM[A14]*Det4_2345_0235 + pM[A15]*Det4_2345_0234;
  const Double_t Det5_12345_12345 = pM[A11]*Det4_2345_2345 - pM[A12]*Det4_2345_1345 
                                    + pM[A13]*Det4_2345_1245 - pM[A14]*Det4_2345_1235 + pM[A15]*Det4_2345_1234;

  // Find the determinant 

  const Double_t det = pM[A00]*Det5_12345_12345 - pM[A01]*Det5_12345_02345 + pM[A02]*Det5_12345_01345 
                       - pM[A03]*Det5_12345_01245 + pM[A04]*Det5_12345_01235 - pM[A05]*Det5_12345_01234;
  if (determ)
    *determ = det;

  if ( det == 0 )
    m.Invalidate();

  const Double_t oneOverDet = 1.0/det;
  const Double_t mn1OverDet = - oneOverDet;

  pM[A00] =  Det5_12345_12345*oneOverDet;
  pM[A01] =  Det5_02345_12345*mn1OverDet;
  pM[A02] =  Det5_01345_12345*oneOverDet;
  pM[A03] =  Det5_01245_12345*mn1OverDet;
  pM[A04] =  Det5_01235_12345*oneOverDet;
  pM[A05] =  Det5_01234_12345*mn1OverDet;

  pM[A10] =  Det5_12345_02345*mn1OverDet;
  pM[A11] =  Det5_02345_02345*oneOverDet;
  pM[A12] =  Det5_01345_02345*mn1OverDet;
  pM[A13] =  Det5_01245_02345*oneOverDet;
  pM[A14] =  Det5_01235_02345*mn1OverDet;
  pM[A15] =  Det5_01234_02345*oneOverDet;

  pM[A20] =  Det5_12345_01345*oneOverDet;
  pM[A21] =  Det5_02345_01345*mn1OverDet;
  pM[A22] =  Det5_01345_01345*oneOverDet;
  pM[A23] =  Det5_01245_01345*mn1OverDet;
  pM[A24] =  Det5_01235_01345*oneOverDet;
  pM[A25] =  Det5_01234_01345*mn1OverDet;

  pM[A30] =  Det5_12345_01245*mn1OverDet;
  pM[A31] =  Det5_02345_01245*oneOverDet;
  pM[A32] =  Det5_01345_01245*mn1OverDet;
  pM[A33] =  Det5_01245_01245*oneOverDet;
  pM[A34] =  Det5_01235_01245*mn1OverDet;
  pM[A35] =  Det5_01234_01245*oneOverDet;

  pM[A40] =  Det5_12345_01235*oneOverDet;
  pM[A41] =  Det5_02345_01235*mn1OverDet;
  pM[A42] =  Det5_01345_01235*oneOverDet;
  pM[A43] =  Det5_01245_01235*mn1OverDet;
  pM[A44] =  Det5_01235_01235*oneOverDet;
  pM[A45] =  Det5_01234_01235*mn1OverDet;

  pM[A50] =  Det5_12345_01234*mn1OverDet;
  pM[A51] =  Det5_02345_01234*oneOverDet;
  pM[A52] =  Det5_01345_01234*mn1OverDet;
  pM[A53] =  Det5_01245_01234*oneOverDet;
  pM[A54] =  Det5_01235_01234*mn1OverDet;
  pM[A55] =  Det5_01234_01234*oneOverDet;

  return kTRUE;
}
