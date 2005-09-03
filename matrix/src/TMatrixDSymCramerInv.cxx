// @(#)root/base:$Name:  $:$Id: TMatrixDSymCramerInv.cxx,v 1.2 2005/09/02 11:04:45 brun Exp $
// Authors: Fons Rademakers, Eddy Offermann  Oct 2004

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMatrixDSymCramerInv                                                 //
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

#include "TMatrixDSymCramerInv.h"

//______________________________________________________________________________
Bool_t TMatrixDSymCramerInv::Inv2x2(TMatrixDSym &m,Double_t *determ)
{
  if (m.GetNrows() != 2) {
    Error("Inv2x2","matrix should be square 2x2");
    m.Invalidate();
    return kFALSE;
  }

  Double_t *pM = m.GetMatrixArray();

  const Double_t det = pM[0] * pM[3] - pM[1] * pM[1];

  if (determ)
    *determ = det;

  if ( det == 0 )
    return kFALSE;

  const Double_t tmp1 =  pM[3] / det;
  pM[3] = pM[0] / det;
  pM[2] = pM[1] = -pM[1] / det;
  pM[0] = tmp1;

  return kTRUE;
}

//______________________________________________________________________________
Bool_t TMatrixDSymCramerInv::Inv3x3(TMatrixDSym &m,Double_t *determ)
{
  if (m.GetNrows() != 3) {
    Error("Inv3x3","matrix should be square 3x3");
    m.Invalidate();
    return kFALSE;
  }

  Double_t *pM = m.GetMatrixArray();

  const Double_t c00 = pM[4] * pM[8] - pM[5] * pM[5];
  const Double_t c01 = pM[5] * pM[2] - pM[1] * pM[8];
  const Double_t c02 = pM[1] * pM[5] - pM[4] * pM[2];
  const Double_t c11 = pM[8] * pM[0] - pM[2] * pM[2];
  const Double_t c12 = pM[2] * pM[1] - pM[5] * pM[0];
  const Double_t c22 = pM[0] * pM[4] - pM[1] * pM[1];

  const Double_t t0  = TMath::Abs(pM[0]);
  const Double_t t1  = TMath::Abs(pM[1]);
  const Double_t t2  = TMath::Abs(pM[2]);

  Double_t det;
  Double_t tmp;

  if (t0 >= t1) {
    if (t2 >= t0) {
      tmp = pM[2];
      det = c12*c01-c11*c02;
    } else {
      tmp = pM[0];
      det = c11*c22-c12*c12;
    }
  } else if (t2 >= t1) {
    tmp = pM[2];
    det = c12*c01-c11*c02;
  } else {
    tmp = pM[1];
    det = c02*c12-c01*c22;
  }

  if ( det == 0 || tmp == 0)
    return kFALSE;

  Double_t s = tmp/det;
  if (determ)
    *determ = 1./s;

  pM[0] = s*c00;
  pM[1] = s*c01;
  pM[2] = s*c02;
  pM[3] = pM[1];
  pM[4] = s*c11;
  pM[5] = s*c12;
  pM[6] = pM[2];
  pM[7] = pM[5];
  pM[8] = s*c22;
  
  return kTRUE;
}

// Fij are indices for a 4x4 symmetric matrix.

#define F00 0
#define F01 1
#define F02 2
#define F03 3

#define F10 1
#define F11 5
#define F12 6
#define F13 7

#define F20 2
#define F21 6
#define F22 10
#define F23 11

#define F30 3
#define F31 7
#define F32 11
#define F33 15

//______________________________________________________________________________
Bool_t TMatrixDSymCramerInv::Inv4x4(TMatrixDSym &m,Double_t *determ)
{
  if (m.GetNrows() != 4) {
    Error("Inv4x4","matrix should be square 4x4");
    m.Invalidate();
    return kFALSE;
  }

  Double_t *pM = m.GetMatrixArray();

  // Find all NECESSARY 2x2 dets:  (14 of them)

  const Double_t mDet2_12_01 = pM[F10]*pM[F21] - pM[F11]*pM[F20];
  const Double_t mDet2_12_02 = pM[F10]*pM[F22] - pM[F12]*pM[F20];
  const Double_t mDet2_12_12 = pM[F11]*pM[F22] - pM[F12]*pM[F21];
  const Double_t mDet2_13_01 = pM[F10]*pM[F31] - pM[F11]*pM[F30];
  const Double_t mDet2_13_02 = pM[F10]*pM[F32] - pM[F12]*pM[F30];
  const Double_t mDet2_13_03 = pM[F10]*pM[F33] - pM[F13]*pM[F30];
  const Double_t mDet2_13_12 = pM[F11]*pM[F32] - pM[F12]*pM[F31];
  const Double_t mDet2_13_13 = pM[F11]*pM[F33] - pM[F13]*pM[F31];
  const Double_t mDet2_23_01 = pM[F20]*pM[F31] - pM[F21]*pM[F30];
  const Double_t mDet2_23_02 = pM[F20]*pM[F32] - pM[F22]*pM[F30];
  const Double_t mDet2_23_03 = pM[F20]*pM[F33] - pM[F23]*pM[F30];
  const Double_t mDet2_23_12 = pM[F21]*pM[F32] - pM[F22]*pM[F31];
  const Double_t mDet2_23_13 = pM[F21]*pM[F33] - pM[F23]*pM[F31];
  const Double_t mDet2_23_23 = pM[F22]*pM[F33] - pM[F23]*pM[F32];

  // Find all NECESSFRY 3x3 dets:   (10 of them)
  
  const Double_t mDet3_012_012 = pM[F00]*mDet2_12_12 - pM[F01]*mDet2_12_02
                               + pM[F02]*mDet2_12_01;
  const Double_t mDet3_013_012 = pM[F00]*mDet2_13_12 - pM[F01]*mDet2_13_02
                               + pM[F02]*mDet2_13_01;
  const Double_t mDet3_013_013 = pM[F00]*mDet2_13_13 - pM[F01]*mDet2_13_03
                               + pM[F03]*mDet2_13_01;
  const Double_t mDet3_023_012 = pM[F00]*mDet2_23_12 - pM[F01]*mDet2_23_02
                               + pM[F02]*mDet2_23_01;
  const Double_t mDet3_023_013 = pM[F00]*mDet2_23_13 - pM[F01]*mDet2_23_03
                               + pM[F03]*mDet2_23_01;
  const Double_t mDet3_023_023 = pM[F00]*mDet2_23_23 - pM[F02]*mDet2_23_03
                               + pM[F03]*mDet2_23_02;
  const Double_t mDet3_123_012 = pM[F10]*mDet2_23_12 - pM[F11]*mDet2_23_02
                               + pM[F12]*mDet2_23_01;
  const Double_t mDet3_123_013 = pM[F10]*mDet2_23_13 - pM[F11]*mDet2_23_03
                               + pM[F13]*mDet2_23_01;
  const Double_t mDet3_123_023 = pM[F10]*mDet2_23_23 - pM[F12]*mDet2_23_03
                               + pM[F13]*mDet2_23_02;
  const Double_t mDet3_123_123 = pM[F11]*mDet2_23_23 - pM[F12]*mDet2_23_13
                               + pM[F13]*mDet2_23_12;

  // Find the 4x4 det:

  const Double_t det = pM[F00]*mDet3_123_123 - pM[F01]*mDet3_123_023
                       + pM[F02]*mDet3_123_013 - pM[F03]*mDet3_123_012;

  if (determ)
    *determ = det;

  if ( det == 0 )
    return kFALSE;

  const Double_t oneOverDet = 1.0/det;
  const Double_t mn1OverDet = - oneOverDet;

  pM[F00] =  mDet3_123_123 * oneOverDet;
  pM[F01] =  mDet3_123_023 * mn1OverDet;
  pM[F02] =  mDet3_123_013 * oneOverDet;
  pM[F03] =  mDet3_123_012 * mn1OverDet;

  pM[F11] =  mDet3_023_023 * oneOverDet;
  pM[F12] =  mDet3_023_013 * mn1OverDet;
  pM[F13] =  mDet3_023_012 * oneOverDet;

  pM[F22] =  mDet3_013_013 * oneOverDet;
  pM[F23] =  mDet3_013_012 * mn1OverDet;

  pM[F33] =  mDet3_012_012 * oneOverDet;

  for (Int_t irow = 0; irow < 4; irow++) {
    const Int_t rowOff1 = irow*4; 
    for (Int_t icol = 0; icol < irow; icol++) {
      const Int_t rowOff2 = icol*4; 
      pM[rowOff1+icol] = pM[rowOff2+irow];
    }
  } 

  return kTRUE;
}

// Mij are indices for a 5x5 matrix.
    
#define M00 0
#define M01 1
#define M02 2
#define M03 3
#define M04 4
  
#define M10 1
#define M11 6
#define M12 7
#define M13 8
#define M14 9

#define M20 2
#define M21 7
#define M22 12
#define M23 13
#define M24 14

#define M30 3
#define M31 8
#define M32 13
#define M33 18
#define M34 19

#define M40 4
#define M41 9
#define M42 14
#define M43 19
#define M44 24

//______________________________________________________________________________
Bool_t TMatrixDSymCramerInv::Inv5x5(TMatrixDSym &m,Double_t *determ)
{
  if (m.GetNrows() != 5) {
    Error("Inv5x5","matrix should be square 5x5");
    m.Invalidate();
    return kFALSE;
  }

  Double_t *pM = m.GetMatrixArray();

  // Find all NECESSARY 2x2 dets:  (25 of them)

  const Double_t mDet2_23_01 = pM[M20]*pM[M31] - pM[M21]*pM[M30];
  const Double_t mDet2_23_02 = pM[M20]*pM[M32] - pM[M22]*pM[M30];
  const Double_t mDet2_23_03 = pM[M20]*pM[M33] - pM[M23]*pM[M30];
  const Double_t mDet2_23_12 = pM[M21]*pM[M32] - pM[M22]*pM[M31];
  const Double_t mDet2_23_13 = pM[M21]*pM[M33] - pM[M23]*pM[M31];
  const Double_t mDet2_23_23 = pM[M22]*pM[M33] - pM[M23]*pM[M32];
  const Double_t mDet2_24_01 = pM[M20]*pM[M41] - pM[M21]*pM[M40];
  const Double_t mDet2_24_02 = pM[M20]*pM[M42] - pM[M22]*pM[M40];
  const Double_t mDet2_24_03 = pM[M20]*pM[M43] - pM[M23]*pM[M40];
  const Double_t mDet2_24_04 = pM[M20]*pM[M44] - pM[M24]*pM[M40];
  const Double_t mDet2_24_12 = pM[M21]*pM[M42] - pM[M22]*pM[M41];
  const Double_t mDet2_24_13 = pM[M21]*pM[M43] - pM[M23]*pM[M41];
  const Double_t mDet2_24_14 = pM[M21]*pM[M44] - pM[M24]*pM[M41];
  const Double_t mDet2_24_23 = pM[M22]*pM[M43] - pM[M23]*pM[M42];
  const Double_t mDet2_24_24 = pM[M22]*pM[M44] - pM[M24]*pM[M42];
  const Double_t mDet2_34_01 = pM[M30]*pM[M41] - pM[M31]*pM[M40];
  const Double_t mDet2_34_02 = pM[M30]*pM[M42] - pM[M32]*pM[M40];
  const Double_t mDet2_34_03 = pM[M30]*pM[M43] - pM[M33]*pM[M40];
  const Double_t mDet2_34_04 = pM[M30]*pM[M44] - pM[M34]*pM[M40];
  const Double_t mDet2_34_12 = pM[M31]*pM[M42] - pM[M32]*pM[M41];
  const Double_t mDet2_34_13 = pM[M31]*pM[M43] - pM[M33]*pM[M41];
  const Double_t mDet2_34_14 = pM[M31]*pM[M44] - pM[M34]*pM[M41];
  const Double_t mDet2_34_23 = pM[M32]*pM[M43] - pM[M33]*pM[M42];
  const Double_t mDet2_34_24 = pM[M32]*pM[M44] - pM[M34]*pM[M42];
  const Double_t mDet2_34_34 = pM[M33]*pM[M44] - pM[M34]*pM[M43];

  // Find all NECESSMRY 3x3 dets:   (30 of them)

  const Double_t mDet3_123_012 = pM[M10]*mDet2_23_12 - pM[M11]*mDet2_23_02 + pM[M12]*mDet2_23_01;
  const Double_t mDet3_123_013 = pM[M10]*mDet2_23_13 - pM[M11]*mDet2_23_03 + pM[M13]*mDet2_23_01;
  const Double_t mDet3_123_023 = pM[M10]*mDet2_23_23 - pM[M12]*mDet2_23_03 + pM[M13]*mDet2_23_02;
  const Double_t mDet3_123_123 = pM[M11]*mDet2_23_23 - pM[M12]*mDet2_23_13 + pM[M13]*mDet2_23_12;
  const Double_t mDet3_124_012 = pM[M10]*mDet2_24_12 - pM[M11]*mDet2_24_02 + pM[M12]*mDet2_24_01;
  const Double_t mDet3_124_013 = pM[M10]*mDet2_24_13 - pM[M11]*mDet2_24_03 + pM[M13]*mDet2_24_01;
  const Double_t mDet3_124_014 = pM[M10]*mDet2_24_14 - pM[M11]*mDet2_24_04 + pM[M14]*mDet2_24_01;
  const Double_t mDet3_124_023 = pM[M10]*mDet2_24_23 - pM[M12]*mDet2_24_03 + pM[M13]*mDet2_24_02;
  const Double_t mDet3_124_024 = pM[M10]*mDet2_24_24 - pM[M12]*mDet2_24_04 + pM[M14]*mDet2_24_02;
  const Double_t mDet3_124_123 = pM[M11]*mDet2_24_23 - pM[M12]*mDet2_24_13 + pM[M13]*mDet2_24_12;
  const Double_t mDet3_124_124 = pM[M11]*mDet2_24_24 - pM[M12]*mDet2_24_14 + pM[M14]*mDet2_24_12;
  const Double_t mDet3_134_012 = pM[M10]*mDet2_34_12 - pM[M11]*mDet2_34_02 + pM[M12]*mDet2_34_01;
  const Double_t mDet3_134_013 = pM[M10]*mDet2_34_13 - pM[M11]*mDet2_34_03 + pM[M13]*mDet2_34_01;
  const Double_t mDet3_134_014 = pM[M10]*mDet2_34_14 - pM[M11]*mDet2_34_04 + pM[M14]*mDet2_34_01;
  const Double_t mDet3_134_023 = pM[M10]*mDet2_34_23 - pM[M12]*mDet2_34_03 + pM[M13]*mDet2_34_02;
  const Double_t mDet3_134_024 = pM[M10]*mDet2_34_24 - pM[M12]*mDet2_34_04 + pM[M14]*mDet2_34_02;
  const Double_t mDet3_134_034 = pM[M10]*mDet2_34_34 - pM[M13]*mDet2_34_04 + pM[M14]*mDet2_34_03;
  const Double_t mDet3_134_123 = pM[M11]*mDet2_34_23 - pM[M12]*mDet2_34_13 + pM[M13]*mDet2_34_12;
  const Double_t mDet3_134_124 = pM[M11]*mDet2_34_24 - pM[M12]*mDet2_34_14 + pM[M14]*mDet2_34_12;
  const Double_t mDet3_134_134 = pM[M11]*mDet2_34_34 - pM[M13]*mDet2_34_14 + pM[M14]*mDet2_34_13;
  const Double_t mDet3_234_012 = pM[M20]*mDet2_34_12 - pM[M21]*mDet2_34_02 + pM[M22]*mDet2_34_01;
  const Double_t mDet3_234_013 = pM[M20]*mDet2_34_13 - pM[M21]*mDet2_34_03 + pM[M23]*mDet2_34_01;
  const Double_t mDet3_234_014 = pM[M20]*mDet2_34_14 - pM[M21]*mDet2_34_04 + pM[M24]*mDet2_34_01;
  const Double_t mDet3_234_023 = pM[M20]*mDet2_34_23 - pM[M22]*mDet2_34_03 + pM[M23]*mDet2_34_02;
  const Double_t mDet3_234_024 = pM[M20]*mDet2_34_24 - pM[M22]*mDet2_34_04 + pM[M24]*mDet2_34_02;
  const Double_t mDet3_234_034 = pM[M20]*mDet2_34_34 - pM[M23]*mDet2_34_04 + pM[M24]*mDet2_34_03;
  const Double_t mDet3_234_123 = pM[M21]*mDet2_34_23 - pM[M22]*mDet2_34_13 + pM[M23]*mDet2_34_12;
  const Double_t mDet3_234_124 = pM[M21]*mDet2_34_24 - pM[M22]*mDet2_34_14 + pM[M24]*mDet2_34_12;
  const Double_t mDet3_234_134 = pM[M21]*mDet2_34_34 - pM[M23]*mDet2_34_14 + pM[M24]*mDet2_34_13;
  const Double_t mDet3_234_234 = pM[M22]*mDet2_34_34 - pM[M23]*mDet2_34_24 + pM[M24]*mDet2_34_23;

  // Find all NECESSMRY 4x4 dets:   (15 of them)

  const Double_t mDet4_0123_0123 = pM[M00]*mDet3_123_123 - pM[M01]*mDet3_123_023
                                 + pM[M02]*mDet3_123_013 - pM[M03]*mDet3_123_012;
  const Double_t mDet4_0124_0123 = pM[M00]*mDet3_124_123 - pM[M01]*mDet3_124_023
                                 + pM[M02]*mDet3_124_013 - pM[M03]*mDet3_124_012;
  const Double_t mDet4_0124_0124 = pM[M00]*mDet3_124_124 - pM[M01]*mDet3_124_024
                                 + pM[M02]*mDet3_124_014 - pM[M04]*mDet3_124_012;
  const Double_t mDet4_0134_0123 = pM[M00]*mDet3_134_123 - pM[M01]*mDet3_134_023
                                 + pM[M02]*mDet3_134_013 - pM[M03]*mDet3_134_012;
  const Double_t mDet4_0134_0124 = pM[M00]*mDet3_134_124 - pM[M01]*mDet3_134_024
                                 + pM[M02]*mDet3_134_014 - pM[M04]*mDet3_134_012;
  const Double_t mDet4_0134_0134 = pM[M00]*mDet3_134_134 - pM[M01]*mDet3_134_034
                                 + pM[M03]*mDet3_134_014 - pM[M04]*mDet3_134_013;
  const Double_t mDet4_0234_0123 = pM[M00]*mDet3_234_123 - pM[M01]*mDet3_234_023
                                 + pM[M02]*mDet3_234_013 - pM[M03]*mDet3_234_012;
  const Double_t mDet4_0234_0124 = pM[M00]*mDet3_234_124 - pM[M01]*mDet3_234_024
                                 + pM[M02]*mDet3_234_014 - pM[M04]*mDet3_234_012;
  const Double_t mDet4_0234_0134 = pM[M00]*mDet3_234_134 - pM[M01]*mDet3_234_034
                                 + pM[M03]*mDet3_234_014 - pM[M04]*mDet3_234_013;
  const Double_t mDet4_0234_0234 = pM[M00]*mDet3_234_234 - pM[M02]*mDet3_234_034
                                 + pM[M03]*mDet3_234_024 - pM[M04]*mDet3_234_023;
  const Double_t mDet4_1234_0123 = pM[M10]*mDet3_234_123 - pM[M11]*mDet3_234_023
                                 + pM[M12]*mDet3_234_013 - pM[M13]*mDet3_234_012;
  const Double_t mDet4_1234_0124 = pM[M10]*mDet3_234_124 - pM[M11]*mDet3_234_024
                                 + pM[M12]*mDet3_234_014 - pM[M14]*mDet3_234_012;
  const Double_t mDet4_1234_0134 = pM[M10]*mDet3_234_134 - pM[M11]*mDet3_234_034
                                 + pM[M13]*mDet3_234_014 - pM[M14]*mDet3_234_013;
  const Double_t mDet4_1234_0234 = pM[M10]*mDet3_234_234 - pM[M12]*mDet3_234_034
                                 + pM[M13]*mDet3_234_024 - pM[M14]*mDet3_234_023;
  const Double_t mDet4_1234_1234 = pM[M11]*mDet3_234_234 - pM[M12]*mDet3_234_134
                                 + pM[M13]*mDet3_234_124 - pM[M14]*mDet3_234_123;

  // Find the 5x5 det:

  const Double_t det = pM[M00]*mDet4_1234_1234 - pM[M01]*mDet4_1234_0234 + pM[M02]*mDet4_1234_0134
                       - pM[M03]*mDet4_1234_0124 + pM[M04]*mDet4_1234_0123;
  if (determ)
    *determ = det;

  if ( det == 0 )
    return kFALSE;

  const Double_t oneOverDet = 1.0/det;
  const Double_t mn1OverDet = - oneOverDet;

  pM[M00] = mDet4_1234_1234 * oneOverDet;
  pM[M01] = mDet4_1234_0234 * mn1OverDet;
  pM[M02] = mDet4_1234_0134 * oneOverDet;
  pM[M03] = mDet4_1234_0124 * mn1OverDet;
  pM[M04] = mDet4_1234_0123 * oneOverDet;

  pM[M11] = mDet4_0234_0234 * oneOverDet;
  pM[M12] = mDet4_0234_0134 * mn1OverDet;
  pM[M13] = mDet4_0234_0124 * oneOverDet;
  pM[M14] = mDet4_0234_0123 * mn1OverDet;

  pM[M22] = mDet4_0134_0134 * oneOverDet;
  pM[M23] = mDet4_0134_0124 * mn1OverDet;
  pM[M24] = mDet4_0134_0123 * oneOverDet;

  pM[M33] = mDet4_0124_0124 * oneOverDet;
  pM[M34] = mDet4_0124_0123 * mn1OverDet;

  pM[M44] = mDet4_0123_0123 * oneOverDet;

  for (Int_t irow = 0; irow < 5; irow++) {
    const Int_t rowOff1 = irow*5; 
    for (Int_t icol = 0; icol < irow; icol++) {
      const Int_t rowOff2 = icol*5; 
      pM[rowOff1+icol] = pM[rowOff2+irow];
    }
  } 

  return kTRUE;
}

// Aij are indices for a 6x6 symmetric matrix.

#define A00 0
#define A01 1
#define A02 2
#define A03 3
#define A04 4
#define A05 5

#define A10 1
#define A11 7
#define A12 8
#define A13 9
#define A14 10
#define A15 11

#define A20 2
#define A21 8
#define A22 14
#define A23 15
#define A24 16
#define A25 17

#define A30 3
#define A31 9
#define A32 15
#define A33 21
#define A34 22
#define A35 23

#define A40 4
#define A41 10
#define A42 16
#define A43 22
#define A44 28
#define A45 29

#define A50 5
#define A51 11
#define A52 17
#define A53 23
#define A54 29
#define A55 35

//______________________________________________________________________________
Bool_t TMatrixDSymCramerInv::Inv6x6(TMatrixDSym &m,Double_t *determ)
{
  if (m.GetNrows() != 6 || m.GetNcols() != 6 || m.GetRowLwb() != m.GetColLwb()) {
    Error("Inv6x6","matrix should be square 6x6");
    m.Invalidate();
    return kFALSE;
  }

  Double_t *pM = m.GetMatrixArray();

  // Find all NECESSARY 2x2 dets:  (39 of them)

  const Double_t mDet2_34_01 = pM[A30]*pM[A41] - pM[A31]*pM[A40];
  const Double_t mDet2_34_02 = pM[A30]*pM[A42] - pM[A32]*pM[A40];
  const Double_t mDet2_34_03 = pM[A30]*pM[A43] - pM[A33]*pM[A40];
  const Double_t mDet2_34_04 = pM[A30]*pM[A44] - pM[A34]*pM[A40];
  const Double_t mDet2_34_12 = pM[A31]*pM[A42] - pM[A32]*pM[A41];
  const Double_t mDet2_34_13 = pM[A31]*pM[A43] - pM[A33]*pM[A41];
  const Double_t mDet2_34_14 = pM[A31]*pM[A44] - pM[A34]*pM[A41];
  const Double_t mDet2_34_23 = pM[A32]*pM[A43] - pM[A33]*pM[A42];
  const Double_t mDet2_34_24 = pM[A32]*pM[A44] - pM[A34]*pM[A42];
  const Double_t mDet2_34_34 = pM[A33]*pM[A44] - pM[A34]*pM[A43];
  const Double_t mDet2_35_01 = pM[A30]*pM[A51] - pM[A31]*pM[A50];
  const Double_t mDet2_35_02 = pM[A30]*pM[A52] - pM[A32]*pM[A50];
  const Double_t mDet2_35_03 = pM[A30]*pM[A53] - pM[A33]*pM[A50];
  const Double_t mDet2_35_04 = pM[A30]*pM[A54] - pM[A34]*pM[A50];
  const Double_t mDet2_35_05 = pM[A30]*pM[A55] - pM[A35]*pM[A50];
  const Double_t mDet2_35_12 = pM[A31]*pM[A52] - pM[A32]*pM[A51];
  const Double_t mDet2_35_13 = pM[A31]*pM[A53] - pM[A33]*pM[A51];
  const Double_t mDet2_35_14 = pM[A31]*pM[A54] - pM[A34]*pM[A51];
  const Double_t mDet2_35_15 = pM[A31]*pM[A55] - pM[A35]*pM[A51];
  const Double_t mDet2_35_23 = pM[A32]*pM[A53] - pM[A33]*pM[A52];
  const Double_t mDet2_35_24 = pM[A32]*pM[A54] - pM[A34]*pM[A52];
  const Double_t mDet2_35_25 = pM[A32]*pM[A55] - pM[A35]*pM[A52];
  const Double_t mDet2_35_34 = pM[A33]*pM[A54] - pM[A34]*pM[A53];
  const Double_t mDet2_35_35 = pM[A33]*pM[A55] - pM[A35]*pM[A53];
  const Double_t mDet2_45_01 = pM[A40]*pM[A51] - pM[A41]*pM[A50];
  const Double_t mDet2_45_02 = pM[A40]*pM[A52] - pM[A42]*pM[A50];
  const Double_t mDet2_45_03 = pM[A40]*pM[A53] - pM[A43]*pM[A50];
  const Double_t mDet2_45_04 = pM[A40]*pM[A54] - pM[A44]*pM[A50];
  const Double_t mDet2_45_05 = pM[A40]*pM[A55] - pM[A45]*pM[A50];
  const Double_t mDet2_45_12 = pM[A41]*pM[A52] - pM[A42]*pM[A51];
  const Double_t mDet2_45_13 = pM[A41]*pM[A53] - pM[A43]*pM[A51];
  const Double_t mDet2_45_14 = pM[A41]*pM[A54] - pM[A44]*pM[A51];
  const Double_t mDet2_45_15 = pM[A41]*pM[A55] - pM[A45]*pM[A51];
  const Double_t mDet2_45_23 = pM[A42]*pM[A53] - pM[A43]*pM[A52];
  const Double_t mDet2_45_24 = pM[A42]*pM[A54] - pM[A44]*pM[A52];
  const Double_t mDet2_45_25 = pM[A42]*pM[A55] - pM[A45]*pM[A52];
  const Double_t mDet2_45_34 = pM[A43]*pM[A54] - pM[A44]*pM[A53];
  const Double_t mDet2_45_35 = pM[A43]*pM[A55] - pM[A45]*pM[A53];
  const Double_t mDet2_45_45 = pM[A44]*pM[A55] - pM[A45]*pM[A54];

  // Find all NECESSARY 3x3 dets:  (65 of them)

  const Double_t mDet3_234_012 = pM[A20]*mDet2_34_12 - pM[A21]*mDet2_34_02 + pM[A22]*mDet2_34_01;
  const Double_t mDet3_234_013 = pM[A20]*mDet2_34_13 - pM[A21]*mDet2_34_03 + pM[A23]*mDet2_34_01;
  const Double_t mDet3_234_014 = pM[A20]*mDet2_34_14 - pM[A21]*mDet2_34_04 + pM[A24]*mDet2_34_01;
  const Double_t mDet3_234_023 = pM[A20]*mDet2_34_23 - pM[A22]*mDet2_34_03 + pM[A23]*mDet2_34_02;
  const Double_t mDet3_234_024 = pM[A20]*mDet2_34_24 - pM[A22]*mDet2_34_04 + pM[A24]*mDet2_34_02;
  const Double_t mDet3_234_034 = pM[A20]*mDet2_34_34 - pM[A23]*mDet2_34_04 + pM[A24]*mDet2_34_03;
  const Double_t mDet3_234_123 = pM[A21]*mDet2_34_23 - pM[A22]*mDet2_34_13 + pM[A23]*mDet2_34_12;
  const Double_t mDet3_234_124 = pM[A21]*mDet2_34_24 - pM[A22]*mDet2_34_14 + pM[A24]*mDet2_34_12;
  const Double_t mDet3_234_134 = pM[A21]*mDet2_34_34 - pM[A23]*mDet2_34_14 + pM[A24]*mDet2_34_13;
  const Double_t mDet3_234_234 = pM[A22]*mDet2_34_34 - pM[A23]*mDet2_34_24 + pM[A24]*mDet2_34_23;
  const Double_t mDet3_235_012 = pM[A20]*mDet2_35_12 - pM[A21]*mDet2_35_02 + pM[A22]*mDet2_35_01;
  const Double_t mDet3_235_013 = pM[A20]*mDet2_35_13 - pM[A21]*mDet2_35_03 + pM[A23]*mDet2_35_01;
  const Double_t mDet3_235_014 = pM[A20]*mDet2_35_14 - pM[A21]*mDet2_35_04 + pM[A24]*mDet2_35_01;
  const Double_t mDet3_235_015 = pM[A20]*mDet2_35_15 - pM[A21]*mDet2_35_05 + pM[A25]*mDet2_35_01;
  const Double_t mDet3_235_023 = pM[A20]*mDet2_35_23 - pM[A22]*mDet2_35_03 + pM[A23]*mDet2_35_02;
  const Double_t mDet3_235_024 = pM[A20]*mDet2_35_24 - pM[A22]*mDet2_35_04 + pM[A24]*mDet2_35_02;
  const Double_t mDet3_235_025 = pM[A20]*mDet2_35_25 - pM[A22]*mDet2_35_05 + pM[A25]*mDet2_35_02;
  const Double_t mDet3_235_034 = pM[A20]*mDet2_35_34 - pM[A23]*mDet2_35_04 + pM[A24]*mDet2_35_03;
  const Double_t mDet3_235_035 = pM[A20]*mDet2_35_35 - pM[A23]*mDet2_35_05 + pM[A25]*mDet2_35_03;
  const Double_t mDet3_235_123 = pM[A21]*mDet2_35_23 - pM[A22]*mDet2_35_13 + pM[A23]*mDet2_35_12;
  const Double_t mDet3_235_124 = pM[A21]*mDet2_35_24 - pM[A22]*mDet2_35_14 + pM[A24]*mDet2_35_12;
  const Double_t mDet3_235_125 = pM[A21]*mDet2_35_25 - pM[A22]*mDet2_35_15 + pM[A25]*mDet2_35_12;
  const Double_t mDet3_235_134 = pM[A21]*mDet2_35_34 - pM[A23]*mDet2_35_14 + pM[A24]*mDet2_35_13;
  const Double_t mDet3_235_135 = pM[A21]*mDet2_35_35 - pM[A23]*mDet2_35_15 + pM[A25]*mDet2_35_13;
  const Double_t mDet3_235_234 = pM[A22]*mDet2_35_34 - pM[A23]*mDet2_35_24 + pM[A24]*mDet2_35_23;
  const Double_t mDet3_235_235 = pM[A22]*mDet2_35_35 - pM[A23]*mDet2_35_25 + pM[A25]*mDet2_35_23;
  const Double_t mDet3_245_012 = pM[A20]*mDet2_45_12 - pM[A21]*mDet2_45_02 + pM[A22]*mDet2_45_01;
  const Double_t mDet3_245_013 = pM[A20]*mDet2_45_13 - pM[A21]*mDet2_45_03 + pM[A23]*mDet2_45_01;
  const Double_t mDet3_245_014 = pM[A20]*mDet2_45_14 - pM[A21]*mDet2_45_04 + pM[A24]*mDet2_45_01;
  const Double_t mDet3_245_015 = pM[A20]*mDet2_45_15 - pM[A21]*mDet2_45_05 + pM[A25]*mDet2_45_01;
  const Double_t mDet3_245_023 = pM[A20]*mDet2_45_23 - pM[A22]*mDet2_45_03 + pM[A23]*mDet2_45_02;
  const Double_t mDet3_245_024 = pM[A20]*mDet2_45_24 - pM[A22]*mDet2_45_04 + pM[A24]*mDet2_45_02;
  const Double_t mDet3_245_025 = pM[A20]*mDet2_45_25 - pM[A22]*mDet2_45_05 + pM[A25]*mDet2_45_02;
  const Double_t mDet3_245_034 = pM[A20]*mDet2_45_34 - pM[A23]*mDet2_45_04 + pM[A24]*mDet2_45_03;
  const Double_t mDet3_245_035 = pM[A20]*mDet2_45_35 - pM[A23]*mDet2_45_05 + pM[A25]*mDet2_45_03;
  const Double_t mDet3_245_045 = pM[A20]*mDet2_45_45 - pM[A24]*mDet2_45_05 + pM[A25]*mDet2_45_04;
  const Double_t mDet3_245_123 = pM[A21]*mDet2_45_23 - pM[A22]*mDet2_45_13 + pM[A23]*mDet2_45_12;
  const Double_t mDet3_245_124 = pM[A21]*mDet2_45_24 - pM[A22]*mDet2_45_14 + pM[A24]*mDet2_45_12;
  const Double_t mDet3_245_125 = pM[A21]*mDet2_45_25 - pM[A22]*mDet2_45_15 + pM[A25]*mDet2_45_12;
  const Double_t mDet3_245_134 = pM[A21]*mDet2_45_34 - pM[A23]*mDet2_45_14 + pM[A24]*mDet2_45_13;
  const Double_t mDet3_245_135 = pM[A21]*mDet2_45_35 - pM[A23]*mDet2_45_15 + pM[A25]*mDet2_45_13;
  const Double_t mDet3_245_145 = pM[A21]*mDet2_45_45 - pM[A24]*mDet2_45_15 + pM[A25]*mDet2_45_14;
  const Double_t mDet3_245_234 = pM[A22]*mDet2_45_34 - pM[A23]*mDet2_45_24 + pM[A24]*mDet2_45_23;
  const Double_t mDet3_245_235 = pM[A22]*mDet2_45_35 - pM[A23]*mDet2_45_25 + pM[A25]*mDet2_45_23;
  const Double_t mDet3_245_245 = pM[A22]*mDet2_45_45 - pM[A24]*mDet2_45_25 + pM[A25]*mDet2_45_24;
  const Double_t mDet3_345_012 = pM[A30]*mDet2_45_12 - pM[A31]*mDet2_45_02 + pM[A32]*mDet2_45_01;
  const Double_t mDet3_345_013 = pM[A30]*mDet2_45_13 - pM[A31]*mDet2_45_03 + pM[A33]*mDet2_45_01;
  const Double_t mDet3_345_014 = pM[A30]*mDet2_45_14 - pM[A31]*mDet2_45_04 + pM[A34]*mDet2_45_01;
  const Double_t mDet3_345_015 = pM[A30]*mDet2_45_15 - pM[A31]*mDet2_45_05 + pM[A35]*mDet2_45_01;
  const Double_t mDet3_345_023 = pM[A30]*mDet2_45_23 - pM[A32]*mDet2_45_03 + pM[A33]*mDet2_45_02;
  const Double_t mDet3_345_024 = pM[A30]*mDet2_45_24 - pM[A32]*mDet2_45_04 + pM[A34]*mDet2_45_02;
  const Double_t mDet3_345_025 = pM[A30]*mDet2_45_25 - pM[A32]*mDet2_45_05 + pM[A35]*mDet2_45_02;
  const Double_t mDet3_345_034 = pM[A30]*mDet2_45_34 - pM[A33]*mDet2_45_04 + pM[A34]*mDet2_45_03;
  const Double_t mDet3_345_035 = pM[A30]*mDet2_45_35 - pM[A33]*mDet2_45_05 + pM[A35]*mDet2_45_03;
  const Double_t mDet3_345_045 = pM[A30]*mDet2_45_45 - pM[A34]*mDet2_45_05 + pM[A35]*mDet2_45_04;
  const Double_t mDet3_345_123 = pM[A31]*mDet2_45_23 - pM[A32]*mDet2_45_13 + pM[A33]*mDet2_45_12;
  const Double_t mDet3_345_124 = pM[A31]*mDet2_45_24 - pM[A32]*mDet2_45_14 + pM[A34]*mDet2_45_12;
  const Double_t mDet3_345_125 = pM[A31]*mDet2_45_25 - pM[A32]*mDet2_45_15 + pM[A35]*mDet2_45_12;
  const Double_t mDet3_345_134 = pM[A31]*mDet2_45_34 - pM[A33]*mDet2_45_14 + pM[A34]*mDet2_45_13;
  const Double_t mDet3_345_135 = pM[A31]*mDet2_45_35 - pM[A33]*mDet2_45_15 + pM[A35]*mDet2_45_13;
  const Double_t mDet3_345_145 = pM[A31]*mDet2_45_45 - pM[A34]*mDet2_45_15 + pM[A35]*mDet2_45_14;
  const Double_t mDet3_345_234 = pM[A32]*mDet2_45_34 - pM[A33]*mDet2_45_24 + pM[A34]*mDet2_45_23;
  const Double_t mDet3_345_235 = pM[A32]*mDet2_45_35 - pM[A33]*mDet2_45_25 + pM[A35]*mDet2_45_23;
  const Double_t mDet3_345_245 = pM[A32]*mDet2_45_45 - pM[A34]*mDet2_45_25 + pM[A35]*mDet2_45_24;
  const Double_t mDet3_345_345 = pM[A33]*mDet2_45_45 - pM[A34]*mDet2_45_35 + pM[A35]*mDet2_45_34;

  // Find all NECESSARY 4x4 dets:  (55 of them)

  const Double_t mDet4_1234_0123 = pM[A10]*mDet3_234_123 - pM[A11]*mDet3_234_023
                                 + pM[A12]*mDet3_234_013 - pM[A13]*mDet3_234_012;
  const Double_t mDet4_1234_0124 = pM[A10]*mDet3_234_124 - pM[A11]*mDet3_234_024
                                 + pM[A12]*mDet3_234_014 - pM[A14]*mDet3_234_012;
  const Double_t mDet4_1234_0134 = pM[A10]*mDet3_234_134 - pM[A11]*mDet3_234_034
                                 + pM[A13]*mDet3_234_014 - pM[A14]*mDet3_234_013;
  const Double_t mDet4_1234_0234 = pM[A10]*mDet3_234_234 - pM[A12]*mDet3_234_034
                                 + pM[A13]*mDet3_234_024 - pM[A14]*mDet3_234_023;
  const Double_t mDet4_1234_1234 = pM[A11]*mDet3_234_234 - pM[A12]*mDet3_234_134
                                 + pM[A13]*mDet3_234_124 - pM[A14]*mDet3_234_123;
  const Double_t mDet4_1235_0123 = pM[A10]*mDet3_235_123 - pM[A11]*mDet3_235_023
                                 + pM[A12]*mDet3_235_013 - pM[A13]*mDet3_235_012;
  const Double_t mDet4_1235_0124 = pM[A10]*mDet3_235_124 - pM[A11]*mDet3_235_024
                                 + pM[A12]*mDet3_235_014 - pM[A14]*mDet3_235_012;
  const Double_t mDet4_1235_0125 = pM[A10]*mDet3_235_125 - pM[A11]*mDet3_235_025
                                 + pM[A12]*mDet3_235_015 - pM[A15]*mDet3_235_012;
  const Double_t mDet4_1235_0134 = pM[A10]*mDet3_235_134 - pM[A11]*mDet3_235_034
                                 + pM[A13]*mDet3_235_014 - pM[A14]*mDet3_235_013;
  const Double_t mDet4_1235_0135 = pM[A10]*mDet3_235_135 - pM[A11]*mDet3_235_035
                                 + pM[A13]*mDet3_235_015 - pM[A15]*mDet3_235_013;
  const Double_t mDet4_1235_0234 = pM[A10]*mDet3_235_234 - pM[A12]*mDet3_235_034
                                 + pM[A13]*mDet3_235_024 - pM[A14]*mDet3_235_023;
  const Double_t mDet4_1235_0235 = pM[A10]*mDet3_235_235 - pM[A12]*mDet3_235_035
                                 + pM[A13]*mDet3_235_025 - pM[A15]*mDet3_235_023;
  const Double_t mDet4_1235_1234 = pM[A11]*mDet3_235_234 - pM[A12]*mDet3_235_134
                                 + pM[A13]*mDet3_235_124 - pM[A14]*mDet3_235_123;
  const Double_t mDet4_1235_1235 = pM[A11]*mDet3_235_235 - pM[A12]*mDet3_235_135
                                 + pM[A13]*mDet3_235_125 - pM[A15]*mDet3_235_123;
  const Double_t mDet4_1245_0123 = pM[A10]*mDet3_245_123 - pM[A11]*mDet3_245_023
                                 + pM[A12]*mDet3_245_013 - pM[A13]*mDet3_245_012;
  const Double_t mDet4_1245_0124 = pM[A10]*mDet3_245_124 - pM[A11]*mDet3_245_024
                                 + pM[A12]*mDet3_245_014 - pM[A14]*mDet3_245_012;
  const Double_t mDet4_1245_0125 = pM[A10]*mDet3_245_125 - pM[A11]*mDet3_245_025
                                 + pM[A12]*mDet3_245_015 - pM[A15]*mDet3_245_012;
  const Double_t mDet4_1245_0134 = pM[A10]*mDet3_245_134 - pM[A11]*mDet3_245_034
                                 + pM[A13]*mDet3_245_014 - pM[A14]*mDet3_245_013;
  const Double_t mDet4_1245_0135 = pM[A10]*mDet3_245_135 - pM[A11]*mDet3_245_035
                                 + pM[A13]*mDet3_245_015 - pM[A15]*mDet3_245_013;
  const Double_t mDet4_1245_0145 = pM[A10]*mDet3_245_145 - pM[A11]*mDet3_245_045
                                 + pM[A14]*mDet3_245_015 - pM[A15]*mDet3_245_014;
  const Double_t mDet4_1245_0234 = pM[A10]*mDet3_245_234 - pM[A12]*mDet3_245_034
                                 + pM[A13]*mDet3_245_024 - pM[A14]*mDet3_245_023;
  const Double_t mDet4_1245_0235 = pM[A10]*mDet3_245_235 - pM[A12]*mDet3_245_035
                                 + pM[A13]*mDet3_245_025 - pM[A15]*mDet3_245_023;
  const Double_t mDet4_1245_0245 = pM[A10]*mDet3_245_245 - pM[A12]*mDet3_245_045
                                 + pM[A14]*mDet3_245_025 - pM[A15]*mDet3_245_024;
  const Double_t mDet4_1245_1234 = pM[A11]*mDet3_245_234 - pM[A12]*mDet3_245_134
                                 + pM[A13]*mDet3_245_124 - pM[A14]*mDet3_245_123;
  const Double_t mDet4_1245_1235 = pM[A11]*mDet3_245_235 - pM[A12]*mDet3_245_135
                                 + pM[A13]*mDet3_245_125 - pM[A15]*mDet3_245_123;
  const Double_t mDet4_1245_1245 = pM[A11]*mDet3_245_245 - pM[A12]*mDet3_245_145
                                 + pM[A14]*mDet3_245_125 - pM[A15]*mDet3_245_124;
  const Double_t mDet4_1345_0123 = pM[A10]*mDet3_345_123 - pM[A11]*mDet3_345_023
                                 + pM[A12]*mDet3_345_013 - pM[A13]*mDet3_345_012;
  const Double_t mDet4_1345_0124 = pM[A10]*mDet3_345_124 - pM[A11]*mDet3_345_024
                                 + pM[A12]*mDet3_345_014 - pM[A14]*mDet3_345_012;
  const Double_t mDet4_1345_0125 = pM[A10]*mDet3_345_125 - pM[A11]*mDet3_345_025
                                 + pM[A12]*mDet3_345_015 - pM[A15]*mDet3_345_012;
  const Double_t mDet4_1345_0134 = pM[A10]*mDet3_345_134 - pM[A11]*mDet3_345_034
                                 + pM[A13]*mDet3_345_014 - pM[A14]*mDet3_345_013;
  const Double_t mDet4_1345_0135 = pM[A10]*mDet3_345_135 - pM[A11]*mDet3_345_035
                                 + pM[A13]*mDet3_345_015 - pM[A15]*mDet3_345_013;
  const Double_t mDet4_1345_0145 = pM[A10]*mDet3_345_145 - pM[A11]*mDet3_345_045
                                 + pM[A14]*mDet3_345_015 - pM[A15]*mDet3_345_014;
  const Double_t mDet4_1345_0234 = pM[A10]*mDet3_345_234 - pM[A12]*mDet3_345_034
                                 + pM[A13]*mDet3_345_024 - pM[A14]*mDet3_345_023;
  const Double_t mDet4_1345_0235 = pM[A10]*mDet3_345_235 - pM[A12]*mDet3_345_035
                                 + pM[A13]*mDet3_345_025 - pM[A15]*mDet3_345_023;
  const Double_t mDet4_1345_0245 = pM[A10]*mDet3_345_245 - pM[A12]*mDet3_345_045
                                 + pM[A14]*mDet3_345_025 - pM[A15]*mDet3_345_024;
  const Double_t mDet4_1345_0345 = pM[A10]*mDet3_345_345 - pM[A13]*mDet3_345_045
                                 + pM[A14]*mDet3_345_035 - pM[A15]*mDet3_345_034;
  const Double_t mDet4_1345_1234 = pM[A11]*mDet3_345_234 - pM[A12]*mDet3_345_134
                                 + pM[A13]*mDet3_345_124 - pM[A14]*mDet3_345_123;
  const Double_t mDet4_1345_1235 = pM[A11]*mDet3_345_235 - pM[A12]*mDet3_345_135
                                 + pM[A13]*mDet3_345_125 - pM[A15]*mDet3_345_123;
  const Double_t mDet4_1345_1245 = pM[A11]*mDet3_345_245 - pM[A12]*mDet3_345_145
                                 + pM[A14]*mDet3_345_125 - pM[A15]*mDet3_345_124;
  const Double_t mDet4_1345_1345 = pM[A11]*mDet3_345_345 - pM[A13]*mDet3_345_145
                                 + pM[A14]*mDet3_345_135 - pM[A15]*mDet3_345_134;
  const Double_t mDet4_2345_0123 = pM[A20]*mDet3_345_123 - pM[A21]*mDet3_345_023
                                 + pM[A22]*mDet3_345_013 - pM[A23]*mDet3_345_012;
  const Double_t mDet4_2345_0124 = pM[A20]*mDet3_345_124 - pM[A21]*mDet3_345_024
                                 + pM[A22]*mDet3_345_014 - pM[A24]*mDet3_345_012;
  const Double_t mDet4_2345_0125 = pM[A20]*mDet3_345_125 - pM[A21]*mDet3_345_025
                                 + pM[A22]*mDet3_345_015 - pM[A25]*mDet3_345_012;
  const Double_t mDet4_2345_0134 = pM[A20]*mDet3_345_134 - pM[A21]*mDet3_345_034
                                 + pM[A23]*mDet3_345_014 - pM[A24]*mDet3_345_013;
  const Double_t mDet4_2345_0135 = pM[A20]*mDet3_345_135 - pM[A21]*mDet3_345_035
                                 + pM[A23]*mDet3_345_015 - pM[A25]*mDet3_345_013;
  const Double_t mDet4_2345_0145 = pM[A20]*mDet3_345_145 - pM[A21]*mDet3_345_045
                                 + pM[A24]*mDet3_345_015 - pM[A25]*mDet3_345_014;
  const Double_t mDet4_2345_0234 = pM[A20]*mDet3_345_234 - pM[A22]*mDet3_345_034
                                 + pM[A23]*mDet3_345_024 - pM[A24]*mDet3_345_023;
  const Double_t mDet4_2345_0235 = pM[A20]*mDet3_345_235 - pM[A22]*mDet3_345_035
                                 + pM[A23]*mDet3_345_025 - pM[A25]*mDet3_345_023;
  const Double_t mDet4_2345_0245 = pM[A20]*mDet3_345_245 - pM[A22]*mDet3_345_045
                                 + pM[A24]*mDet3_345_025 - pM[A25]*mDet3_345_024;
  const Double_t mDet4_2345_0345 = pM[A20]*mDet3_345_345 - pM[A23]*mDet3_345_045
                                 + pM[A24]*mDet3_345_035 - pM[A25]*mDet3_345_034;
  const Double_t mDet4_2345_1234 = pM[A21]*mDet3_345_234 - pM[A22]*mDet3_345_134
                                 + pM[A23]*mDet3_345_124 - pM[A24]*mDet3_345_123;
  const Double_t mDet4_2345_1235 = pM[A21]*mDet3_345_235 - pM[A22]*mDet3_345_135
                                 + pM[A23]*mDet3_345_125 - pM[A25]*mDet3_345_123;
  const Double_t mDet4_2345_1245 = pM[A21]*mDet3_345_245 - pM[A22]*mDet3_345_145
                                 + pM[A24]*mDet3_345_125 - pM[A25]*mDet3_345_124;
  const Double_t mDet4_2345_1345 = pM[A21]*mDet3_345_345 - pM[A23]*mDet3_345_145
                                 + pM[A24]*mDet3_345_135 - pM[A25]*mDet3_345_134;
  const Double_t mDet4_2345_2345 = pM[A22]*mDet3_345_345 - pM[A23]*mDet3_345_245
                                 + pM[A24]*mDet3_345_235 - pM[A25]*mDet3_345_234;

  // Find all NECESSARY 5x5 dets:  (19 of them)

  const Double_t mDet5_01234_01234 = pM[A00]*mDet4_1234_1234 - pM[A01]*mDet4_1234_0234
                                   + pM[A02]*mDet4_1234_0134 - pM[A03]*mDet4_1234_0124 + pM[A04]*mDet4_1234_0123;
  const Double_t mDet5_01235_01234 = pM[A00]*mDet4_1235_1234 - pM[A01]*mDet4_1235_0234
                                   + pM[A02]*mDet4_1235_0134 - pM[A03]*mDet4_1235_0124 + pM[A04]*mDet4_1235_0123;
  const Double_t mDet5_01235_01235 = pM[A00]*mDet4_1235_1235 - pM[A01]*mDet4_1235_0235
                                   + pM[A02]*mDet4_1235_0135 - pM[A03]*mDet4_1235_0125 + pM[A05]*mDet4_1235_0123;
  const Double_t mDet5_01245_01234 = pM[A00]*mDet4_1245_1234 - pM[A01]*mDet4_1245_0234
                                   + pM[A02]*mDet4_1245_0134 - pM[A03]*mDet4_1245_0124 + pM[A04]*mDet4_1245_0123;
  const Double_t mDet5_01245_01235 = pM[A00]*mDet4_1245_1235 - pM[A01]*mDet4_1245_0235
                                   + pM[A02]*mDet4_1245_0135 - pM[A03]*mDet4_1245_0125 + pM[A05]*mDet4_1245_0123;
  const Double_t mDet5_01245_01245 = pM[A00]*mDet4_1245_1245 - pM[A01]*mDet4_1245_0245
                                   + pM[A02]*mDet4_1245_0145 - pM[A04]*mDet4_1245_0125 + pM[A05]*mDet4_1245_0124;
  const Double_t mDet5_01345_01234 = pM[A00]*mDet4_1345_1234 - pM[A01]*mDet4_1345_0234
                                   + pM[A02]*mDet4_1345_0134 - pM[A03]*mDet4_1345_0124 + pM[A04]*mDet4_1345_0123;
  const Double_t mDet5_01345_01235 = pM[A00]*mDet4_1345_1235 - pM[A01]*mDet4_1345_0235
                                   + pM[A02]*mDet4_1345_0135 - pM[A03]*mDet4_1345_0125 + pM[A05]*mDet4_1345_0123;
  const Double_t mDet5_01345_01245 = pM[A00]*mDet4_1345_1245 - pM[A01]*mDet4_1345_0245
                                   + pM[A02]*mDet4_1345_0145 - pM[A04]*mDet4_1345_0125 + pM[A05]*mDet4_1345_0124;
  const Double_t mDet5_01345_01345 = pM[A00]*mDet4_1345_1345 - pM[A01]*mDet4_1345_0345
                                   + pM[A03]*mDet4_1345_0145 - pM[A04]*mDet4_1345_0135 + pM[A05]*mDet4_1345_0134;
  const Double_t mDet5_02345_01234 = pM[A00]*mDet4_2345_1234 - pM[A01]*mDet4_2345_0234
                                   + pM[A02]*mDet4_2345_0134 - pM[A03]*mDet4_2345_0124 + pM[A04]*mDet4_2345_0123;
  const Double_t mDet5_02345_01235 = pM[A00]*mDet4_2345_1235 - pM[A01]*mDet4_2345_0235
                                   + pM[A02]*mDet4_2345_0135 - pM[A03]*mDet4_2345_0125 + pM[A05]*mDet4_2345_0123;
  const Double_t mDet5_02345_01245 = pM[A00]*mDet4_2345_1245 - pM[A01]*mDet4_2345_0245
                                   + pM[A02]*mDet4_2345_0145 - pM[A04]*mDet4_2345_0125 + pM[A05]*mDet4_2345_0124;
  const Double_t mDet5_02345_01345 = pM[A00]*mDet4_2345_1345 - pM[A01]*mDet4_2345_0345
                                   + pM[A03]*mDet4_2345_0145 - pM[A04]*mDet4_2345_0135 + pM[A05]*mDet4_2345_0134;
  const Double_t mDet5_02345_02345 = pM[A00]*mDet4_2345_2345 - pM[A02]*mDet4_2345_0345
                                   + pM[A03]*mDet4_2345_0245 - pM[A04]*mDet4_2345_0235 + pM[A05]*mDet4_2345_0234;
  const Double_t mDet5_12345_01234 = pM[A10]*mDet4_2345_1234 - pM[A11]*mDet4_2345_0234
                                   + pM[A12]*mDet4_2345_0134 - pM[A13]*mDet4_2345_0124 + pM[A14]*mDet4_2345_0123;
  const Double_t mDet5_12345_01235 = pM[A10]*mDet4_2345_1235 - pM[A11]*mDet4_2345_0235
                                   + pM[A12]*mDet4_2345_0135 - pM[A13]*mDet4_2345_0125 + pM[A15]*mDet4_2345_0123;
  const Double_t mDet5_12345_01245 = pM[A10]*mDet4_2345_1245 - pM[A11]*mDet4_2345_0245
                                   + pM[A12]*mDet4_2345_0145 - pM[A14]*mDet4_2345_0125 + pM[A15]*mDet4_2345_0124;
  const Double_t mDet5_12345_01345 = pM[A10]*mDet4_2345_1345 - pM[A11]*mDet4_2345_0345
                                   + pM[A13]*mDet4_2345_0145 - pM[A14]*mDet4_2345_0135 + pM[A15]*mDet4_2345_0134;
  const Double_t mDet5_12345_02345 = pM[A10]*mDet4_2345_2345 - pM[A12]*mDet4_2345_0345
                                   + pM[A13]*mDet4_2345_0245 - pM[A14]*mDet4_2345_0235 + pM[A15]*mDet4_2345_0234;
  const Double_t mDet5_12345_12345 = pM[A11]*mDet4_2345_2345 - pM[A12]*mDet4_2345_1345
                                   + pM[A13]*mDet4_2345_1245 - pM[A14]*mDet4_2345_1235 + pM[A15]*mDet4_2345_1234;

  // Find the determinant 

  const Double_t det = pM[A00]*mDet5_12345_12345 - pM[A01]*mDet5_12345_02345 + pM[A02]*mDet5_12345_01345
                     - pM[A03]*mDet5_12345_01245 + pM[A04]*mDet5_12345_01235 - pM[A05]*mDet5_12345_01234;

  if (determ)
    *determ = det;

  if ( det == 0 )
    m.Invalidate();

  const Double_t oneOverDet = 1.0/det;
  const Double_t mn1OverDet = - oneOverDet;

  pM[A00] =  mDet5_12345_12345*oneOverDet;
  pM[A01] =  mDet5_12345_02345*mn1OverDet;
  pM[A02] =  mDet5_12345_01345*oneOverDet;
  pM[A03] =  mDet5_12345_01245*mn1OverDet;
  pM[A04] =  mDet5_12345_01235*oneOverDet;
  pM[A05] =  mDet5_12345_01234*mn1OverDet;

  pM[A11] =  mDet5_02345_02345*oneOverDet;
  pM[A12] =  mDet5_02345_01345*mn1OverDet;
  pM[A13] =  mDet5_02345_01245*oneOverDet;
  pM[A14] =  mDet5_02345_01235*mn1OverDet;
  pM[A15] =  mDet5_02345_01234*oneOverDet;

  pM[A22] =  mDet5_01345_01345*oneOverDet;
  pM[A23] =  mDet5_01345_01245*mn1OverDet;
  pM[A24] =  mDet5_01345_01235*oneOverDet;
  pM[A25] =  mDet5_01345_01234*mn1OverDet;

  pM[A33] =  mDet5_01245_01245*oneOverDet;
  pM[A34] =  mDet5_01245_01235*mn1OverDet;
  pM[A35] =  mDet5_01245_01234*oneOverDet;

  pM[A44] =  mDet5_01235_01235*oneOverDet;
  pM[A45] =  mDet5_01235_01234*mn1OverDet;

  pM[A55] =  mDet5_01234_01234*oneOverDet;

  for (Int_t irow = 0; irow < 6; irow++) {
    const Int_t rowOff1 = irow*6; 
    for (Int_t icol = 0; icol < irow; icol++) {
      const Int_t rowOff2 = icol*6; 
      pM[rowOff1+icol] = pM[rowOff2+irow];
    }
  } 

  return kTRUE;
}
