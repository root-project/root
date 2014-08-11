// @(#)root/base:$Id$
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
// TMatrixTSymCramerInv                                                 //
//                                                                      //
// Encapsulate templates of Cramer Inversion routines.                  //
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

#include "TMatrixTSymCramerInv.h"

#if !defined(R__SOLARIS) && !defined(R__ACC) && !defined(R__FBSD)
NamespaceImp(TMatrixTSymCramerInv);
#endif

//______________________________________________________________________________
template<class Element>
Bool_t TMatrixTSymCramerInv::Inv2x2(TMatrixTSym<Element> &m,Double_t *determ)
{
   if (m.GetNrows() != 2) {
      Error("Inv2x2","matrix should be square 2x2");
      return kFALSE;
   }

   Element *pM = m.GetMatrixArray();

   const Double_t det = pM[0] * pM[3] - pM[1] * pM[1];

   if (determ)
      *determ = det;

   if ( det == 0 ) {
      Error("Inv2x2","matrix is singular");
      return kFALSE;
   }

   const Double_t tmp1 =   pM[3] / det;
   pM[3] = pM[0] / det;
   pM[2] = pM[1] = -pM[1] / det;
   pM[0] = tmp1;

   return kTRUE;
}

//______________________________________________________________________________
template<class Element>
Bool_t TMatrixTSymCramerInv::Inv3x3(TMatrixTSym<Element> &m,Double_t *determ)
{
   if (m.GetNrows() != 3) {
      Error("Inv3x3","matrix should be square 3x3");
      return kFALSE;
   }

   Element *pM = m.GetMatrixArray();

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

   if ( det == 0 || tmp == 0) {
      Error("Inv3x3","matrix is singular");
      return kFALSE;
   }

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

// SFij are indices for a 4x4 symmetric matrix.

#define SF00 0
#define SF01 1
#define SF02 2
#define SF03 3

#define SF10 1
#define SF11 5
#define SF12 6
#define SF13 7

#define SF20 2
#define SF21 6
#define SF22 10
#define SF23 11

#define SF30 3
#define SF31 7
#define SF32 11
#define SF33 15

//______________________________________________________________________________
template<class Element>
Bool_t TMatrixTSymCramerInv::Inv4x4(TMatrixTSym<Element> &m,Double_t *determ)
{
   if (m.GetNrows() != 4) {
      Error("Inv4x4","matrix should be square 4x4");
      return kFALSE;
   }

   Element *pM = m.GetMatrixArray();

   // Find all NECESSARY 2x2 dets:  (14 of them)

   const Double_t mDet2_12_01 = pM[SF10]*pM[SF21] - pM[SF11]*pM[SF20];
   const Double_t mDet2_12_02 = pM[SF10]*pM[SF22] - pM[SF12]*pM[SF20];
   const Double_t mDet2_12_12 = pM[SF11]*pM[SF22] - pM[SF12]*pM[SF21];
   const Double_t mDet2_13_01 = pM[SF10]*pM[SF31] - pM[SF11]*pM[SF30];
   const Double_t mDet2_13_02 = pM[SF10]*pM[SF32] - pM[SF12]*pM[SF30];
   const Double_t mDet2_13_03 = pM[SF10]*pM[SF33] - pM[SF13]*pM[SF30];
   const Double_t mDet2_13_12 = pM[SF11]*pM[SF32] - pM[SF12]*pM[SF31];
   const Double_t mDet2_13_13 = pM[SF11]*pM[SF33] - pM[SF13]*pM[SF31];
   const Double_t mDet2_23_01 = pM[SF20]*pM[SF31] - pM[SF21]*pM[SF30];
   const Double_t mDet2_23_02 = pM[SF20]*pM[SF32] - pM[SF22]*pM[SF30];
   const Double_t mDet2_23_03 = pM[SF20]*pM[SF33] - pM[SF23]*pM[SF30];
   const Double_t mDet2_23_12 = pM[SF21]*pM[SF32] - pM[SF22]*pM[SF31];
   const Double_t mDet2_23_13 = pM[SF21]*pM[SF33] - pM[SF23]*pM[SF31];
   const Double_t mDet2_23_23 = pM[SF22]*pM[SF33] - pM[SF23]*pM[SF32];

  // SFind all NECESSSFRY 3x3 dets:   (10 of them)

   const Double_t mDet3_012_012 = pM[SF00]*mDet2_12_12 - pM[SF01]*mDet2_12_02
                                + pM[SF02]*mDet2_12_01;
   const Double_t mDet3_013_012 = pM[SF00]*mDet2_13_12 - pM[SF01]*mDet2_13_02
                                + pM[SF02]*mDet2_13_01;
   const Double_t mDet3_013_013 = pM[SF00]*mDet2_13_13 - pM[SF01]*mDet2_13_03
                                + pM[SF03]*mDet2_13_01;
   const Double_t mDet3_023_012 = pM[SF00]*mDet2_23_12 - pM[SF01]*mDet2_23_02
                                + pM[SF02]*mDet2_23_01;
   const Double_t mDet3_023_013 = pM[SF00]*mDet2_23_13 - pM[SF01]*mDet2_23_03
                                + pM[SF03]*mDet2_23_01;
   const Double_t mDet3_023_023 = pM[SF00]*mDet2_23_23 - pM[SF02]*mDet2_23_03
                                + pM[SF03]*mDet2_23_02;
   const Double_t mDet3_123_012 = pM[SF10]*mDet2_23_12 - pM[SF11]*mDet2_23_02
                                + pM[SF12]*mDet2_23_01;
   const Double_t mDet3_123_013 = pM[SF10]*mDet2_23_13 - pM[SF11]*mDet2_23_03
                                + pM[SF13]*mDet2_23_01;
   const Double_t mDet3_123_023 = pM[SF10]*mDet2_23_23 - pM[SF12]*mDet2_23_03
                                + pM[SF13]*mDet2_23_02;
   const Double_t mDet3_123_123 = pM[SF11]*mDet2_23_23 - pM[SF12]*mDet2_23_13
                                + pM[SF13]*mDet2_23_12;

   // Find the 4x4 det:

   const Double_t det = pM[SF00]*mDet3_123_123 - pM[SF01]*mDet3_123_023
                      + pM[SF02]*mDet3_123_013 - pM[SF03]*mDet3_123_012;

   if (determ)
      *determ = det;

   if ( det == 0 ) {
      Error("Inv4x4","matrix is singular");
      return kFALSE;
   }

   const Double_t oneOverDet = 1.0/det;
   const Double_t mn1OverDet = - oneOverDet;

   pM[SF00] =  mDet3_123_123 * oneOverDet;
   pM[SF01] =  mDet3_123_023 * mn1OverDet;
   pM[SF02] =  mDet3_123_013 * oneOverDet;
   pM[SF03] =  mDet3_123_012 * mn1OverDet;

   pM[SF11] =  mDet3_023_023 * oneOverDet;
   pM[SF12] =  mDet3_023_013 * mn1OverDet;
   pM[SF13] =  mDet3_023_012 * oneOverDet;

   pM[SF22] =  mDet3_013_013 * oneOverDet;
   pM[SF23] =  mDet3_013_012 * mn1OverDet;

   pM[SF33] =  mDet3_012_012 * oneOverDet;

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

#define SM00 0
#define SM01 1
#define SM02 2
#define SM03 3
#define SM04 4

#define SM10 1
#define SM11 6
#define SM12 7
#define SM13 8
#define SM14 9

#define SM20 2
#define SM21 7
#define SM22 12
#define SM23 13
#define SM24 14

#define SM30 3
#define SM31 8
#define SM32 13
#define SM33 18
#define SM34 19

#define SM40 4
#define SM41 9
#define SM42 14
#define SM43 19
#define SM44 24

//______________________________________________________________________________
template<class Element>
Bool_t TMatrixTSymCramerInv::Inv5x5(TMatrixTSym<Element> &m,Double_t *determ)
{
   if (m.GetNrows() != 5) {
      Error("Inv5x5","matrix should be square 5x5");
      return kFALSE;
   }

   Element *pM = m.GetMatrixArray();

   // Find all NECESSARY 2x2 dets:  (25 of them)

   const Double_t mDet2_23_01 = pM[SM20]*pM[SM31] - pM[SM21]*pM[SM30];
   const Double_t mDet2_23_02 = pM[SM20]*pM[SM32] - pM[SM22]*pM[SM30];
   const Double_t mDet2_23_03 = pM[SM20]*pM[SM33] - pM[SM23]*pM[SM30];
   const Double_t mDet2_23_12 = pM[SM21]*pM[SM32] - pM[SM22]*pM[SM31];
   const Double_t mDet2_23_13 = pM[SM21]*pM[SM33] - pM[SM23]*pM[SM31];
   const Double_t mDet2_23_23 = pM[SM22]*pM[SM33] - pM[SM23]*pM[SM32];
   const Double_t mDet2_24_01 = pM[SM20]*pM[SM41] - pM[SM21]*pM[SM40];
   const Double_t mDet2_24_02 = pM[SM20]*pM[SM42] - pM[SM22]*pM[SM40];
   const Double_t mDet2_24_03 = pM[SM20]*pM[SM43] - pM[SM23]*pM[SM40];
   const Double_t mDet2_24_04 = pM[SM20]*pM[SM44] - pM[SM24]*pM[SM40];
   const Double_t mDet2_24_12 = pM[SM21]*pM[SM42] - pM[SM22]*pM[SM41];
   const Double_t mDet2_24_13 = pM[SM21]*pM[SM43] - pM[SM23]*pM[SM41];
   const Double_t mDet2_24_14 = pM[SM21]*pM[SM44] - pM[SM24]*pM[SM41];
   const Double_t mDet2_24_23 = pM[SM22]*pM[SM43] - pM[SM23]*pM[SM42];
   const Double_t mDet2_24_24 = pM[SM22]*pM[SM44] - pM[SM24]*pM[SM42];
   const Double_t mDet2_34_01 = pM[SM30]*pM[SM41] - pM[SM31]*pM[SM40];
   const Double_t mDet2_34_02 = pM[SM30]*pM[SM42] - pM[SM32]*pM[SM40];
   const Double_t mDet2_34_03 = pM[SM30]*pM[SM43] - pM[SM33]*pM[SM40];
   const Double_t mDet2_34_04 = pM[SM30]*pM[SM44] - pM[SM34]*pM[SM40];
   const Double_t mDet2_34_12 = pM[SM31]*pM[SM42] - pM[SM32]*pM[SM41];
   const Double_t mDet2_34_13 = pM[SM31]*pM[SM43] - pM[SM33]*pM[SM41];
   const Double_t mDet2_34_14 = pM[SM31]*pM[SM44] - pM[SM34]*pM[SM41];
   const Double_t mDet2_34_23 = pM[SM32]*pM[SM43] - pM[SM33]*pM[SM42];
   const Double_t mDet2_34_24 = pM[SM32]*pM[SM44] - pM[SM34]*pM[SM42];
   const Double_t mDet2_34_34 = pM[SM33]*pM[SM44] - pM[SM34]*pM[SM43];

   // Find all NECESSARY 3x3 dets:   (30 of them)

   const Double_t mDet3_123_012 = pM[SM10]*mDet2_23_12 - pM[SM11]*mDet2_23_02 + pM[SM12]*mDet2_23_01;
   const Double_t mDet3_123_013 = pM[SM10]*mDet2_23_13 - pM[SM11]*mDet2_23_03 + pM[SM13]*mDet2_23_01;
   const Double_t mDet3_123_023 = pM[SM10]*mDet2_23_23 - pM[SM12]*mDet2_23_03 + pM[SM13]*mDet2_23_02;
   const Double_t mDet3_123_123 = pM[SM11]*mDet2_23_23 - pM[SM12]*mDet2_23_13 + pM[SM13]*mDet2_23_12;
   const Double_t mDet3_124_012 = pM[SM10]*mDet2_24_12 - pM[SM11]*mDet2_24_02 + pM[SM12]*mDet2_24_01;
   const Double_t mDet3_124_013 = pM[SM10]*mDet2_24_13 - pM[SM11]*mDet2_24_03 + pM[SM13]*mDet2_24_01;
   const Double_t mDet3_124_014 = pM[SM10]*mDet2_24_14 - pM[SM11]*mDet2_24_04 + pM[SM14]*mDet2_24_01;
   const Double_t mDet3_124_023 = pM[SM10]*mDet2_24_23 - pM[SM12]*mDet2_24_03 + pM[SM13]*mDet2_24_02;
   const Double_t mDet3_124_024 = pM[SM10]*mDet2_24_24 - pM[SM12]*mDet2_24_04 + pM[SM14]*mDet2_24_02;
   const Double_t mDet3_124_123 = pM[SM11]*mDet2_24_23 - pM[SM12]*mDet2_24_13 + pM[SM13]*mDet2_24_12;
   const Double_t mDet3_124_124 = pM[SM11]*mDet2_24_24 - pM[SM12]*mDet2_24_14 + pM[SM14]*mDet2_24_12;
   const Double_t mDet3_134_012 = pM[SM10]*mDet2_34_12 - pM[SM11]*mDet2_34_02 + pM[SM12]*mDet2_34_01;
   const Double_t mDet3_134_013 = pM[SM10]*mDet2_34_13 - pM[SM11]*mDet2_34_03 + pM[SM13]*mDet2_34_01;
   const Double_t mDet3_134_014 = pM[SM10]*mDet2_34_14 - pM[SM11]*mDet2_34_04 + pM[SM14]*mDet2_34_01;
   const Double_t mDet3_134_023 = pM[SM10]*mDet2_34_23 - pM[SM12]*mDet2_34_03 + pM[SM13]*mDet2_34_02;
   const Double_t mDet3_134_024 = pM[SM10]*mDet2_34_24 - pM[SM12]*mDet2_34_04 + pM[SM14]*mDet2_34_02;
   const Double_t mDet3_134_034 = pM[SM10]*mDet2_34_34 - pM[SM13]*mDet2_34_04 + pM[SM14]*mDet2_34_03;
   const Double_t mDet3_134_123 = pM[SM11]*mDet2_34_23 - pM[SM12]*mDet2_34_13 + pM[SM13]*mDet2_34_12;
   const Double_t mDet3_134_124 = pM[SM11]*mDet2_34_24 - pM[SM12]*mDet2_34_14 + pM[SM14]*mDet2_34_12;
   const Double_t mDet3_134_134 = pM[SM11]*mDet2_34_34 - pM[SM13]*mDet2_34_14 + pM[SM14]*mDet2_34_13;
   const Double_t mDet3_234_012 = pM[SM20]*mDet2_34_12 - pM[SM21]*mDet2_34_02 + pM[SM22]*mDet2_34_01;
   const Double_t mDet3_234_013 = pM[SM20]*mDet2_34_13 - pM[SM21]*mDet2_34_03 + pM[SM23]*mDet2_34_01;
   const Double_t mDet3_234_014 = pM[SM20]*mDet2_34_14 - pM[SM21]*mDet2_34_04 + pM[SM24]*mDet2_34_01;
   const Double_t mDet3_234_023 = pM[SM20]*mDet2_34_23 - pM[SM22]*mDet2_34_03 + pM[SM23]*mDet2_34_02;
   const Double_t mDet3_234_024 = pM[SM20]*mDet2_34_24 - pM[SM22]*mDet2_34_04 + pM[SM24]*mDet2_34_02;
   const Double_t mDet3_234_034 = pM[SM20]*mDet2_34_34 - pM[SM23]*mDet2_34_04 + pM[SM24]*mDet2_34_03;
   const Double_t mDet3_234_123 = pM[SM21]*mDet2_34_23 - pM[SM22]*mDet2_34_13 + pM[SM23]*mDet2_34_12;
   const Double_t mDet3_234_124 = pM[SM21]*mDet2_34_24 - pM[SM22]*mDet2_34_14 + pM[SM24]*mDet2_34_12;
   const Double_t mDet3_234_134 = pM[SM21]*mDet2_34_34 - pM[SM23]*mDet2_34_14 + pM[SM24]*mDet2_34_13;
   const Double_t mDet3_234_234 = pM[SM22]*mDet2_34_34 - pM[SM23]*mDet2_34_24 + pM[SM24]*mDet2_34_23;

   // Find all NECESSARY 4x4 dets:   (15 of them)

   const Double_t mDet4_0123_0123 = pM[SM00]*mDet3_123_123 - pM[SM01]*mDet3_123_023
                                  + pM[SM02]*mDet3_123_013 - pM[SM03]*mDet3_123_012;
   const Double_t mDet4_0124_0123 = pM[SM00]*mDet3_124_123 - pM[SM01]*mDet3_124_023
                                  + pM[SM02]*mDet3_124_013 - pM[SM03]*mDet3_124_012;
   const Double_t mDet4_0124_0124 = pM[SM00]*mDet3_124_124 - pM[SM01]*mDet3_124_024
                                  + pM[SM02]*mDet3_124_014 - pM[SM04]*mDet3_124_012;
   const Double_t mDet4_0134_0123 = pM[SM00]*mDet3_134_123 - pM[SM01]*mDet3_134_023
                                  + pM[SM02]*mDet3_134_013 - pM[SM03]*mDet3_134_012;
   const Double_t mDet4_0134_0124 = pM[SM00]*mDet3_134_124 - pM[SM01]*mDet3_134_024
                                  + pM[SM02]*mDet3_134_014 - pM[SM04]*mDet3_134_012;
   const Double_t mDet4_0134_0134 = pM[SM00]*mDet3_134_134 - pM[SM01]*mDet3_134_034
                                  + pM[SM03]*mDet3_134_014 - pM[SM04]*mDet3_134_013;
   const Double_t mDet4_0234_0123 = pM[SM00]*mDet3_234_123 - pM[SM01]*mDet3_234_023
                                  + pM[SM02]*mDet3_234_013 - pM[SM03]*mDet3_234_012;
   const Double_t mDet4_0234_0124 = pM[SM00]*mDet3_234_124 - pM[SM01]*mDet3_234_024
                                  + pM[SM02]*mDet3_234_014 - pM[SM04]*mDet3_234_012;
   const Double_t mDet4_0234_0134 = pM[SM00]*mDet3_234_134 - pM[SM01]*mDet3_234_034
                                  + pM[SM03]*mDet3_234_014 - pM[SM04]*mDet3_234_013;
   const Double_t mDet4_0234_0234 = pM[SM00]*mDet3_234_234 - pM[SM02]*mDet3_234_034
                                  + pM[SM03]*mDet3_234_024 - pM[SM04]*mDet3_234_023;
   const Double_t mDet4_1234_0123 = pM[SM10]*mDet3_234_123 - pM[SM11]*mDet3_234_023
                                  + pM[SM12]*mDet3_234_013 - pM[SM13]*mDet3_234_012;
   const Double_t mDet4_1234_0124 = pM[SM10]*mDet3_234_124 - pM[SM11]*mDet3_234_024
                                  + pM[SM12]*mDet3_234_014 - pM[SM14]*mDet3_234_012;
   const Double_t mDet4_1234_0134 = pM[SM10]*mDet3_234_134 - pM[SM11]*mDet3_234_034
                                  + pM[SM13]*mDet3_234_014 - pM[SM14]*mDet3_234_013;
   const Double_t mDet4_1234_0234 = pM[SM10]*mDet3_234_234 - pM[SM12]*mDet3_234_034
                                  + pM[SM13]*mDet3_234_024 - pM[SM14]*mDet3_234_023;
   const Double_t mDet4_1234_1234 = pM[SM11]*mDet3_234_234 - pM[SM12]*mDet3_234_134
                                  + pM[SM13]*mDet3_234_124 - pM[SM14]*mDet3_234_123;

   // Find the 5x5 det:

   const Double_t det = pM[SM00]*mDet4_1234_1234 - pM[SM01]*mDet4_1234_0234 + pM[SM02]*mDet4_1234_0134
                      - pM[SM03]*mDet4_1234_0124 + pM[SM04]*mDet4_1234_0123;
   if (determ)
      *determ = det;

   if ( det == 0 ) {
      Error("Inv5x5","matrix is singular");
      return kFALSE;
   }

   const Double_t oneOverDet = 1.0/det;
   const Double_t mn1OverDet = - oneOverDet;

   pM[SM00] = mDet4_1234_1234 * oneOverDet;
   pM[SM01] = mDet4_1234_0234 * mn1OverDet;
   pM[SM02] = mDet4_1234_0134 * oneOverDet;
   pM[SM03] = mDet4_1234_0124 * mn1OverDet;
   pM[SM04] = mDet4_1234_0123 * oneOverDet;

   pM[SM11] = mDet4_0234_0234 * oneOverDet;
   pM[SM12] = mDet4_0234_0134 * mn1OverDet;
   pM[SM13] = mDet4_0234_0124 * oneOverDet;
   pM[SM14] = mDet4_0234_0123 * mn1OverDet;

   pM[SM22] = mDet4_0134_0134 * oneOverDet;
   pM[SM23] = mDet4_0134_0124 * mn1OverDet;
   pM[SM24] = mDet4_0134_0123 * oneOverDet;

   pM[SM33] = mDet4_0124_0124 * oneOverDet;
   pM[SM34] = mDet4_0124_0123 * mn1OverDet;

   pM[SM44] = mDet4_0123_0123 * oneOverDet;

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

#define SA00 0
#define SA01 1
#define SA02 2
#define SA03 3
#define SA04 4
#define SA05 5

#define SA10 1
#define SA11 7
#define SA12 8
#define SA13 9
#define SA14 10
#define SA15 11

#define SA20 2
#define SA21 8
#define SA22 14
#define SA23 15
#define SA24 16
#define SA25 17

#define SA30 3
#define SA31 9
#define SA32 15
#define SA33 21
#define SA34 22
#define SA35 23

#define SA40 4
#define SA41 10
#define SA42 16
#define SA43 22
#define SA44 28
#define SA45 29

#define SA50 5
#define SA51 11
#define SA52 17
#define SA53 23
#define SA54 29
#define SA55 35

//______________________________________________________________________________
template<class Element>
Bool_t TMatrixTSymCramerInv::Inv6x6(TMatrixTSym<Element> &m,Double_t *determ)
{
   if (m.GetNrows() != 6 || m.GetNcols() != 6 || m.GetRowLwb() != m.GetColLwb()) {
      Error("Inv6x6","matrix should be square 6x6");
      return kFALSE;
   }

   Element *pM = m.GetMatrixArray();

   // Find all NECESSSARY 2x2 dets:  (39 of them)

   const Double_t mDet2_34_01 = pM[SA30]*pM[SA41] - pM[SA31]*pM[SA40];
   const Double_t mDet2_34_02 = pM[SA30]*pM[SA42] - pM[SA32]*pM[SA40];
   const Double_t mDet2_34_03 = pM[SA30]*pM[SA43] - pM[SA33]*pM[SA40];
   const Double_t mDet2_34_04 = pM[SA30]*pM[SA44] - pM[SA34]*pM[SA40];
   const Double_t mDet2_34_12 = pM[SA31]*pM[SA42] - pM[SA32]*pM[SA41];
   const Double_t mDet2_34_13 = pM[SA31]*pM[SA43] - pM[SA33]*pM[SA41];
   const Double_t mDet2_34_14 = pM[SA31]*pM[SA44] - pM[SA34]*pM[SA41];
   const Double_t mDet2_34_23 = pM[SA32]*pM[SA43] - pM[SA33]*pM[SA42];
   const Double_t mDet2_34_24 = pM[SA32]*pM[SA44] - pM[SA34]*pM[SA42];
   const Double_t mDet2_34_34 = pM[SA33]*pM[SA44] - pM[SA34]*pM[SA43];
   const Double_t mDet2_35_01 = pM[SA30]*pM[SA51] - pM[SA31]*pM[SA50];
   const Double_t mDet2_35_02 = pM[SA30]*pM[SA52] - pM[SA32]*pM[SA50];
   const Double_t mDet2_35_03 = pM[SA30]*pM[SA53] - pM[SA33]*pM[SA50];
   const Double_t mDet2_35_04 = pM[SA30]*pM[SA54] - pM[SA34]*pM[SA50];
   const Double_t mDet2_35_05 = pM[SA30]*pM[SA55] - pM[SA35]*pM[SA50];
   const Double_t mDet2_35_12 = pM[SA31]*pM[SA52] - pM[SA32]*pM[SA51];
   const Double_t mDet2_35_13 = pM[SA31]*pM[SA53] - pM[SA33]*pM[SA51];
   const Double_t mDet2_35_14 = pM[SA31]*pM[SA54] - pM[SA34]*pM[SA51];
   const Double_t mDet2_35_15 = pM[SA31]*pM[SA55] - pM[SA35]*pM[SA51];
   const Double_t mDet2_35_23 = pM[SA32]*pM[SA53] - pM[SA33]*pM[SA52];
   const Double_t mDet2_35_24 = pM[SA32]*pM[SA54] - pM[SA34]*pM[SA52];
   const Double_t mDet2_35_25 = pM[SA32]*pM[SA55] - pM[SA35]*pM[SA52];
   const Double_t mDet2_35_34 = pM[SA33]*pM[SA54] - pM[SA34]*pM[SA53];
   const Double_t mDet2_35_35 = pM[SA33]*pM[SA55] - pM[SA35]*pM[SA53];
   const Double_t mDet2_45_01 = pM[SA40]*pM[SA51] - pM[SA41]*pM[SA50];
   const Double_t mDet2_45_02 = pM[SA40]*pM[SA52] - pM[SA42]*pM[SA50];
   const Double_t mDet2_45_03 = pM[SA40]*pM[SA53] - pM[SA43]*pM[SA50];
   const Double_t mDet2_45_04 = pM[SA40]*pM[SA54] - pM[SA44]*pM[SA50];
   const Double_t mDet2_45_05 = pM[SA40]*pM[SA55] - pM[SA45]*pM[SA50];
   const Double_t mDet2_45_12 = pM[SA41]*pM[SA52] - pM[SA42]*pM[SA51];
   const Double_t mDet2_45_13 = pM[SA41]*pM[SA53] - pM[SA43]*pM[SA51];
   const Double_t mDet2_45_14 = pM[SA41]*pM[SA54] - pM[SA44]*pM[SA51];
   const Double_t mDet2_45_15 = pM[SA41]*pM[SA55] - pM[SA45]*pM[SA51];
   const Double_t mDet2_45_23 = pM[SA42]*pM[SA53] - pM[SA43]*pM[SA52];
   const Double_t mDet2_45_24 = pM[SA42]*pM[SA54] - pM[SA44]*pM[SA52];
   const Double_t mDet2_45_25 = pM[SA42]*pM[SA55] - pM[SA45]*pM[SA52];
   const Double_t mDet2_45_34 = pM[SA43]*pM[SA54] - pM[SA44]*pM[SA53];
   const Double_t mDet2_45_35 = pM[SA43]*pM[SA55] - pM[SA45]*pM[SA53];
   const Double_t mDet2_45_45 = pM[SA44]*pM[SA55] - pM[SA45]*pM[SA54];

   // Find all NECESSSARY 3x3 dets:  (65 of them)

   const Double_t mDet3_234_012 = pM[SA20]*mDet2_34_12 - pM[SA21]*mDet2_34_02 + pM[SA22]*mDet2_34_01;
   const Double_t mDet3_234_013 = pM[SA20]*mDet2_34_13 - pM[SA21]*mDet2_34_03 + pM[SA23]*mDet2_34_01;
   const Double_t mDet3_234_014 = pM[SA20]*mDet2_34_14 - pM[SA21]*mDet2_34_04 + pM[SA24]*mDet2_34_01;
   const Double_t mDet3_234_023 = pM[SA20]*mDet2_34_23 - pM[SA22]*mDet2_34_03 + pM[SA23]*mDet2_34_02;
   const Double_t mDet3_234_024 = pM[SA20]*mDet2_34_24 - pM[SA22]*mDet2_34_04 + pM[SA24]*mDet2_34_02;
   const Double_t mDet3_234_034 = pM[SA20]*mDet2_34_34 - pM[SA23]*mDet2_34_04 + pM[SA24]*mDet2_34_03;
   const Double_t mDet3_234_123 = pM[SA21]*mDet2_34_23 - pM[SA22]*mDet2_34_13 + pM[SA23]*mDet2_34_12;
   const Double_t mDet3_234_124 = pM[SA21]*mDet2_34_24 - pM[SA22]*mDet2_34_14 + pM[SA24]*mDet2_34_12;
   const Double_t mDet3_234_134 = pM[SA21]*mDet2_34_34 - pM[SA23]*mDet2_34_14 + pM[SA24]*mDet2_34_13;
   const Double_t mDet3_234_234 = pM[SA22]*mDet2_34_34 - pM[SA23]*mDet2_34_24 + pM[SA24]*mDet2_34_23;
   const Double_t mDet3_235_012 = pM[SA20]*mDet2_35_12 - pM[SA21]*mDet2_35_02 + pM[SA22]*mDet2_35_01;
   const Double_t mDet3_235_013 = pM[SA20]*mDet2_35_13 - pM[SA21]*mDet2_35_03 + pM[SA23]*mDet2_35_01;
   const Double_t mDet3_235_014 = pM[SA20]*mDet2_35_14 - pM[SA21]*mDet2_35_04 + pM[SA24]*mDet2_35_01;
   const Double_t mDet3_235_015 = pM[SA20]*mDet2_35_15 - pM[SA21]*mDet2_35_05 + pM[SA25]*mDet2_35_01;
   const Double_t mDet3_235_023 = pM[SA20]*mDet2_35_23 - pM[SA22]*mDet2_35_03 + pM[SA23]*mDet2_35_02;
   const Double_t mDet3_235_024 = pM[SA20]*mDet2_35_24 - pM[SA22]*mDet2_35_04 + pM[SA24]*mDet2_35_02;
   const Double_t mDet3_235_025 = pM[SA20]*mDet2_35_25 - pM[SA22]*mDet2_35_05 + pM[SA25]*mDet2_35_02;
   const Double_t mDet3_235_034 = pM[SA20]*mDet2_35_34 - pM[SA23]*mDet2_35_04 + pM[SA24]*mDet2_35_03;
   const Double_t mDet3_235_035 = pM[SA20]*mDet2_35_35 - pM[SA23]*mDet2_35_05 + pM[SA25]*mDet2_35_03;
   const Double_t mDet3_235_123 = pM[SA21]*mDet2_35_23 - pM[SA22]*mDet2_35_13 + pM[SA23]*mDet2_35_12;
   const Double_t mDet3_235_124 = pM[SA21]*mDet2_35_24 - pM[SA22]*mDet2_35_14 + pM[SA24]*mDet2_35_12;
   const Double_t mDet3_235_125 = pM[SA21]*mDet2_35_25 - pM[SA22]*mDet2_35_15 + pM[SA25]*mDet2_35_12;
   const Double_t mDet3_235_134 = pM[SA21]*mDet2_35_34 - pM[SA23]*mDet2_35_14 + pM[SA24]*mDet2_35_13;
   const Double_t mDet3_235_135 = pM[SA21]*mDet2_35_35 - pM[SA23]*mDet2_35_15 + pM[SA25]*mDet2_35_13;
   const Double_t mDet3_235_234 = pM[SA22]*mDet2_35_34 - pM[SA23]*mDet2_35_24 + pM[SA24]*mDet2_35_23;
   const Double_t mDet3_235_235 = pM[SA22]*mDet2_35_35 - pM[SA23]*mDet2_35_25 + pM[SA25]*mDet2_35_23;
   const Double_t mDet3_245_012 = pM[SA20]*mDet2_45_12 - pM[SA21]*mDet2_45_02 + pM[SA22]*mDet2_45_01;
   const Double_t mDet3_245_013 = pM[SA20]*mDet2_45_13 - pM[SA21]*mDet2_45_03 + pM[SA23]*mDet2_45_01;
   const Double_t mDet3_245_014 = pM[SA20]*mDet2_45_14 - pM[SA21]*mDet2_45_04 + pM[SA24]*mDet2_45_01;
   const Double_t mDet3_245_015 = pM[SA20]*mDet2_45_15 - pM[SA21]*mDet2_45_05 + pM[SA25]*mDet2_45_01;
   const Double_t mDet3_245_023 = pM[SA20]*mDet2_45_23 - pM[SA22]*mDet2_45_03 + pM[SA23]*mDet2_45_02;
   const Double_t mDet3_245_024 = pM[SA20]*mDet2_45_24 - pM[SA22]*mDet2_45_04 + pM[SA24]*mDet2_45_02;
   const Double_t mDet3_245_025 = pM[SA20]*mDet2_45_25 - pM[SA22]*mDet2_45_05 + pM[SA25]*mDet2_45_02;
   const Double_t mDet3_245_034 = pM[SA20]*mDet2_45_34 - pM[SA23]*mDet2_45_04 + pM[SA24]*mDet2_45_03;
   const Double_t mDet3_245_035 = pM[SA20]*mDet2_45_35 - pM[SA23]*mDet2_45_05 + pM[SA25]*mDet2_45_03;
   const Double_t mDet3_245_045 = pM[SA20]*mDet2_45_45 - pM[SA24]*mDet2_45_05 + pM[SA25]*mDet2_45_04;
   const Double_t mDet3_245_123 = pM[SA21]*mDet2_45_23 - pM[SA22]*mDet2_45_13 + pM[SA23]*mDet2_45_12;
   const Double_t mDet3_245_124 = pM[SA21]*mDet2_45_24 - pM[SA22]*mDet2_45_14 + pM[SA24]*mDet2_45_12;
   const Double_t mDet3_245_125 = pM[SA21]*mDet2_45_25 - pM[SA22]*mDet2_45_15 + pM[SA25]*mDet2_45_12;
   const Double_t mDet3_245_134 = pM[SA21]*mDet2_45_34 - pM[SA23]*mDet2_45_14 + pM[SA24]*mDet2_45_13;
   const Double_t mDet3_245_135 = pM[SA21]*mDet2_45_35 - pM[SA23]*mDet2_45_15 + pM[SA25]*mDet2_45_13;
   const Double_t mDet3_245_145 = pM[SA21]*mDet2_45_45 - pM[SA24]*mDet2_45_15 + pM[SA25]*mDet2_45_14;
   const Double_t mDet3_245_234 = pM[SA22]*mDet2_45_34 - pM[SA23]*mDet2_45_24 + pM[SA24]*mDet2_45_23;
   const Double_t mDet3_245_235 = pM[SA22]*mDet2_45_35 - pM[SA23]*mDet2_45_25 + pM[SA25]*mDet2_45_23;
   const Double_t mDet3_245_245 = pM[SA22]*mDet2_45_45 - pM[SA24]*mDet2_45_25 + pM[SA25]*mDet2_45_24;
   const Double_t mDet3_345_012 = pM[SA30]*mDet2_45_12 - pM[SA31]*mDet2_45_02 + pM[SA32]*mDet2_45_01;
   const Double_t mDet3_345_013 = pM[SA30]*mDet2_45_13 - pM[SA31]*mDet2_45_03 + pM[SA33]*mDet2_45_01;
   const Double_t mDet3_345_014 = pM[SA30]*mDet2_45_14 - pM[SA31]*mDet2_45_04 + pM[SA34]*mDet2_45_01;
   const Double_t mDet3_345_015 = pM[SA30]*mDet2_45_15 - pM[SA31]*mDet2_45_05 + pM[SA35]*mDet2_45_01;
   const Double_t mDet3_345_023 = pM[SA30]*mDet2_45_23 - pM[SA32]*mDet2_45_03 + pM[SA33]*mDet2_45_02;
   const Double_t mDet3_345_024 = pM[SA30]*mDet2_45_24 - pM[SA32]*mDet2_45_04 + pM[SA34]*mDet2_45_02;
   const Double_t mDet3_345_025 = pM[SA30]*mDet2_45_25 - pM[SA32]*mDet2_45_05 + pM[SA35]*mDet2_45_02;
   const Double_t mDet3_345_034 = pM[SA30]*mDet2_45_34 - pM[SA33]*mDet2_45_04 + pM[SA34]*mDet2_45_03;
   const Double_t mDet3_345_035 = pM[SA30]*mDet2_45_35 - pM[SA33]*mDet2_45_05 + pM[SA35]*mDet2_45_03;
   const Double_t mDet3_345_045 = pM[SA30]*mDet2_45_45 - pM[SA34]*mDet2_45_05 + pM[SA35]*mDet2_45_04;
   const Double_t mDet3_345_123 = pM[SA31]*mDet2_45_23 - pM[SA32]*mDet2_45_13 + pM[SA33]*mDet2_45_12;
   const Double_t mDet3_345_124 = pM[SA31]*mDet2_45_24 - pM[SA32]*mDet2_45_14 + pM[SA34]*mDet2_45_12;
   const Double_t mDet3_345_125 = pM[SA31]*mDet2_45_25 - pM[SA32]*mDet2_45_15 + pM[SA35]*mDet2_45_12;
   const Double_t mDet3_345_134 = pM[SA31]*mDet2_45_34 - pM[SA33]*mDet2_45_14 + pM[SA34]*mDet2_45_13;
   const Double_t mDet3_345_135 = pM[SA31]*mDet2_45_35 - pM[SA33]*mDet2_45_15 + pM[SA35]*mDet2_45_13;
   const Double_t mDet3_345_145 = pM[SA31]*mDet2_45_45 - pM[SA34]*mDet2_45_15 + pM[SA35]*mDet2_45_14;
   const Double_t mDet3_345_234 = pM[SA32]*mDet2_45_34 - pM[SA33]*mDet2_45_24 + pM[SA34]*mDet2_45_23;
   const Double_t mDet3_345_235 = pM[SA32]*mDet2_45_35 - pM[SA33]*mDet2_45_25 + pM[SA35]*mDet2_45_23;
   const Double_t mDet3_345_245 = pM[SA32]*mDet2_45_45 - pM[SA34]*mDet2_45_25 + pM[SA35]*mDet2_45_24;
   const Double_t mDet3_345_345 = pM[SA33]*mDet2_45_45 - pM[SA34]*mDet2_45_35 + pM[SA35]*mDet2_45_34;

   // Find all NECESSSARY 4x4 dets:  (55 of them)

   const Double_t mDet4_1234_0123 = pM[SA10]*mDet3_234_123 - pM[SA11]*mDet3_234_023
                                  + pM[SA12]*mDet3_234_013 - pM[SA13]*mDet3_234_012;
   const Double_t mDet4_1234_0124 = pM[SA10]*mDet3_234_124 - pM[SA11]*mDet3_234_024
                                  + pM[SA12]*mDet3_234_014 - pM[SA14]*mDet3_234_012;
   const Double_t mDet4_1234_0134 = pM[SA10]*mDet3_234_134 - pM[SA11]*mDet3_234_034
                                  + pM[SA13]*mDet3_234_014 - pM[SA14]*mDet3_234_013;
   const Double_t mDet4_1234_0234 = pM[SA10]*mDet3_234_234 - pM[SA12]*mDet3_234_034
                                  + pM[SA13]*mDet3_234_024 - pM[SA14]*mDet3_234_023;
   const Double_t mDet4_1234_1234 = pM[SA11]*mDet3_234_234 - pM[SA12]*mDet3_234_134
                                  + pM[SA13]*mDet3_234_124 - pM[SA14]*mDet3_234_123;
   const Double_t mDet4_1235_0123 = pM[SA10]*mDet3_235_123 - pM[SA11]*mDet3_235_023
                                  + pM[SA12]*mDet3_235_013 - pM[SA13]*mDet3_235_012;
   const Double_t mDet4_1235_0124 = pM[SA10]*mDet3_235_124 - pM[SA11]*mDet3_235_024
                                  + pM[SA12]*mDet3_235_014 - pM[SA14]*mDet3_235_012;
   const Double_t mDet4_1235_0125 = pM[SA10]*mDet3_235_125 - pM[SA11]*mDet3_235_025
                                  + pM[SA12]*mDet3_235_015 - pM[SA15]*mDet3_235_012;
   const Double_t mDet4_1235_0134 = pM[SA10]*mDet3_235_134 - pM[SA11]*mDet3_235_034
                                  + pM[SA13]*mDet3_235_014 - pM[SA14]*mDet3_235_013;
   const Double_t mDet4_1235_0135 = pM[SA10]*mDet3_235_135 - pM[SA11]*mDet3_235_035
                                  + pM[SA13]*mDet3_235_015 - pM[SA15]*mDet3_235_013;
   const Double_t mDet4_1235_0234 = pM[SA10]*mDet3_235_234 - pM[SA12]*mDet3_235_034
                                  + pM[SA13]*mDet3_235_024 - pM[SA14]*mDet3_235_023;
   const Double_t mDet4_1235_0235 = pM[SA10]*mDet3_235_235 - pM[SA12]*mDet3_235_035
                                  + pM[SA13]*mDet3_235_025 - pM[SA15]*mDet3_235_023;
   const Double_t mDet4_1235_1234 = pM[SA11]*mDet3_235_234 - pM[SA12]*mDet3_235_134
                                  + pM[SA13]*mDet3_235_124 - pM[SA14]*mDet3_235_123;
   const Double_t mDet4_1235_1235 = pM[SA11]*mDet3_235_235 - pM[SA12]*mDet3_235_135
                                  + pM[SA13]*mDet3_235_125 - pM[SA15]*mDet3_235_123;
   const Double_t mDet4_1245_0123 = pM[SA10]*mDet3_245_123 - pM[SA11]*mDet3_245_023
                                  + pM[SA12]*mDet3_245_013 - pM[SA13]*mDet3_245_012;
   const Double_t mDet4_1245_0124 = pM[SA10]*mDet3_245_124 - pM[SA11]*mDet3_245_024
                                  + pM[SA12]*mDet3_245_014 - pM[SA14]*mDet3_245_012;
   const Double_t mDet4_1245_0125 = pM[SA10]*mDet3_245_125 - pM[SA11]*mDet3_245_025
                                  + pM[SA12]*mDet3_245_015 - pM[SA15]*mDet3_245_012;
   const Double_t mDet4_1245_0134 = pM[SA10]*mDet3_245_134 - pM[SA11]*mDet3_245_034
                                  + pM[SA13]*mDet3_245_014 - pM[SA14]*mDet3_245_013;
   const Double_t mDet4_1245_0135 = pM[SA10]*mDet3_245_135 - pM[SA11]*mDet3_245_035
                                  + pM[SA13]*mDet3_245_015 - pM[SA15]*mDet3_245_013;
   const Double_t mDet4_1245_0145 = pM[SA10]*mDet3_245_145 - pM[SA11]*mDet3_245_045
                                  + pM[SA14]*mDet3_245_015 - pM[SA15]*mDet3_245_014;
   const Double_t mDet4_1245_0234 = pM[SA10]*mDet3_245_234 - pM[SA12]*mDet3_245_034
                                  + pM[SA13]*mDet3_245_024 - pM[SA14]*mDet3_245_023;
   const Double_t mDet4_1245_0235 = pM[SA10]*mDet3_245_235 - pM[SA12]*mDet3_245_035
                                  + pM[SA13]*mDet3_245_025 - pM[SA15]*mDet3_245_023;
   const Double_t mDet4_1245_0245 = pM[SA10]*mDet3_245_245 - pM[SA12]*mDet3_245_045
                                  + pM[SA14]*mDet3_245_025 - pM[SA15]*mDet3_245_024;
   const Double_t mDet4_1245_1234 = pM[SA11]*mDet3_245_234 - pM[SA12]*mDet3_245_134
                                  + pM[SA13]*mDet3_245_124 - pM[SA14]*mDet3_245_123;
   const Double_t mDet4_1245_1235 = pM[SA11]*mDet3_245_235 - pM[SA12]*mDet3_245_135
                                  + pM[SA13]*mDet3_245_125 - pM[SA15]*mDet3_245_123;
   const Double_t mDet4_1245_1245 = pM[SA11]*mDet3_245_245 - pM[SA12]*mDet3_245_145
                                  + pM[SA14]*mDet3_245_125 - pM[SA15]*mDet3_245_124;
   const Double_t mDet4_1345_0123 = pM[SA10]*mDet3_345_123 - pM[SA11]*mDet3_345_023
                                  + pM[SA12]*mDet3_345_013 - pM[SA13]*mDet3_345_012;
   const Double_t mDet4_1345_0124 = pM[SA10]*mDet3_345_124 - pM[SA11]*mDet3_345_024
                                  + pM[SA12]*mDet3_345_014 - pM[SA14]*mDet3_345_012;
   const Double_t mDet4_1345_0125 = pM[SA10]*mDet3_345_125 - pM[SA11]*mDet3_345_025
                                  + pM[SA12]*mDet3_345_015 - pM[SA15]*mDet3_345_012;
   const Double_t mDet4_1345_0134 = pM[SA10]*mDet3_345_134 - pM[SA11]*mDet3_345_034
                                  + pM[SA13]*mDet3_345_014 - pM[SA14]*mDet3_345_013;
   const Double_t mDet4_1345_0135 = pM[SA10]*mDet3_345_135 - pM[SA11]*mDet3_345_035
                                  + pM[SA13]*mDet3_345_015 - pM[SA15]*mDet3_345_013;
   const Double_t mDet4_1345_0145 = pM[SA10]*mDet3_345_145 - pM[SA11]*mDet3_345_045
                                  + pM[SA14]*mDet3_345_015 - pM[SA15]*mDet3_345_014;
   const Double_t mDet4_1345_0234 = pM[SA10]*mDet3_345_234 - pM[SA12]*mDet3_345_034
                                  + pM[SA13]*mDet3_345_024 - pM[SA14]*mDet3_345_023;
   const Double_t mDet4_1345_0235 = pM[SA10]*mDet3_345_235 - pM[SA12]*mDet3_345_035
                                  + pM[SA13]*mDet3_345_025 - pM[SA15]*mDet3_345_023;
   const Double_t mDet4_1345_0245 = pM[SA10]*mDet3_345_245 - pM[SA12]*mDet3_345_045
                                  + pM[SA14]*mDet3_345_025 - pM[SA15]*mDet3_345_024;
   const Double_t mDet4_1345_0345 = pM[SA10]*mDet3_345_345 - pM[SA13]*mDet3_345_045
                                  + pM[SA14]*mDet3_345_035 - pM[SA15]*mDet3_345_034;
   const Double_t mDet4_1345_1234 = pM[SA11]*mDet3_345_234 - pM[SA12]*mDet3_345_134
                                  + pM[SA13]*mDet3_345_124 - pM[SA14]*mDet3_345_123;
   const Double_t mDet4_1345_1235 = pM[SA11]*mDet3_345_235 - pM[SA12]*mDet3_345_135
                                  + pM[SA13]*mDet3_345_125 - pM[SA15]*mDet3_345_123;
   const Double_t mDet4_1345_1245 = pM[SA11]*mDet3_345_245 - pM[SA12]*mDet3_345_145
                                  + pM[SA14]*mDet3_345_125 - pM[SA15]*mDet3_345_124;
   const Double_t mDet4_1345_1345 = pM[SA11]*mDet3_345_345 - pM[SA13]*mDet3_345_145
                                  + pM[SA14]*mDet3_345_135 - pM[SA15]*mDet3_345_134;
   const Double_t mDet4_2345_0123 = pM[SA20]*mDet3_345_123 - pM[SA21]*mDet3_345_023
                                  + pM[SA22]*mDet3_345_013 - pM[SA23]*mDet3_345_012;
   const Double_t mDet4_2345_0124 = pM[SA20]*mDet3_345_124 - pM[SA21]*mDet3_345_024
                                  + pM[SA22]*mDet3_345_014 - pM[SA24]*mDet3_345_012;
   const Double_t mDet4_2345_0125 = pM[SA20]*mDet3_345_125 - pM[SA21]*mDet3_345_025
                                  + pM[SA22]*mDet3_345_015 - pM[SA25]*mDet3_345_012;
   const Double_t mDet4_2345_0134 = pM[SA20]*mDet3_345_134 - pM[SA21]*mDet3_345_034
                                  + pM[SA23]*mDet3_345_014 - pM[SA24]*mDet3_345_013;
   const Double_t mDet4_2345_0135 = pM[SA20]*mDet3_345_135 - pM[SA21]*mDet3_345_035
                                  + pM[SA23]*mDet3_345_015 - pM[SA25]*mDet3_345_013;
   const Double_t mDet4_2345_0145 = pM[SA20]*mDet3_345_145 - pM[SA21]*mDet3_345_045
                                  + pM[SA24]*mDet3_345_015 - pM[SA25]*mDet3_345_014;
   const Double_t mDet4_2345_0234 = pM[SA20]*mDet3_345_234 - pM[SA22]*mDet3_345_034
                                  + pM[SA23]*mDet3_345_024 - pM[SA24]*mDet3_345_023;
   const Double_t mDet4_2345_0235 = pM[SA20]*mDet3_345_235 - pM[SA22]*mDet3_345_035
                                  + pM[SA23]*mDet3_345_025 - pM[SA25]*mDet3_345_023;
   const Double_t mDet4_2345_0245 = pM[SA20]*mDet3_345_245 - pM[SA22]*mDet3_345_045
                                  + pM[SA24]*mDet3_345_025 - pM[SA25]*mDet3_345_024;
   const Double_t mDet4_2345_0345 = pM[SA20]*mDet3_345_345 - pM[SA23]*mDet3_345_045
                                  + pM[SA24]*mDet3_345_035 - pM[SA25]*mDet3_345_034;
   const Double_t mDet4_2345_1234 = pM[SA21]*mDet3_345_234 - pM[SA22]*mDet3_345_134
                                  + pM[SA23]*mDet3_345_124 - pM[SA24]*mDet3_345_123;
   const Double_t mDet4_2345_1235 = pM[SA21]*mDet3_345_235 - pM[SA22]*mDet3_345_135
                                  + pM[SA23]*mDet3_345_125 - pM[SA25]*mDet3_345_123;
   const Double_t mDet4_2345_1245 = pM[SA21]*mDet3_345_245 - pM[SA22]*mDet3_345_145
                                  + pM[SA24]*mDet3_345_125 - pM[SA25]*mDet3_345_124;
   const Double_t mDet4_2345_1345 = pM[SA21]*mDet3_345_345 - pM[SA23]*mDet3_345_145
                                  + pM[SA24]*mDet3_345_135 - pM[SA25]*mDet3_345_134;
   const Double_t mDet4_2345_2345 = pM[SA22]*mDet3_345_345 - pM[SA23]*mDet3_345_245
                                  + pM[SA24]*mDet3_345_235 - pM[SA25]*mDet3_345_234;

   // Find all NECESSSARY 5x5 dets:  (19 of them)

   const Double_t mDet5_01234_01234 = pM[SA00]*mDet4_1234_1234 - pM[SA01]*mDet4_1234_0234
                                    + pM[SA02]*mDet4_1234_0134 - pM[SA03]*mDet4_1234_0124 + pM[SA04]*mDet4_1234_0123;
   const Double_t mDet5_01235_01234 = pM[SA00]*mDet4_1235_1234 - pM[SA01]*mDet4_1235_0234
                                    + pM[SA02]*mDet4_1235_0134 - pM[SA03]*mDet4_1235_0124 + pM[SA04]*mDet4_1235_0123;
   const Double_t mDet5_01235_01235 = pM[SA00]*mDet4_1235_1235 - pM[SA01]*mDet4_1235_0235
                                    + pM[SA02]*mDet4_1235_0135 - pM[SA03]*mDet4_1235_0125 + pM[SA05]*mDet4_1235_0123;
   const Double_t mDet5_01245_01234 = pM[SA00]*mDet4_1245_1234 - pM[SA01]*mDet4_1245_0234
                                    + pM[SA02]*mDet4_1245_0134 - pM[SA03]*mDet4_1245_0124 + pM[SA04]*mDet4_1245_0123;
   const Double_t mDet5_01245_01235 = pM[SA00]*mDet4_1245_1235 - pM[SA01]*mDet4_1245_0235
                                    + pM[SA02]*mDet4_1245_0135 - pM[SA03]*mDet4_1245_0125 + pM[SA05]*mDet4_1245_0123;
   const Double_t mDet5_01245_01245 = pM[SA00]*mDet4_1245_1245 - pM[SA01]*mDet4_1245_0245
                                    + pM[SA02]*mDet4_1245_0145 - pM[SA04]*mDet4_1245_0125 + pM[SA05]*mDet4_1245_0124;
   const Double_t mDet5_01345_01234 = pM[SA00]*mDet4_1345_1234 - pM[SA01]*mDet4_1345_0234
                                    + pM[SA02]*mDet4_1345_0134 - pM[SA03]*mDet4_1345_0124 + pM[SA04]*mDet4_1345_0123;
   const Double_t mDet5_01345_01235 = pM[SA00]*mDet4_1345_1235 - pM[SA01]*mDet4_1345_0235
                                    + pM[SA02]*mDet4_1345_0135 - pM[SA03]*mDet4_1345_0125 + pM[SA05]*mDet4_1345_0123;
   const Double_t mDet5_01345_01245 = pM[SA00]*mDet4_1345_1245 - pM[SA01]*mDet4_1345_0245
                                    + pM[SA02]*mDet4_1345_0145 - pM[SA04]*mDet4_1345_0125 + pM[SA05]*mDet4_1345_0124;
   const Double_t mDet5_01345_01345 = pM[SA00]*mDet4_1345_1345 - pM[SA01]*mDet4_1345_0345
                                    + pM[SA03]*mDet4_1345_0145 - pM[SA04]*mDet4_1345_0135 + pM[SA05]*mDet4_1345_0134;
   const Double_t mDet5_02345_01234 = pM[SA00]*mDet4_2345_1234 - pM[SA01]*mDet4_2345_0234
                                    + pM[SA02]*mDet4_2345_0134 - pM[SA03]*mDet4_2345_0124 + pM[SA04]*mDet4_2345_0123;
   const Double_t mDet5_02345_01235 = pM[SA00]*mDet4_2345_1235 - pM[SA01]*mDet4_2345_0235
                                    + pM[SA02]*mDet4_2345_0135 - pM[SA03]*mDet4_2345_0125 + pM[SA05]*mDet4_2345_0123;
   const Double_t mDet5_02345_01245 = pM[SA00]*mDet4_2345_1245 - pM[SA01]*mDet4_2345_0245
                                    + pM[SA02]*mDet4_2345_0145 - pM[SA04]*mDet4_2345_0125 + pM[SA05]*mDet4_2345_0124;
   const Double_t mDet5_02345_01345 = pM[SA00]*mDet4_2345_1345 - pM[SA01]*mDet4_2345_0345
                                    + pM[SA03]*mDet4_2345_0145 - pM[SA04]*mDet4_2345_0135 + pM[SA05]*mDet4_2345_0134;
   const Double_t mDet5_02345_02345 = pM[SA00]*mDet4_2345_2345 - pM[SA02]*mDet4_2345_0345
                                    + pM[SA03]*mDet4_2345_0245 - pM[SA04]*mDet4_2345_0235 + pM[SA05]*mDet4_2345_0234;
   const Double_t mDet5_12345_01234 = pM[SA10]*mDet4_2345_1234 - pM[SA11]*mDet4_2345_0234
                                    + pM[SA12]*mDet4_2345_0134 - pM[SA13]*mDet4_2345_0124 + pM[SA14]*mDet4_2345_0123;
   const Double_t mDet5_12345_01235 = pM[SA10]*mDet4_2345_1235 - pM[SA11]*mDet4_2345_0235
                                    + pM[SA12]*mDet4_2345_0135 - pM[SA13]*mDet4_2345_0125 + pM[SA15]*mDet4_2345_0123;
   const Double_t mDet5_12345_01245 = pM[SA10]*mDet4_2345_1245 - pM[SA11]*mDet4_2345_0245
                                    + pM[SA12]*mDet4_2345_0145 - pM[SA14]*mDet4_2345_0125 + pM[SA15]*mDet4_2345_0124;
   const Double_t mDet5_12345_01345 = pM[SA10]*mDet4_2345_1345 - pM[SA11]*mDet4_2345_0345
                                    + pM[SA13]*mDet4_2345_0145 - pM[SA14]*mDet4_2345_0135 + pM[SA15]*mDet4_2345_0134;
   const Double_t mDet5_12345_02345 = pM[SA10]*mDet4_2345_2345 - pM[SA12]*mDet4_2345_0345
                                    + pM[SA13]*mDet4_2345_0245 - pM[SA14]*mDet4_2345_0235 + pM[SA15]*mDet4_2345_0234;
   const Double_t mDet5_12345_12345 = pM[SA11]*mDet4_2345_2345 - pM[SA12]*mDet4_2345_1345
                                    + pM[SA13]*mDet4_2345_1245 - pM[SA14]*mDet4_2345_1235 + pM[SA15]*mDet4_2345_1234;

   // Find the determinant

   const Double_t det = pM[SA00]*mDet5_12345_12345 - pM[SA01]*mDet5_12345_02345 + pM[SA02]*mDet5_12345_01345
                      - pM[SA03]*mDet5_12345_01245 + pM[SA04]*mDet5_12345_01235 - pM[SA05]*mDet5_12345_01234;

   if (determ)
      *determ = det;

   if ( det == 0 ) {
      Error("Inv6x6","matrix is singular");
      return kFALSE;
   }

   const Double_t oneOverDet = 1.0/det;
   const Double_t mn1OverDet = - oneOverDet;

   pM[SA00] =  mDet5_12345_12345*oneOverDet;
   pM[SA01] =  mDet5_12345_02345*mn1OverDet;
   pM[SA02] =  mDet5_12345_01345*oneOverDet;
   pM[SA03] =  mDet5_12345_01245*mn1OverDet;
   pM[SA04] =  mDet5_12345_01235*oneOverDet;
   pM[SA05] =  mDet5_12345_01234*mn1OverDet;

   pM[SA11] =  mDet5_02345_02345*oneOverDet;
   pM[SA12] =  mDet5_02345_01345*mn1OverDet;
   pM[SA13] =  mDet5_02345_01245*oneOverDet;
   pM[SA14] =  mDet5_02345_01235*mn1OverDet;
   pM[SA15] =  mDet5_02345_01234*oneOverDet;

   pM[SA22] =  mDet5_01345_01345*oneOverDet;
   pM[SA23] =  mDet5_01345_01245*mn1OverDet;
   pM[SA24] =  mDet5_01345_01235*oneOverDet;
   pM[SA25] =  mDet5_01345_01234*mn1OverDet;

   pM[SA33] =  mDet5_01245_01245*oneOverDet;
   pM[SA34] =  mDet5_01245_01235*mn1OverDet;
   pM[SA35] =  mDet5_01245_01234*oneOverDet;

   pM[SA44] =  mDet5_01235_01235*oneOverDet;
   pM[SA45] =  mDet5_01235_01234*mn1OverDet;

   pM[SA55] =  mDet5_01234_01234*oneOverDet;

   for (Int_t irow = 0; irow < 6; irow++) {
      const Int_t rowOff1 = irow*6;
      for (Int_t icol = 0; icol < irow; icol++) {
         const Int_t rowOff2 = icol*6;
         pM[rowOff1+icol] = pM[rowOff2+irow];
      }
   }

   return kTRUE;
}

#ifndef ROOT_TMatrixFSymfwd
#include "TMatrixFSymfwd.h"
#endif

template Bool_t TMatrixTSymCramerInv::Inv2x2<Float_t>(TMatrixFSym&,Double_t*);
template Bool_t TMatrixTSymCramerInv::Inv3x3<Float_t>(TMatrixFSym&,Double_t*);
template Bool_t TMatrixTSymCramerInv::Inv4x4<Float_t>(TMatrixFSym&,Double_t*);
template Bool_t TMatrixTSymCramerInv::Inv5x5<Float_t>(TMatrixFSym&,Double_t*);
template Bool_t TMatrixTSymCramerInv::Inv6x6<Float_t>(TMatrixFSym&,Double_t*);

#ifndef ROOT_TMatrixDSymfwd
#include "TMatrixDSymfwd.h"
#endif

template Bool_t TMatrixTSymCramerInv::Inv2x2<Double_t>(TMatrixDSym&,Double_t*);
template Bool_t TMatrixTSymCramerInv::Inv3x3<Double_t>(TMatrixDSym&,Double_t*);
template Bool_t TMatrixTSymCramerInv::Inv4x4<Double_t>(TMatrixDSym&,Double_t*);
template Bool_t TMatrixTSymCramerInv::Inv5x5<Double_t>(TMatrixDSym&,Double_t*);
template Bool_t TMatrixTSymCramerInv::Inv6x6<Double_t>(TMatrixDSym&,Double_t*);
